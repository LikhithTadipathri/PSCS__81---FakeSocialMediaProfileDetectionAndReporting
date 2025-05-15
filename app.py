from flask import Flask, render_template, request, redirect, session, flash, make_response
from flask_bcrypt import Bcrypt
import sqlite3
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
import joblib
import traceback
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


app = Flask(__name__)
app.secret_key = 'your_secure_key_12345'
app.config['SESSION_PERMANENT'] = False
bcrypt = Bcrypt(app)

# Load LSTM model and tokenizer
model = load_model('lstm_model.h5')
tokenizer = joblib.load('tokenizer.pkl')

def get_db_connection():
    conn = sqlite3.connect('profiles.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    try:
        conn = get_db_connection()
        c = conn.cursor()
        
        # Create users table
        c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )''')
        
        # Create profiles table
        c.execute('''CREATE TABLE IF NOT EXISTS profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    profile_username TEXT NOT NULL,
                    bio TEXT NOT NULL,
                    followers INTEGER NOT NULL,
                    following INTEGER NOT NULL,
                    prediction REAL NOT NULL,
                    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )''')
        
        conn.commit()
        print("Database initialized successfully")
    except sqlite3.Error as e:
        print(f"Database error: {str(e)}")
        raise
    finally:
        conn.close()

init_db()

def preprocess_input(bio, followers, following):
    try:
        # Text processing
        seq = tokenizer.texts_to_sequences([bio])
        padded = pad_sequences(seq, maxlen=50)
        
        # Numerical normalization
        nums = np.array([[followers, following]]) / np.array([10000.0, 5000.0])
        
        return padded, nums
    except Exception as e:
        print(f"Preprocessing error: {str(e)}")
        raise

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            username = request.form.get('username', '').strip()
            password = request.form.get('password', '').strip()
            
            if not username or not password:
                flash('All fields are required', 'error')
                return render_template('register.html')
            
            if len(password) < 8:
                flash('Password must be at least 8 characters', 'error')
                return render_template('register.html')
            
            hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')
            
            with get_db_connection() as conn:
                conn.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                            (username, hashed_pw))
                conn.commit()
            
            flash('Registration successful! Please login', 'success')
            return redirect('/')
        except sqlite3.IntegrityError:
            flash('Username already exists', 'error')
        except Exception as e:
            traceback.print_exc()
            flash('Registration failed. Please try again.', 'error')
    
    return render_template('register.html')

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        try:
            username = request.form.get('username', '').strip()
            password = request.form.get('password', '').strip()
            
            if not username or not password:
                flash('All fields are required', 'error')
                return render_template('login.html')
            
            with get_db_connection() as conn:
                user = conn.execute("SELECT * FROM users WHERE username = ?", 
                                   (username,)).fetchone()
            
            if user and bcrypt.check_password_hash(user['password'], password):
                session.clear()
                session['user_id'] = user['id']
                session['username'] = user['username']
                session['logged_in'] = True
                return redirect('/dashboard')
            
            flash('Invalid credentials', 'error')
        except Exception as e:
            traceback.print_exc()
            flash('Login failed. Please try again.', 'error')
    
    return render_template('login.html')

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if not session.get('logged_in'):
        return redirect('/')
    
    if request.method == 'POST':
        try:
            form_data = {
                'username': request.form.get('username', '').strip(),
                'bio': request.form.get('bio', '').strip(),
                'followers': int(request.form.get('followers', 0)),
                'following': int(request.form.get('following', 0))
            }
            
            if not all(form_data.values()):
                flash('All fields are required', 'error')
                return render_template('dashboard.html')
            
            text_input, num_input = preprocess_input(
                form_data['bio'],
                form_data['followers'],
                form_data['following']
            )
            prediction = model.predict([text_input, num_input])[0][0]
            is_fake = prediction > 0.5
            
            with get_db_connection() as conn:
                result = conn.execute('''INSERT INTO profiles 
                            (user_id, profile_username, bio, followers, following, prediction)
                            VALUES (?, ?, ?, ?, ?, ?)''',
                         (session['user_id'],
                          form_data['username'],
                          form_data['bio'],
                          form_data['followers'],
                          form_data['following'],
                          float(prediction)))
                report_id = result.lastrowid
                conn.commit()
            
            return render_template('result.html',
                                 prediction=round(prediction*100, 2),
                                 is_fake=is_fake,
                                 report_id=report_id)
        except ValueError:
            flash('Invalid numerical input', 'error')
        except Exception as e:
            traceback.print_exc()
            flash('Analysis failed. Please check inputs and try again.', 'error')
    
    return render_template('dashboard.html')

@app.route('/download-report/<int:report_id>')
def download_report(report_id):
    try:
        with get_db_connection() as conn:
            report = conn.execute('''SELECT * FROM profiles WHERE id = ?''', 
                                (report_id,)).fetchone()
        
        if not report or report['user_id'] != session.get('user_id'):
            return redirect('/dashboard')
        
        # Create PDF
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []
        
        # Title with proper ParagraphStyle import
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Title'],
            fontSize=18,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=20
        )
        elements.append(Paragraph("Social Media Profile Analysis Report", title_style))
        
        # Content Table
        data = [
            ['Field', 'Value'],
            ['Profile Username', report['profile_username']],
            ['Followers', report['followers']],
            ['Following', report['following']],
            ['Bio Excerpt', report['bio'][:100] + '...'],
            ['Prediction Confidence', f"{report['prediction']*100:.1f}%"],
            ['Verdict', 'Fake Profile' if report['prediction'] > 0.5 else 'Genuine Profile']
        ]
        
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 12),
            ('BOTTOMPADDING', (0,0), (-1,0), 12),
            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#f8f9fa')),
            ('GRID', (0,0), (-1,-1), 1, colors.HexColor('#dfdfdf')),
        ]))
        elements.append(table)
        
        doc.build(elements)
        buffer.seek(0)
        
        response = make_response(buffer.getvalue())
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = \
            f'attachment; filename=report_{report_id}.pdf'
        return response
        
    except Exception as e:
        traceback.print_exc()
        return redirect('/dashboard')

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)