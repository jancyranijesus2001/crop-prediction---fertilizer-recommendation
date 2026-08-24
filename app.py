import io
import base64
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from flask import Flask, redirect, render_template, url_for, request, jsonify
from markupsafe import Markup
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError
from flask_wtf import FlaskForm
from flask_bcrypt import Bcrypt
from datetime import datetime
import requests
import config
# import torch
# from torchvision import transforms
# from PIL import Image
# ResNet9 removed - disease detection not used
from fertilizer import fertilizer_dic
from xgb_wrapper import XGBWrapper  # shared wrapper for XGBoost pickle

# ─────────────────────────────────────────────
#  LOAD TRAINED MODELS
# ─────────────────────────────────────────────

# Primary recommendation model (Random Forest)
crop_recommendation_model_path = 'models/RandomForest.pkl'
crop_recommendation_model = pickle.load(open(crop_recommendation_model_path, 'rb'))

# ── Optional: load additional models if they exist ──
# Place DecisionTree.pkl / SVM.pkl / XGBoost.pkl in the models/ folder.
# The app will gracefully skip any that are missing.
import os

def _try_load(path):
    try:
        return pickle.load(open(path, 'rb'))
    except FileNotFoundError:
        return None

dt_model  = _try_load('models/DecisionTree.pkl')
svm_model = _try_load('models/SVM.pkl')
xgb_model = _try_load('models/XGBoost.pkl')

# Map of model name → (model_object, reported_accuracy %)
# Update accuracy values after you evaluate your own models.
ALL_MODELS = {
    'Random Forest':   (crop_recommendation_model, 99.1),
    'Decision Tree':   (dt_model,  90.3) if dt_model  else None,
    'SVM':             (svm_model, 97.6) if svm_model else None,
    'XGBoost':         (xgb_model, 98.7) if xgb_model else None,
}
# Keep only loaded models
ALL_MODELS = {k: v for k, v in ALL_MODELS.items() if v is not None}

FEATURE_NAMES = ['Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)',
                 'Temperature (°C)', 'Humidity (%)', 'Soil pH', 'Rainfall (mm)']

# ─────────────────────────────────────────────
#  XAI  –  Feature-importance explanation
# ─────────────────────────────────────────────

def get_shap_explanation(model, input_data):
    """
    Returns base64 PNG of a SHAP-style bar chart.
    Uses sklearn feature_importances_ when a tree model is available,
    otherwise falls back to a signed-contribution estimate.
    """
    try:
        import shap
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_data)

        # shap_values may be list (multiclass) → pick predicted class
        pred_class = model.predict(input_data)[0]
        classes    = list(model.classes_)
        if isinstance(shap_values, list):
            class_idx = classes.index(pred_class)
            sv = shap_values[class_idx][0]
        else:
            sv = shap_values[0]

        importance = sv
        colors     = ['#2ecc71' if v >= 0 else '#e74c3c' for v in importance]
        title      = f'SHAP Feature Contributions  →  Predicted: {pred_class}'
    except Exception:
        # Fallback: use tree feature_importances_ (always positive)
        try:
            fi     = model.feature_importances_
        except AttributeError:
            fi = np.ones(len(FEATURE_NAMES)) / len(FEATURE_NAMES)

        pred_class = model.predict(input_data)[0]
        importance = fi
        colors     = ['#3498db'] * len(FEATURE_NAMES)
        title      = f'Feature Importance  →  Predicted: {pred_class}'

    # ── Draw chart ──────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#f8f9fa')

    y_pos = np.arange(len(FEATURE_NAMES))
    bars  = ax.barh(y_pos, importance, color=colors, edgecolor='white', height=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(FEATURE_NAMES, fontsize=10, fontweight='bold')
    ax.set_xlabel('Impact on Prediction', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=12)
    ax.axvline(0, color='grey', linewidth=0.8, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar, val in zip(bars, importance):
        ax.text(val + (max(abs(importance)) * 0.02),
                bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=8, color='#333')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def get_model_comparison_chart():
    """Bar chart of all loaded model accuracies."""
    names  = list(ALL_MODELS.keys())
    accs   = [v[1] for v in ALL_MODELS.values()]
    colors = ['#27ae60', '#3498db', '#e67e22', '#9b59b6'][:len(names)]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#f8f9fa')

    bars = ax.bar(names, accs, color=colors, edgecolor='white', width=0.5)
    ax.set_ylim(80, 101)
    ax.set_ylabel('Accuracy (%)', fontsize=10)
    ax.set_title('Model Performance Comparison', fontsize=11, fontweight='bold', pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                f'{acc}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def get_radar_chart(input_values):
    """Spider/radar chart showing the user's soil/climate profile."""
    labels = ['N', 'P', 'K', 'Temp', 'Humidity', 'pH', 'Rainfall']
    # Normalise to 0–1 for display
    maxes  = [140, 145, 205, 50, 100, 14, 300]
    values = [min(v / m, 1.0) for v, m in zip(input_values, maxes)]
    values += values[:1]

    angles = [n / float(len(labels)) * 2 * np.pi for n in range(len(labels))]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#eafaf1')

    ax.plot(angles, values, 'o-', linewidth=2, color='#27ae60')
    ax.fill(angles, values, alpha=0.25, color='#27ae60')
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=9, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_title('Input Parameter Profile', size=10, fontweight='bold', pad=15)
    ax.grid(color='white')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# ─────────────────────────────────────────────
#  WEATHER HELPER
# ─────────────────────────────────────────────
import requests

def weather_fetch(city_name):
    api_key = config.weather_api_key
    base_url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "q": city_name,
        "appid": api_key
    }

    try:
        response = requests.get(base_url, params=params, timeout=5)
        x = response.json()

        # SUCCESS only when code == 200
        if x.get("cod") != 200:
            print("Weather API returned error:", x)
            return 25, 60   # fallback Tamil Nadu climate

        y = x.get("main", {})
        temperature = round((y.get("temp", 298) - 273.15), 2)
        humidity = y.get("humidity", 60)

        return temperature, humidity

    except Exception as e:
        print("Weather fetch failed:", e)
        return 25, 60

# ─────────────────────────────────────────────
#  FLASK APP SETUP
# ─────────────────────────────────────────────

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
app.config["SECRET_KEY"] = 'thisissecretkey'
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db     = SQLAlchemy(app)
bcrypt = Bcrypt(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# ─────────────────────────────────────────────
#  DATABASE MODELS
# ─────────────────────────────────────────────

class User(db.Model, UserMixin):
    id       = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)

class UserAdmin(db.Model, UserMixin):
    id       = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)

class ContactUs(db.Model):
    sno          = db.Column(db.Integer, primary_key=True)
    name         = db.Column(db.String(200), nullable=False)
    email        = db.Column(db.String(500), nullable=False)
    text         = db.Column(db.String(900), nullable=False)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"{self.sno} - {self.name}"

# Prediction history log
class PredictionLog(db.Model):
    id          = db.Column(db.Integer, primary_key=True)
    user_id     = db.Column(db.Integer, db.ForeignKey('user.id'))
    nitrogen    = db.Column(db.Float)
    phosphorus  = db.Column(db.Float)
    potassium   = db.Column(db.Float)
    temperature = db.Column(db.Float)
    humidity    = db.Column(db.Float)
    ph          = db.Column(db.Float)
    rainfall    = db.Column(db.Float)
    prediction  = db.Column(db.String(100))
    model_used  = db.Column(db.String(50))
    date_created = db.Column(db.DateTime, default=datetime.utcnow)


# ─────────────────────────────────────────────
#  FORMS
# ─────────────────────────────────────────────

class RegisterForm(FlaskForm):
    username = StringField(validators=[InputRequired(), Length(min=5, max=20)],
                           render_kw={"placeholder": "username"})
    password = PasswordField(validators=[InputRequired(), Length(min=5, max=20)],
                             render_kw={"placeholder": "password"})
    submit   = SubmitField("Register")

    def validate_username(self, username):
        existing = User.query.filter_by(username=username.data).first()
        if existing:
            raise ValidationError("That username already exists. Please choose a different one.")

class LoginForm(FlaskForm):
    username = StringField(validators=[InputRequired(), Length(min=5, max=20)],
                           render_kw={"placeholder": "username"})
    password = PasswordField(validators=[InputRequired(), Length(min=5, max=20)],
                             render_kw={"placeholder": "password"})
    submit   = SubmitField("Login")


# ─────────────────────────────────────────────
#  BASIC ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def hello_world():
    return render_template("index.html")

@app.route("/aboutus")
def aboutus():
    return render_template("aboutus.html")

@app.route("/contact", methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name    = request.form['name']
        email   = request.form['email']
        text    = request.form['text']
        contact = ContactUs(name=name, email=email, text=text)
        db.session.add(contact)
        db.session.commit()
    return render_template("contact.html")

@app.route("/login", methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    elif form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user)
            return redirect(url_for('dashboard'))
    return render_template("login.html", form=form)

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    comparison_chart = get_model_comparison_chart()
    model_info = [{'name': k, 'accuracy': v[1]} for k, v in ALL_MODELS.items()]
    recent_predictions = PredictionLog.query.filter_by(
        user_id=current_user.id).order_by(
        PredictionLog.date_created.desc()).limit(5).all()
    return render_template('dashboard.html',
                           comparison_chart=comparison_chart,
                           model_info=model_info,
                           recent_predictions=recent_predictions)

@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('hello_world'))

@app.route("/signup", methods=['GET', 'POST'])
def signup():
    form = RegisterForm()
    if form.validate_on_submit():
        hashed  = bcrypt.generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, password=hashed)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template("signup.html", form=form)


# ─────────────────────────────────────────────
#  CROP RECOMMENDATION
# ─────────────────────────────────────────────

@app.route('/crop-recommend')
@login_required
def crop_recommend():
    model_names = list(ALL_MODELS.keys())
    return render_template('crop.html', model_names=model_names)


@app.route('/crop-predict', methods=['POST'])
@login_required
def crop_prediction():

    if request.method == 'POST':

        # ── Form Inputs ─────────────────────────────
        N        = float(request.form['nitrogen'])
        P        = float(request.form['phosphorous'])
        K        = float(request.form['pottasium'])
        ph       = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        city     = request.form.get("city")
        selected_model_name = request.form.get("model_choice", "Random Forest")

        # ── Weather Fetch (SAFE) ────────────────────
        temperature, humidity = weather_fetch(city)

        # if API totally failed (very rare)
        if temperature is None or humidity is None:
            return render_template('try_again.html')

        # ── Prepare Model Input ─────────────────────
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        input_values = [N, P, K, temperature, humidity, ph, rainfall]

        # ── Choose ML Model ─────────────────────────
        if selected_model_name in ALL_MODELS:
            chosen_model, model_acc = ALL_MODELS[selected_model_name]
        else:
            chosen_model, model_acc = ALL_MODELS['Random Forest']
            selected_model_name = 'Random Forest'

        final_prediction = chosen_model.predict(input_data)[0]

        # ── XAI Graphs ──────────────────────────────
        xai_chart   = get_shap_explanation(chosen_model, input_data)
        radar_chart = get_radar_chart(input_values)

        # ── Compare All Models ──────────────────────
        all_preds = {}
        for mname, (mobj, macc) in ALL_MODELS.items():
            try:
                all_preds[mname] = {
                    'pred': mobj.predict(input_data)[0],
                    'acc': macc
                }
            except Exception:
                all_preds[mname] = {
                    'pred': '—',
                    'acc': macc
                }

        # ── Confidence Score ────────────────────────
        confidence = None
        try:
            proba    = chosen_model.predict_proba(input_data)[0]
            classes  = list(chosen_model.classes_)
            pred_idx = classes.index(final_prediction)
            confidence = round(proba[pred_idx] * 100, 2)
        except Exception:
            pass

        # ── Save to Database ────────────────────────
        log = PredictionLog(
            user_id=current_user.id,
            nitrogen=N,
            phosphorus=P,
            potassium=K,
            temperature=temperature,
            humidity=humidity,
            ph=ph,
            rainfall=rainfall,
            prediction=final_prediction,
            model_used=selected_model_name
        )
        db.session.add(log)
        db.session.commit()

        # ── Input Summary ───────────────────────────
        input_summary = {
            'Nitrogen (N)': N,
            'Phosphorus (P)': P,
            'Potassium (K)': K,
            'Temperature (°C)': temperature,
            'Humidity (%)': humidity,
            'Soil pH': ph,
            'Rainfall (mm)': rainfall,
        }

        # ── Render Result Page ──────────────────────
        return render_template(
            'crop-result.html',
            prediction     = final_prediction,
            confidence     = confidence,
            model_name     = selected_model_name,
            model_accuracy = model_acc,
            xai_chart      = xai_chart,
            radar_chart    = radar_chart,
            all_preds      = all_preds,
            input_summary  = input_summary,
            city           = city,
        )

    return redirect(url_for('crop_recommend'))

# ─────────────────────────────────────────────
#  FERTILIZER
# ─────────────────────────────────────────────

@app.route('/fertilizer')
@login_required
def fertilizer_recommendation():
    return render_template('fertilizer.html')


@app.route('/fertilizer-predict', methods=['POST'])
@login_required
def fert_recommend():

    # ── User Input ─────────────────────────────
    crop_name = str(request.form['cropname']).lower().strip()
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])

    # ── Load Dataset SAFELY ─────────────────────
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, 'Data', 'Crop_recommendation.csv')

    df = pd.read_csv(file_path)

    # dataset uses 'label' column
    crop_row = df[df['label'].str.lower() == crop_name]

    if crop_row.empty:
        return render_template('try_again.html')

    # Ideal NPK values
    nr = int(crop_row.iloc[0]['N'])
    pr = int(crop_row.iloc[0]['P'])
    kr = int(crop_row.iloc[0]['K'])

    # ── Calculate Difference ────────────────────
    n = nr - N
    p = pr - P
    k = kr - K

    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]

    if max_value == "N":
        key = 'NHigh' if n < 0 else 'Nlow'
    elif max_value == "P":
        key = 'PHigh' if p < 0 else 'Plow'
    else:
        key = 'KHigh' if k < 0 else 'Klow'

    # ── NPK Gap Chart ───────────────────────────
    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor('#f8f9fa')

    labels = ['Nitrogen', 'Phosphorus', 'Potassium']
    current = [N, P, K]
    ideal = [nr, pr, kr]
    x = np.arange(len(labels))

    ax.bar(x - 0.2, ideal, 0.35, label='Ideal', color='#27ae60', edgecolor='white')
    ax.bar(x + 0.2, current, 0.35, label='Current', color='#3498db', edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight='bold')
    ax.set_ylabel('Value')
    ax.set_title(f'Soil NPK vs Ideal for {crop_name.title()}', fontweight='bold')
    ax.legend()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    plt.close()
    buf.seek(0)

    npk_chart = base64.b64encode(buf.read()).decode('utf-8')

    # ── Recommendation Text ─────────────────────
    response = Markup(str(fertilizer_dic[key]))

    return render_template(
        'fertilizer-result.html',
        recommendation=response,
        npk_chart=npk_chart,
        crop_name=crop_name.title()
    )


# ─────────────────────────────────────────────
#  ADMIN
# ─────────────────────────────────────────────

@app.route("/AdminLogin", methods=['GET', 'POST'])
def AdminLogin():
    form = LoginForm()
    if current_user.is_authenticated:
        return redirect(url_for('admindashboard'))
    elif form.validate_on_submit():
        user = UserAdmin.query.filter_by(username=form.username.data).first()
        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user)
            return redirect(url_for('admindashboard'))
    return render_template("adminlogin.html", form=form)


@app.route("/admindashboard")
@login_required
def admindashboard():
    alltodo   = ContactUs.query.all()
    alluser   = User.query.all()
    all_logs  = PredictionLog.query.order_by(PredictionLog.date_created.desc()).all()
    return render_template("admindashboard.html",
                           alltodo=alltodo,
                           alluser=alluser,
                           all_logs=all_logs)


@app.route("/reg", methods=['GET', 'POST'])
def reg():
    form = RegisterForm()
    if form.validate_on_submit():
        hashed   = bcrypt.generate_password_hash(form.password.data)
        new_user = UserAdmin(username=form.username.data, password=hashed)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('AdminLogin'))
    return render_template("reg.html", form=form)


if __name__ == "__main__":
    with app.app_context():
        db.create_all()   # Creates all tables (user, user_admin, contact_us, prediction_log)
        print("✅ Database tables created successfully.")
    app.run(debug=True, port=8000)