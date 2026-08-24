# 🌱 Crop Prediction & Fertilizer Recommendation System

<p align="center">
  <b>Machine Learning Based Smart Agriculture Decision Support System</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Flask-Web%20Framework-black?logo=flask" alt="Flask">
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange" alt="Machine Learning">
  <img src="https://img.shields.io/badge/XGBoost-ML-red" alt="XGBoost">
  <img src="https://img.shields.io/badge/Explainable%20AI-SHAP-purple" alt="SHAP">
  <img src="https://img.shields.io/badge/Database-SQLite-lightgrey" alt="SQLite">
</p>

---

## 📌 Overview

**Crop Prediction & Fertilizer Recommendation System** is a web-based machine learning application developed using **Python and Flask**.

The system helps users identify a suitable crop based on soil and environmental parameters and provides fertilizer recommendations based on crop-specific nutrient requirements.

The crop recommendation module uses the following parameters:

* Nitrogen (N)
* Phosphorus (P)
* Potassium (K)
* Temperature
* Humidity
* Soil pH
* Rainfall

## The application also retrieves weather information based on the selected city and provides machine learning prediction results, model comparison, confidence information, and explainable AI visualizations.

## 🎯 Objectives

The main objectives of this project are:

* To recommend suitable crops using machine learning.
* To use soil and environmental parameters for crop prediction.
* To integrate weather information into the prediction process.
* To compare multiple machine learning models.
* To provide explainable AI-based prediction insights.
* To recommend fertilizers based on crop-specific NPK requirements.
* To maintain prediction history for registered users.
* To provide an administrator dashboard for managing users and prediction records.

---

# 🚀 Key Features

## 🌾 1. Crop Recommendation

Users can enter soil and environmental parameters to predict a suitable crop.

### Input Parameters

| Parameter      | Description                |
| -------------- | -------------------------- |
| Nitrogen (N)   | Nitrogen content in soil   |
| Phosphorus (P) | Phosphorus content in soil |
| Potassium (K)  | Potassium content in soil  |
| Temperature    | Environmental temperature  |
| Humidity       | Environmental humidity     |
| Soil pH        | Soil acidity/alkalinity    |
| Rainfall       | Rainfall measurement       |

The application combines these seven parameters and passes them to the selected machine learning model for prediction.

---

## 🤖 2. Multiple Machine Learning Models

The system supports multiple machine learning models:

* **Random Forest**
* **Decision Tree**
* **Support Vector Machine (SVM)**
* **XGBoost**

The application loads the available trained models from the `models/` directory and allows the user to select a model for prediction.

### Reported Model Accuracy

The current application contains the following reported accuracy values:

| Model         | Reported Accuracy |
| ------------- | ----------------: |
| Random Forest |             99.1% |
| Decision Tree |             90.3% |
| SVM           |             97.6% |
| XGBoost       |             98.7% |

> **Note:** These accuracy values are the values currently configured in the application and should be independently evaluated on the project's dataset before being presented as validated experimental results.

---

## 📊 3. Model Performance Comparison

The dashboard generates a model performance comparison chart for the loaded machine learning models.

This allows users to visually compare the reported accuracy of the available models.

---

## 🧠 4. Explainable AI (XAI)

The system includes an explainability component for crop predictions.

When supported, the application uses **SHAP TreeExplainer** to calculate feature contributions for the prediction.

The generated visualization shows the contribution/importance of the input features for the predicted crop.

### XAI Features

* SHAP-based feature contribution
* Feature importance visualization
* Predicted crop display
* Feature contribution chart
* Fallback feature importance for compatible tree models

---

## 📈 5. Input Parameter Radar Chart

The application generates a radar chart representing the user's input profile.

The chart visualizes:

```text
N
P
K
Temperature
Humidity
Soil pH
Rainfall
```

This provides a graphical representation of the soil and environmental conditions supplied for prediction.

---

# 🌤️ 6. Weather Integration

The application integrates weather information using a weather API.

The selected city is used to retrieve:

* Temperature
* Humidity

These values are then combined with the user's soil parameters before making the crop prediction.

### Weather Flow

```text
User selects city
       ↓
Weather API request
       ↓
Temperature + Humidity
       ↓
Combine with soil parameters
       ↓
Machine Learning Model
       ↓
Crop Prediction
```

If the weather request fails, the current application contains fallback temperature and humidity values.

---

# 🧪 7. Fertilizer Recommendation

The fertilizer module recommends fertilizer requirements based on the selected crop and current soil NPK values.

The application:

1. Receives the crop name.
2. Receives current NPK values.
3. Loads the crop recommendation dataset.
4. Finds the selected crop.
5. Retrieves the crop's ideal NPK values.
6. Calculates the nutrient differences.
7. Determines the nutrient with the largest difference.
8. Generates a fertilizer recommendation.
9. Displays an NPK comparison chart.

This workflow is implemented in the fertilizer prediction route.

---

# 👤 8. User Authentication

The system provides user authentication functionality.

### User Features

* User registration
* User login
* Password hashing
* Session management
* Protected pages
* Logout

## The application uses **Flask-Login** for authentication and **Flask-Bcrypt** for password hashing.

# 📋 9. Prediction History

The system stores crop prediction records in the database.

Each prediction log can contain:

* User ID
* Nitrogen
* Phosphorus
* Potassium
* Temperature
* Humidity
* Soil pH
* Rainfall
* Predicted crop
* Model used
* Date created

This allows users and administrators to maintain a history of crop prediction activities.

---

# 👨‍💼 10. Admin Dashboard

The application includes an administrator module.

### Admin Features

* Admin registration
* Admin login
* View registered users
* View contact messages
* View prediction logs

The administrator dashboard retrieves users, contact messages, and prediction history from the database.

---

# 🛠️ Technologies Used

## Backend

* Python
* Flask
* Flask-SQLAlchemy
* Flask-Login
* Flask-Bcrypt
* Flask-WTF

## Machine Learning

* Scikit-learn
* Random Forest
* Decision Tree
* Support Vector Machine
* XGBoost

## Explainable AI

* SHAP

## Data Processing

* NumPy
* Pandas

## Visualization

* Matplotlib

## Database

* SQLite
* SQLAlchemy

## Frontend

* HTML5
* CSS3
* JavaScript

## API

* Weather API

The application imports and uses Flask, SQLAlchemy, Flask-Login, WTForms, Bcrypt, NumPy, Pandas, Matplotlib, Requests, SHAP, and the project's trained model components.

---

# 📁 Project Structure

```text
crop-prediction---fertilizer-recommendation/
│
├── README.md
├── .gitignore
├── app.py
├── fertilizer.py
├── retrain_models.py
├── xgb_wrapper.py
├── version.txt
│
├── Data/
│   └── Crop_recommendation.csv
│
├── models/
│   ├── DecisionTree.pkl
│   ├── RandomForest.pkl
│   ├── SVM.pkl
│   └── XGBoost.pkl
│
├── static/
│   ├── css/
│   │   └── style.css
│   │
│   ├── images/
│   │   ├── 1.jpeg
│   │   ├── 2.jpeg
│   │   ├── 3.jpeg
│   │   └── ...
│   │
│   └── scripts/
│       └── cities.js
│
└── templates/
    ├── index.html
    ├── aboutus.html
    ├── contact.html
    ├── login.html
    ├── signup.html
    ├── dashboard.html
    ├── crop.html
    ├── crop-result.html
    ├── fertilizer.html
    ├── fertilizer-result.html
    ├── adminlogin.html
    ├── admindashboard.html
    ├── reg.html
    └── try_again.html
```

## The application expects the trained machine learning models inside `models/` and the crop dataset inside `Data/Crop_recommendation.csv`.

# 🔄 System Workflow

## Crop Recommendation Workflow

```text
                    ┌─────────────────┐
                    │      User       │
                    └────────┬────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ User Authentication │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Enter Soil Details  │
                  │ N, P, K, pH, Rain   │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Select City         │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Weather Information │
                  │ Temp + Humidity     │
                  └──────────┬──────────┘
                             │
                             ▼
                ┌─────────────────────────┐
                │ Machine Learning Model  │
                └───────────┬─────────────┘
                            │
                            ▼
                  ┌─────────────────────┐
                  │ Crop Prediction     │
                  └──────────┬──────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        Confidence       XAI/SHAP       Model Comparison
              │              │              │
              └──────────────┼──────────────┘
                             ▼
                  ┌─────────────────────┐
                  │ Save Prediction Log │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Display Result      │
                  └─────────────────────┘
```

---

# 🌱 Fertilizer Recommendation Workflow

```text
User
  ↓
Select Crop
  ↓
Enter Current NPK
  ↓
Load Crop Dataset
  ↓
Find Selected Crop
  ↓
Get Ideal NPK
  ↓
Compare Current vs Ideal
  ↓
Calculate NPK Difference
  ↓
Identify Major Nutrient Difference
  ↓
Generate Recommendation
  ↓
Display NPK Comparison Chart
```

---

# 📊 Dataset

The application uses:

```text
Data/Crop_recommendation.csv
```

The dataset is used by the fertilizer recommendation module to retrieve crop-specific NPK values.

The crop prediction model uses seven input features:

```text
Nitrogen
Phosphorus
Potassium
Temperature
Humidity
pH
Rainfall
```

---

# 💻 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/jancyranijesus2001/crop-prediction---fertilizer-recommendation.git
```

## 2. Open the Project

```bash
cd crop-prediction---fertilizer-recommendation
```

If the application code is inside a subdirectory, move into that directory before running the application.

---

## 3. Create a Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
```

Activate:

```bash
source venv/bin/activate
```

---

## 4. Install Dependencies

Create a `requirements.txt` file containing the packages required by the project and install them using:

```bash
pip install -r requirements.txt
```

The project requires Python packages corresponding to the Flask application, machine learning models, data processing, visualization, authentication, database, and XAI components.

---

# 🔑 Weather API Configuration

The application uses a weather API key through:

```python
config.weather_api_key
```

The weather helper sends a request to the weather service using the selected city and retrieves temperature and humidity.

### Important Security Note

**Do not commit API keys, passwords, secret keys, or other sensitive credentials to a public GitHub repository.**

For production/deployment, use environment variables.

Example:

```text
WEATHER_API_KEY=your_api_key_here
SECRET_KEY=your_secret_key_here
```

---

# ▶️ Run the Application

Run:

```bash
python app.py
```

The application is configured to run on:

```text
http://127.0.0.1:8000/
```

The Flask application creates the database tables when the application is started directly.

---

# 🖥️ Main Routes

| Route                 | Purpose                        |
| --------------------- | ------------------------------ |
| `/`                   | Home page                      |
| `/aboutus`            | About page                     |
| `/contact`            | Contact form                   |
| `/login`              | User login                     |
| `/signup`             | User registration              |
| `/dashboard`          | User dashboard                 |
| `/crop-recommend`     | Crop recommendation page       |
| `/crop-predict`       | Crop prediction                |
| `/fertilizer`         | Fertilizer recommendation page |
| `/fertilizer-predict` | Fertilizer prediction          |
| `/logout`             | User logout                    |
| `/AdminLogin`         | Admin login                    |
| `/admindashboard`     | Admin dashboard                |
| `/reg`                | Admin registration             |

## These routes are implemented in the Flask application.

# 🔐 Security Considerations

The project includes:

* Flask-Login authentication
* Password hashing using Bcrypt
* Protected prediction routes
* SQLAlchemy database integration
* Flask-WTF forms

However, before production deployment, the following should be improved:

* Store Flask secret keys in environment variables.
* Store weather API keys securely.
* Do not commit database files containing real user information.
* Disable Flask debug mode in production.
* Use a production-grade database for deployment.
* Validate and sanitize all user input.
* Use HTTPS in production.

---

# 📈 Machine Learning Prediction

The selected model receives the following input structure:

```text
[N, P, K, Temperature, Humidity, pH, Rainfall]
```

The model then generates the predicted crop.

The application also attempts to calculate a prediction confidence score when the selected model supports `predict_proba()`.

---

# 🧠 Explainable AI Architecture

```text
Input Parameters
       ↓
Machine Learning Model
       ↓
Crop Prediction
       ↓
SHAP TreeExplainer
       ↓
Feature Contributions
       ↓
XAI Visualization
```

If SHAP processing is unavailable, the application attempts to use model feature importance as a fallback.

---

# 📊 Prediction Result

The crop prediction result page can receive:

* Predicted crop
* Confidence score
* Selected model
* Model accuracy value
* XAI chart
* Radar chart
* Predictions from other available models
* Input parameter summary
* Selected city

---

# 🧪 Fertilizer Analysis

The fertilizer module compares:

```text
Current NPK
     VS
Ideal Crop NPK
```

Example:

```text
Current NPK
     ↓
N = Current Nitrogen
P = Current Phosphorus
K = Current Potassium

Ideal NPK
     ↓
N = Ideal Nitrogen
P = Ideal Phosphorus
K = Ideal Potassium
```

The application then calculates the differences and identifies the nutrient requiring attention.

---

# 🎓 Academic Applications

This project can be used as an academic demonstration of:

* Machine Learning
* Web Application Development
* Smart Agriculture
* Agricultural Decision Support Systems
* Explainable Artificial Intelligence
* Data Visualization
* Predictive Analytics
* Multiple Model Comparison

---

# 🔮 Future Enhancements

The project can be further enhanced with:

### 🌐 Deployment

* Deploy the Flask application to a cloud platform.
* Use PostgreSQL/MySQL instead of SQLite for production.
* Add CI/CD using GitHub Actions.

### 📱 Mobile Application

Develop an Android/iOS application that communicates with the Flask backend.

### 🌡️ IoT Integration

Integrate real-time sensors for:

* Soil moisture
* Soil temperature
* NPK values
* Humidity
* Environmental conditions

### 🛰️ Smart Agriculture

Future versions could integrate:

* Satellite data
* GPS-based recommendations
* Weather forecasting
* Soil mapping
* Agricultural analytics

### 🤖 Advanced AI

Potential improvements include:

* Deep learning models
* Ensemble learning
* Automated hyperparameter tuning
* Advanced explainability methods
* Crop disease detection
* Yield prediction

### 🌍 Multilingual Support

Support multiple regional languages to make the application more accessible to farmers.

---

# ⚠️ Disclaimer

This project is developed for **educational, research, and demonstration purposes**.

Machine learning predictions and fertilizer recommendations should not be treated as a replacement for professional agricultural advice. Real-world agricultural decisions should consider local soil conditions, crop variety, climate, farming practices, and expert recommendations.

---

# 👩‍💻 Developer

## Jancy Rani T

**Python Developer | Machine Learning | AI**

GitHub:

[GitHub Profile](https://github.com/jancyranijesus2001?utm_source=chatgpt.com)

Project Repository:

[Crop Prediction & Fertilizer Recommendation](https://github.com/jancyranijesus2001/crop-prediction---fertilizer-recommendation?utm_source=chatgpt.com)

---

# ⭐ Contribution

Contributions, suggestions, and improvements are welcome.

If you would like to contribute:

```bash
git clone https://github.com/jancyranijesus2001/crop-prediction---fertilizer-recommendation.git
```

Create a new branch:

```bash
git checkout -b feature/new-feature
```

Commit your changes:

```bash
git add .
git commit -m "Add new feature"
```

Push your branch:

```bash
git push origin feature/new-feature
```

Then create a Pull Request.

---

# ⭐ Support

If you find this project useful, please consider giving the repository a ⭐ on GitHub.

---

# 📄 License

This project is intended for **educational and research purposes**.

If an open-source license is added to the repository, the terms of that license will apply.
