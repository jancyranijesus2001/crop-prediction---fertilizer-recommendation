"""
retrain_models.py
─────────────────
Run this ONCE on YOUR machine to create sklearn-version-matched .pkl files.

    python retrain_models.py
    python app.py

Place this file in the same folder as app.py.
Dataset must be at:  Data/Crop_recommendation.csv
"""

import os, pickle, warnings, sys
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import sklearn
print(f"  scikit-learn version: {sklearn.__version__}")
print(f"  Python version      : {sys.version.split()[0]}\n")

# ─── 1. Load dataset ───────────────────────────────────────────────────────
DATA_PATH = os.path.join('Data', 'Crop_recommendation.csv')
if not os.path.exists(DATA_PATH):
    for alt in ['Data/crop_recommendation.csv', 'Data/crop.csv']:
        if os.path.exists(alt):
            DATA_PATH = alt
            break
    else:
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()

print(f"  Dataset : {DATA_PATH}")
print(f"  Shape   : {df.shape}")
print(f"  Crops   : {sorted(df['label'].unique())}\n")

# ─── 2. Prepare ────────────────────────────────────────────────────────────
FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
for col in FEATURES:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna()

X = df[FEATURES].values.astype(np.float64)
y = df['label'].to_numpy(dtype=str)

# Sanity check
from sklearn.ensemble import RandomForestClassifier as _RF
_chk = _RF(n_estimators=5, random_state=42)
_chk.fit(X, y)
sanity = accuracy_score(y, _chk.predict(X))
print(f"  Sanity check (overfit, expect >95%): {sanity*100:.1f}%")
if sanity < 0.5:
    print("  ERROR: Dataset has no learnable pattern. Check your CSV!")
    sys.exit(1)
print("  Data OK\n")

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {len(X_tr)}  |  Test: {len(X_te)}\n")

# ─── 3. Train & save ───────────────────────────────────────────────────────
os.makedirs('models', exist_ok=True)
results = {}

# --- Random Forest ---
print("  [1/4] Training Random Forest ...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)
acc = accuracy_score(y_te, rf.predict(X_te))
results['RandomForest'] = round(acc * 100, 2)
with open(os.path.join('models', 'RandomForest.pkl'), 'wb') as f:
    pickle.dump(rf, f)
print(f"       Accuracy : {results['RandomForest']}%  -> models/RandomForest.pkl")

# --- Decision Tree ---
print("  [2/4] Training Decision Tree ...")
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_tr, y_tr)
acc = accuracy_score(y_te, dt.predict(X_te))
results['DecisionTree'] = round(acc * 100, 2)
with open(os.path.join('models', 'DecisionTree.pkl'), 'wb') as f:
    pickle.dump(dt, f)
print(f"       Accuracy : {results['DecisionTree']}%  -> models/DecisionTree.pkl")

# --- SVM (with StandardScaler pipeline — critical for SVM performance) ---
print("  [3/4] Training SVM (with StandardScaler) ...")
svm_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm',    SVC(kernel='rbf', C=10, gamma=0.1, probability=True, random_state=42))
])
svm_pipe.fit(X_tr, y_tr)
acc = accuracy_score(y_te, svm_pipe.predict(X_te))
results['SVM'] = round(acc * 100, 2)
with open(os.path.join('models', 'SVM.pkl'), 'wb') as f:
    pickle.dump(svm_pipe, f)
print(f"       Accuracy : {results['SVM']}%  -> models/SVM.pkl")

# --- XGBoost (optional) ---
print("  [4/4] Training XGBoost ...")
try:
    from xgboost import XGBClassifier
    from xgb_wrapper import XGBWrapper

    le = LabelEncoder()
    y_tr_enc = le.fit_transform(y_tr)
    y_te_enc = le.transform(y_te)

    xgb = XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss')
    xgb.fit(X_tr, y_tr_enc)
    y_pred = le.inverse_transform(xgb.predict(X_te))
    acc = accuracy_score(y_te, y_pred)
    results['XGBoost'] = round(acc * 100, 2)

    wrapped = XGBWrapper(xgb, le)
    with open(os.path.join('models', 'XGBoost.pkl'), 'wb') as f:
        pickle.dump(wrapped, f)
    print(f"       Accuracy : {results['XGBoost']}%  -> models/XGBoost.pkl")
except ImportError:
    print("       XGBoost not installed — skipping.  (pip install xgboost)")
except Exception as e:
    print(f"       XGBoost error: {e} — skipping.")

# ─── 4. Summary ────────────────────────────────────────────────────────────
print("\n" + "=" * 52)
print("  TRAINING COMPLETE")
print("=" * 52)
for name, acc in results.items():
    bar = chr(9608) * int(acc / 5)
    print(f"  {name:<15} {acc:>6.2f}%  {bar}")
print("=" * 52)
print(f"\n  sklearn {sklearn.__version__} .pkl files saved to  models/")
print("  Now run:  python app.py\n")