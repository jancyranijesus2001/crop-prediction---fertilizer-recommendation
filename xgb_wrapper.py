"""
xgb_wrapper.py
──────────────
Shared wrapper so XGBoost model can be pickled/unpickled from both
retrain_models.py and app.py without a NameError.

Place this file in the SAME folder as app.py and retrain_models.py.
"""

class XGBWrapper:
    """
    Wraps an XGBClassifier so that predict() returns original
    crop-name strings instead of encoded integers.
    """
    def __init__(self, model, label_encoder):
        self._m   = model
        self._enc = label_encoder
        self.classes_ = label_encoder.classes_

    def predict(self, X):
        return self._enc.inverse_transform(self._m.predict(X))

    def predict_proba(self, X):
        return self._m.predict_proba(X)

    @property
    def feature_importances_(self):
        return self._m.feature_importances_
