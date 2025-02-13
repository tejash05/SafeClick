import os
import re
import joblib
import pandas as pd

def extract_url_features(url):
    """Extracts numerical features from a URL for phishing detection."""
    suspicious_words = ["secure", "login", "verify", "update", "free", "gift", "money", "account"]
    return {
        "url_length": len(url),
        "num_hyphens": url.count("-"),
        "num_underscores": url.count("_"),
        "num_slashes": url.count("/"),
        "num_digits": sum(c.isdigit() for c in url),
        "num_subdomains": url.count(".") - 1,
        "contains_ip": 1 if re.match(r"\d+\.\d+\.\d+\.\d+", url) else 0,
        "num_special_chars": sum(url.count(c) for c in "!@#$%^&*()+=[]{}|\\:;\"'<>,?"),
        "digit_ratio": sum(c.isdigit() for c in url) / len(url) if len(url) > 0 else 0,
        "special_char_ratio": sum(url.count(c) for c in "!@#$%^&*()+=[]{}|\\:;\"'<>,?") / len(url) if len(url) > 0 else 0,
        "suspicious_word_count": sum(1 for word in suspicious_words if word in url)
    }

# 🚀 Load the trained XGBoost model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "final_optimized_xgboost_model.pkl")
try:
    phishing_model = joblib.load(MODEL_PATH)
    print("✅ Phishing Model Loaded Successfully")
except FileNotFoundError:
    print(f"❌ Model file not found at {MODEL_PATH}")
    exit(1)
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

def check_phishing(url):
    """Predicts whether a given URL is phishing or safe."""
    features = extract_url_features(url)
    df_features = pd.DataFrame([features])
    prediction = phishing_model.predict(df_features)[0]
    return "Malicious" if prediction == 1 else "Benign"

