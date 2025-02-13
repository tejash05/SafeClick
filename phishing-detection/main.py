import os
from flask import Flask, request, jsonify
from phishing_detection import check_phishing  # Import phishing detection function

# 🚀 Flask API Setup
app = Flask(__name__)

@app.route("/check", methods=["POST"])
def check_website():
    """API endpoint to check if a URL is phishing."""
    data = request.json
    url = data.get("url")
    if not url:
        return jsonify({"error": "URL is required"}), 400
    phishing_result = check_phishing(url)
    return jsonify({"url": url, "phishing_result": phishing_result})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))  # Default to 8000 if PORT is not set
    app.run(host="0.0.0.0", port=port)

