# from flask import Flask, request, render_template
# import joblib
# import pandas as pd
# from feature_extraction import extract_features, highlight_phishing_indicators

# app = Flask(__name__)
# model = joblib.load("phishing_model.pkl")

# @app.route("/", methods=["GET", "POST"])
# def index():
#     prediction = None
#     highlighted_email = ""
#     if request.method == "POST":
#         email = request.form["email"]
#         features = extract_features(email)
#         df = pd.DataFrame([features])
#         pred = model.predict(df)[0]
#         prediction = "Phishing" if pred == 1 else "Not Phishing"
#         highlighted_email = highlight_phishing_indicators(email)

#     return render_template("index.html", prediction=prediction, highlighted_email=highlighted_email)

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, request, render_template
import pandas as pd
from feature_extraction import extract_features, highlight_phishing_indicators

app = Flask(__name__)

# --- Mock prediction logic ---
def mock_predict(features_df):
    # Mock rule: If the number of suspicious words > 1 or num_links > 0, label as phishing
    if features_df["num_suspicious_words"][0] > 1 or features_df["num_links"][0] > 0:
        return [1]
    return [0]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    highlighted_email = ""
    if request.method == "POST":
        email = request.form["email"]
        features = extract_features(email)
        df = pd.DataFrame([features])
        pred = mock_predict(df)[0]
        prediction = "Phishing" if pred == 1 else "Not Phishing"
        highlighted_email = highlight_phishing_indicators(email)

    return render_template("index.html", prediction=prediction, highlighted_email=highlighted_email)

if __name__ == "__main__":
    app.run(debug=True)
