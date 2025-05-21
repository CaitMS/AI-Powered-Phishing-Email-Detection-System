from flask import Flask, request, render_template, jsonify
import pickle
import shap
import numpy as np
import pandas as pd
import re
import email
from email import policy
from email.parser import BytesParser
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import io
import base64
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Load the model pipeline
try:
    with open('models/phishing_detection_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully")
    print(model)

except FileNotFoundError:
    print("Model file not found. Using a dummy model for demonstration.")
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from xgboost import XGBClassifier
    model = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('xgb', XGBClassifier())
    ])
    model.fit(["test"], [0])

def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if w not in stop_words]
    return " ".join(filtered_tokens)

def parse_email_file(file_content):
    try:
        msg = BytesParser(policy=policy.default).parse(io.BytesIO(file_content))
        subject = msg.get('subject', '')
        date = msg.get('date', '')
        sender = msg.get('from', '')
        body = ""
        if msg.is_multipart():
            for part in msg.iter_parts():
                content_type = part.get_content_type()
                if content_type in ['text/plain', 'text/html']:
                    body += part.get_content()
        else:
            body = msg.get_content()
        return {'subject': subject, 'date': date, 'sender': sender, 'body': body}
    except Exception as e:
        print(f"Error parsing email: {e}")
        return {'subject': '', 'date': '', 'sender': '', 'body': str(file_content)}

def extract_features(email_data):
    combined_text = f"{email_data['sender']} {email_data['subject']} {email_data['body']} {email_data['date']}"
    return preprocess_text(combined_text)

def get_shap_explanation(text_input):
    # Vectorize the input for SHAP
    tfidf = model.named_steps['tfidf']
    vector_input = tfidf.transform([text_input])
    classifier = model.named_steps['xgb']
    
    # Create explainer using the classifier
    explainer = shap.Explainer(classifier)
    shap_values = explainer(vector_input)
    
    # Get feature names (words) from the vectorizer
    feature_names = np.array(tfidf.get_feature_names_out())
    
    # Get non-zero indices from the sparse matrix
    non_zero_indices = vector_input.nonzero()[1]
    
    # Extract the words that are present in this specific text
    present_words = feature_names[non_zero_indices]
    
    # Extract SHAP values for these words
    present_shap_values = shap_values.values[0, non_zero_indices]
    
    # Create a DataFrame for sorting
    word_importance_df = pd.DataFrame({
        'word': present_words,
        'importance': present_shap_values
    })
    
    # Sort by absolute importance
    word_importance_df['abs_importance'] = word_importance_df['importance'].abs()
    word_importance_df = word_importance_df.sort_values('abs_importance', ascending=False)
    
    # Take top 20 features for visualization
    top_features = word_importance_df.head(20)
    
    plt.figure(figsize=(10, 6))
    colors = ['red' if x > 0 else 'blue' for x in top_features['importance']]
    plt.barh(top_features['word'], top_features['importance'], color=colors)
    plt.xlabel('SHAP Impact on Prediction')
    plt.title('Top Words Influencing Phishing Prediction')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    # Red indicates pushing towards phishing, blue against
    plt.text(0.95, 0.05, 'Red: Increases phishing likelihood\nBlue: Decreases phishing likelihood', 
             transform=plt.gca().transAxes, ha='right', fontsize=9, bbox=dict(facecolor='white', alpha=0.8))
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    # Convert to dictionary for later use in highlighting
    word_importance = dict(zip(top_features['word'], top_features['importance']))
    
    return {
        'plot': img_str,
        'importance': word_importance
    }

def highlight_suspicious_content(text, word_importance):
    highlighted_text = text
    
    # Words that suggest phishing (positive SHAP values)
    phishing_words = [word for word, importance in word_importance.items() 
                     if importance > 0]
    
    # Words that suggest legitimacy (negative SHAP values)
    legitimate_words = [word for word, importance in word_importance.items() 
                       if importance < 0]
    
    # Highlight phishing indicators in red
    if phishing_words:
        pattern = r'\b(' + '|'.join(re.escape(word) for word in phishing_words) + r')\b'
        highlighted_text = re.sub(
            pattern,
            r'<span class="highlight-phishing">\1</span>',
            highlighted_text,
            flags=re.IGNORECASE
        )
    
    # Highlight legitimacy indicators in green
    if legitimate_words:
        pattern = r'\b(' + '|'.join(re.escape(word) for word in legitimate_words) + r')\b'
        highlighted_text = re.sub(
            pattern,
            r'<span class="highlight-legitimate">\1</span>',
            highlighted_text,
            flags=re.IGNORECASE
        )
    
    # Also highlight URLs as they're common in phishing
    highlighted_text = re.sub(
        r'(https?://\S+)',
        r'<span class="highlight-phishing">\1</span>',
        highlighted_text
    )
    
    return highlighted_text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/styles.css')
def styles():
    css = """
    .highlight-phishing {
        background-color: rgba(255, 99, 71, 0.3);
        padding: 2px;
        border-radius: 3px;
        font-weight: bold;
        border-bottom: 2px solid #FF6347;
    }
    
    .highlight-legitimate {
        background-color: rgba(50, 205, 50, 0.3);
        padding: 2px;
        border-radius: 3px;
        font-weight: bold;
        border-bottom: 2px solid #32CD32;
    }
    """
    return css, 200, {'Content-Type': 'text/css'}

@app.route('/analyse', methods=['POST'])
def analyse():
    try:
        if 'email_file' in request.files and request.files['email_file'].filename:
            file = request.files['email_file']
            file_content = file.read()
            email_data = parse_email_file(file_content)
        else:
            email_data = {
                'subject': request.form.get('subject', ''),
                'date': request.form.get('date', ''),
                'sender': request.form.get('sender', ''),
                'body': request.form.get('body', '')
            }

        preprocessed_text = extract_features(email_data)
        X_input = [preprocessed_text]

        prediction = model.predict(X_input)[0]
        prediction_proba = model.predict_proba(X_input)[0]

        shap_explanation = get_shap_explanation(preprocessed_text)

        highlighted_text = highlight_suspicious_content(
            f"Subject: {email_data['subject']}<br>"
            f"From: {email_data['sender']}<br>"
            f"Date: {email_data['date']}<br><br>"
            f"{email_data['body']}",
            shap_explanation['importance']
        )

        result = {
            'is_phishing': bool(prediction),
            'confidence': float(prediction_proba[1] if prediction else prediction_proba[0]),
            'shap_plot': shap_explanation['plot'],
            'highlighted_text': highlighted_text,
            'phishing_words': [word for word, importance in shap_explanation['importance'].items() if importance > 0],
            'legitimate_words': [word for word, importance in shap_explanation['importance'].items() if importance < 0]
        }

        return render_template('result.html', result=result)
    
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': str(e.__traceback__)}), 500

if __name__ == '__main__':
    # app.run(debug=True)
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)