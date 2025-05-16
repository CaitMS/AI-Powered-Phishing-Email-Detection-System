import re

SUSPICIOUS_KEYWORDS = [
    "click here", "urgent", "verify your account", "login now", "update your information",
    "password", "bank", "security", "confirm", "limited time", "act now", "suspended"
]

def extract_features(email_text):
    # TODO: Use your actual model features here
    return {
        "length": len(email_text),
        "num_links": email_text.count("http"),
        "num_suspicious_words": sum(kw in email_text.lower() for kw in SUSPICIOUS_KEYWORDS),
        # Add your real features...
    }

def highlight_phishing_indicators(email_text):
    highlighted = email_text
    for keyword in SUSPICIOUS_KEYWORDS:
        highlighted = re.sub(f"(?i)({re.escape(keyword)})", r'<span class="highlight">\1</span>', highlighted)
    return highlighted.replace("\n", "<br>")
