from flask import Flask, render_template, request
from joblib import load
import re
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load the TF-IDF vectorizer and trained Logistic Regression model
tfidf_vectorizer = load('Models/tfidf_vectorizer.pkl')
best_logreg_model = load('Models/best_logreg_model.pkl')

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def predict_sentiment(text):
    cleaned_text = preprocess_text(text)
    text_vectorized = tfidf_vectorizer.transform([cleaned_text])
    prediction = best_logreg_model.predict(text_vectorized)
    return prediction[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    sentiment = predict_sentiment(text)
    return render_template('index.html', text=text, sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
