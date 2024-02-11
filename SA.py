import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from joblib import dump

# Load the dataset
df = pd.read_csv('Data/Sampled_reviews.csv')

# Drop any rows with missing values
df.dropna(inplace=True)

# Perform basic text preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
sid = SentimentIntensityAnalyzer()

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords and non-alphabetic characters
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    # Join the tokens back into a string
    return ' '.join(tokens)

# Apply preprocessing to the 'Text' column
df['cleaned_text'] = df['Text'].apply(preprocess_text)

# Add a new column for sentiment polarity
df['sentiment'] = df['cleaned_text'].apply(lambda x: sid.polarity_scores(x)['compound'])

# Convert sentiment polarity scores to discrete classes
def convert_to_class(sentiment_score):
    if sentiment_score > 0.2:
        return 'positive'
    elif sentiment_score < -0.2:
        return 'negative'
    else:
        return 'neutral'

# Apply the conversion function to create a new column with discrete sentiment classes
df['sentiment_class'] = df['sentiment'].apply(convert_to_class)

# Features and target variables
X = df['cleaned_text']
y = df['sentiment_class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)

# Fit and transform the training data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Transform the testing data
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Logistic Regression
logreg_model = LogisticRegression(max_iter=1000)
logreg_model.fit(X_train_tfidf, y_train)
y_pred = logreg_model.predict(X_test_tfidf)

# Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
y_pred_nb = nb_model.predict(X_test_tfidf)

# Evaluate models
def evaluate_model(model, y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    print(f"{model_name} Accuracy: {accuracy}")
    print(f"{model_name} Classification Report:")
    print(classification_report(y_true, y_pred))
    return accuracy

# Plot accuracy comparison
def plot_accuracy_comparison(models, accuracies):
    plt.figure(figsize=(8, 5))
    sns.barplot(x=models, y=accuracies)
    plt.title('Accuracy Comparison of Models')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.show()

# Plot confusion matrix
def plot_confusion_matrix(model_name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# Model evaluation
models = ['Logistic Regression', 'Naive Bayes']
accuracies = [evaluate_model(logreg_model, y_test, y_pred, 'Logistic Regression'),
              evaluate_model(nb_model, y_test, y_pred_nb, 'Naive Bayes')]

# Plot accuracy comparison
plot_accuracy_comparison(models, accuracies)

# Plot confusion matrices
plot_confusion_matrix('Logistic Regression', y_test, y_pred)
plot_confusion_matrix('Naive Bayes', y_test, y_pred_nb)

# Hyperparameters for Logistic Regression
logreg_params = {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2']}

# Grid Search for Logistic Regression
logreg_grid = GridSearchCV(LogisticRegression(max_iter=1000), logreg_params, cv=5, verbose=1, n_jobs=-1)
logreg_grid.fit(X_train_tfidf, y_train)

# Best hyperparameters for Logistic Regression
print("Best hyperparameters for Logistic Regression:", logreg_grid.best_params_)

# Evaluate the best Logistic Regression model
best_logreg_model = logreg_grid.best_estimator_
y_pred_logreg = best_logreg_model.predict(X_test_tfidf)

# Evaluate best Logistic Regression model
accuracy_logreg = evaluate_model(best_logreg_model, y_test, y_pred_logreg, 'Best Logistic Regression')

# Save models
dump(tfidf_vectorizer, 'Models/tfidf_vectorizer.pkl')
dump(best_logreg_model, 'Models/best_logreg_model.pkl')
