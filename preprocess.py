import pandas as pd
import numpy as np
import re
import nltk
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# Load dataset
def load_data(train_file, test_file):
    train_data = pd.read_csv(train_file, sep=' ::: ', engine='python', names=['ID', 'TITLE', 'GENRE', 'DESCRIPTION'])
    test_data = pd.read_csv(test_file, sep=' ::: ', engine='python', names=['ID', 'TITLE', 'DESCRIPTION'])
    
    # Clean text descriptions
    train_data['CLEAN_DESCRIPTION'] = train_data['DESCRIPTION'].apply(clean_text)
    test_data['CLEAN_DESCRIPTION'] = test_data['DESCRIPTION'].apply(clean_text)

    return train_data, test_data

# Feature Extraction
def extract_features(train_data, test_data):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_data['CLEAN_DESCRIPTION'])
    X_test = vectorizer.transform(test_data['CLEAN_DESCRIPTION'])
    y_train = train_data['GENRE']

    joblib.dump(vectorizer, 'vectorizer.pkl')  # Save vectorizer for deployment
    return X_train, X_test, y_train

# Train Model
def train_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, 'genre_model.pkl')  # Save model for deployment
    return model

# Evaluate Model
def evaluate_model(model, X_train, y_train):
    y_pred = model.predict(X_train)
    print("Accuracy:", accuracy_score(y_train, y_pred))
    print(classification_report(y_train, y_pred))

# Main Function
if __name__ == "__main__":
    train_file = "train_data.txt"
    test_file = "test_data.txt"

    train_data, test_data = load_data(train_file, test_file)
    X_train, X_test, y_train = extract_features(train_data, test_data)
    
    model = train_model(X_train, y_train)
    evaluate_model(model, X_train, y_train)
    
    print("Model trained and saved successfully!")
