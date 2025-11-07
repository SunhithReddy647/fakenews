import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
import string
import joblib
import os

# Download stopwords if not already present
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# Load data
fake = pd.read_csv("data/Fake.csv", nrows=1000)
true = pd.read_csv("data/True.csv", nrows=1000)

# Add target column
fake['target'] = 'fake'
true['target'] = 'true'

# Concatenate and shuffle
data = pd.concat([fake, true]).reset_index(drop=True)
data = shuffle(data).reset_index(drop=True)

# Clean data
data.drop(["date", "title"], axis=1, inplace=True)
data['text'] = data['text'].apply(lambda x: x.lower())

def punctuation_removal(text):
    all_list = [char for char in text if char not in string.punctuation]
    clean_str = ''.join(all_list)
    return clean_str
data['text'] = data['text'].apply(punctuation_removal)

stop = stopwords.words('english')
data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

# Split data
X_train, X_test, y_train, y_test = train_test_split(data['text'], data.target, test_size=0.2, random_state=42)

# Create and train pipeline
pipe = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('model', RandomForestClassifier(n_estimators=50, criterion="gini"))
])
model = pipe.fit(X_train, y_train)

# Print accuracy
prediction = model.predict(X_test)
print(f"Accuracy: {round(accuracy_score(y_test, prediction)*100,2)}%")

# Save the model
model_path = os.path.abspath('model.joblib')
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")
if os.path.exists(model_path):
    print("Model file confirmed to exist.")
else:
    print("Error: Model file was not created.")
