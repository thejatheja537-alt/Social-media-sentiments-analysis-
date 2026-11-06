import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ==========================================
# 1. LOAD AND INSPECT THE DATASET
# ==========================================
# Update this path if your CSV file is in a different folder
file_name = '/Users/ranjith/Desktop/sentiments-analysis/socialmedia sentimentdataset.csv'

df = pd.read_csv(file_name)

print('--- Dataset Info ---')
print(f'Dataset Shape: {df.shape}')
print(df.head(3))

# ==========================================
# 2. DATA PREPROCESSING
# ==========================================
# Drop any rows where Text or Sentiment is missing
df = df.dropna(subset=['Text', 'Sentiment'])

X = df['Text'].astype(str)  # Feature: Social media text content
y = df['Sentiment']  # Target: Sentiment label

# Split into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================================
# 3. TEXT VECTORIZATION (TF-IDF)
# ==========================================
tfidf = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# ==========================================
# 4. LOGISTIC REGRESSION MODEL TRAINING
# ==========================================
lr_classifier = LogisticRegression(max_iter=1000, random_state=42)
lr_classifier.fit(X_train_tfidf, y_train)

# ==========================================
# 5. MODEL EVALUATION & PERFORMANCE
# ==========================================
y_pred = lr_classifier.predict(X_test_tfidf)

print('\n--- Model Performance ---')
print(f'Accuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%\n')
print('Classification Report:')
print(classification_report(y_test, y_pred, zero_division=0))

# ==========================================
# 6. INTERACTIVE SAMPLE PREDICTION
# ==========================================
sample_posts = [
    'Enjoying a wonderful sunny day at the park with friends!',
    'Traffic was completely terrible this morning and ruined my mood.',
    'Just finished an amazing workout session, feeling full of energy!',
]

sample_tfidf = tfidf.transform(sample_posts)
sample_predictions = lr_classifier.predict(sample_tfidf)

print('\n--- Custom Sample Predictions ---')
for post, sentiment in zip(sample_posts, sample_predictions):
    print(f'Post: "{post}"')
    print(f'Predicted Sentiment: ➔ {sentiment}\n')