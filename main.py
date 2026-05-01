import pandas as pd
from utils.preprocess import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#Loading the datatset
data = pd.read_csv("data/spam.csv", encoding="latin-1")

#keeping one the required columns
data = data[["v1","v2"]]
data.columns = ['label', 'text']

#print(data.head())

data['clean_text'] = data['text'].apply(clean_text)

#print(data[['text', 'clean_text']].head())

data['label'] = data['label'].map({'ham': 0, 'spam': 1})

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(data['clean_text'])
y = data['label']

#splitting the dataset
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

#training the model
model = LogisticRegression()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)
