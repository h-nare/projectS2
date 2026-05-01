import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from utils.preprocess import clean_text

data = pd.read_csv("data/spam.csv", encoding="latin-1")

data = data[["v1", "v2"]]
data.columns = ["label", "text"]

data["clean_text"] = data["text"].apply(clean_text)
data["label"] = data["label"].map({"ham": 0, "spam": 1})

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(data["clean_text"])
y = data["label"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)