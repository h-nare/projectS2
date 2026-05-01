import string
from ctfidf import compute_tf


def clean_text(text):
    text = text.lower()

    for char in string.punctuation:
        text = text.replace(char, "")

    text = " ".join(text.split())

    return text


def extract_features(text):
    cleaned = clean_text(text)
    return compute_tf(cleaned)
