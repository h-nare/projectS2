from flask import Flask, request, render_template
from model import model, vectorizer
from utils.preprocess import clean_text, extract_features
from datetime import datetime

app = Flask(__name__)

history = []

SPAM_WORDS = {
    "free", "win", "winner", "prize", "urgent",
    "offer", "cash", "call", "claim", "limited"
}


@app.route("/")
def home():
    return render_template("index.html", history=history[-5:])


@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("message", "").strip()

    if not text:
        return render_template(
            "index.html",
            error="Please enter a message.",
            history=history[-5:]
        )

    cleaned = clean_text(text)

    tf_features = extract_features(text)

    vector = vectorizer.transform([cleaned])

    prob = model.predict_proba(vector)[0]
    pred = model.predict(vector)[0]

    label = "Spam ❌" if pred == 1 else "Not Spam ✅"

    spam_prob = round(prob[1] * 100, 2)
    ham_prob = round(prob[0] * 100, 2)
    confidence = round(max(prob) * 100, 2)

    word_count = len(cleaned.split())
    char_count = len(text)

    if confidence >= 80:
        confidence_text = "Very confident"
    elif confidence >= 60:
        confidence_text = "Moderately confident"
    else:
        confidence_text = "Low confidence"

    suspicious_words = {
        word: count
        for word, count in tf_features.items()
        if word in SPAM_WORDS
    }

    important_words = sorted(
        tf_features.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]

    top_word = important_words[0][0] if important_words else "None"

    history.append((
        text,
        label,
        confidence,
        datetime.now().strftime("%H:%M")
    ))

    return render_template(
        "index.html",
        prediction=label,
        confidence=confidence,
        spam_prob=spam_prob,
        ham_prob=ham_prob,
        confidence_text=confidence_text,
        cleaned_text=cleaned,
        word_count=word_count,
        char_count=char_count,
        important_words=important_words,
        suspicious_words=suspicious_words,
        top_word=top_word,
        history=history[-5:]
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)