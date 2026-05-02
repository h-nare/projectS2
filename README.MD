# Hybrid Spam Detection System

A machine learning web application that classifies SMS messages as **Spam** or **Not Spam** using NLP, Logistic Regression, and a custom C module for word frequency analysis.

## Features

- Real-time spam prediction
- Confidence score
- Probability breakdown
- Important words display
- Suspicious keyword detection
- Prediction history
- Hybrid Python + C architecture
- Deployable web application

## Technologies

- Python
- Flask
- Scikit-learn
- Pandas
- C
- ctypes
- Gunicorn
- HTML / CSS
- Render

## Project Structure

Hybrid-Spam-Detection-System/
├── app.py  
├── model.py  
├── main.py  
├── ctfidf.py  
├── tfidf.c  
├── requirements.txt  
├── Procfile  
├── data/spam.csv  
├── templates/index.html  
└── utils/preprocess.py  

## How It Works

1. Text is cleaned (lowercase, punctuation removed).
2. TF-IDF converts text into numeric vectors.

TF-IDF(t,d) = TF(t,d) * log(N / DF(t))

3. Logistic Regression predicts:
- Spam
- Not Spam

4. C module calculates word frequency for keyword analysis.

## Installation

Clone repository:

git clone https://github.com/h-nare/projectS2_spamDetection.git

Install dependencies:

pip install -r requirements.txt

Compile C module:

gcc -shared -o tfidf.so -fPIC tfidf.c

Run application:

python app.py

Open:

http://127.0.0.1:8000

Or can instantly open via this link 
https://projects2-ev62.onrender.com

## Deployment

Production start command:

gunicorn app:app

## Example

Input:

Win a free cash prize now!

Output:

Spam  
Confidence: 96%

## Performance

- Accuracy: ~95%
- Cross-validation: ~95%

## Future Improvements

- Deep Learning model
- Better UI
- Multilingual spam detection
- Email spam filtering
