Twitter Sentiment Analysis

This project is an end-to-end Machine Learning pipeline that analyzes tweets and classifies them as Positive or Negative.

📌 Project Overview

The goal of this project is to understand public sentiment from Twitter data using Natural Language Processing (NLP) and Machine Learning.

The system:

Collects tweets using Twitter API v2

Cleans and preprocesses text data

Converts text into numerical format using TF-IDF

Trains ML models for classification

Displays results using a Streamlit dashboard

🛠 Tech Stack

Python

Pandas

NLTK

Scikit-learn

Streamlit

Twitter API v2

⚙️ How the Project Works
1️⃣ Data Collection

Tweets are collected using Twitter API v2 and stored for processing.

2️⃣ Data Preprocessing

Remove URLs, mentions, special characters

Convert text to lowercase

Remove stopwords

Tokenization

3️⃣ Feature Engineering

Text is converted into numerical vectors using TF-IDF Vectorization

4️⃣ Model Training

Trained multiple models:

Logistic Regression

Naive Bayes

Random Forest

Logistic Regression achieved 80%+ accuracy.

5️⃣ Model Evaluation

Evaluated using:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

6️⃣ Deployment

The trained model is saved using pickle and deployed using a Streamlit dashboard for real-time sentiment prediction.
