# Twitter Sentiment Analysis

## Overview

This project is an end-to-end Machine Learning pipeline that analyzes
tweets and classifies them as **Positive** or **Negative** sentiment.

The objective is to understand public opinion using Natural Language
Processing (NLP) techniques and supervised machine learning models.

------------------------------------------------------------------------

## System Workflow

### 1. Data Collection

-   Tweets collected using Twitter API v2
-   Data stored in structured format for processing

### 2. Data Preprocessing

-   Remove URLs, mentions, and special characters
-   Convert text to lowercase
-   Remove stopwords using NLTK
-   Perform tokenization

### 3. Feature Engineering

-   Convert text into numerical vectors using TF-IDF Vectorization
-   Control dimensionality using feature limits

### 4. Model Training

Models trained: - Logistic Regression - Naive Bayes - Random Forest

Logistic Regression achieved **80%+ accuracy** and performed best on
sparse TF-IDF features.

### 5. Model Evaluation

Evaluation metrics used: - Accuracy - Precision - Recall - F1-Score -
Confusion Matrix

### 6. Deployment

-   Model and vectorizer saved using pickle
-   Integrated with a Streamlit dashboard
-   Enables real-time sentiment prediction

------------------------------------------------------------------------

## Tech Stack

-   Python
-   Pandas
-   NLTK
-   Scikit-learn
-   Streamlit
-   Twitter API v2

------------------------------------------------------------------------

## How to Run

1.  Install dependencies: pip install -r requirements.txt

2.  Run the application: streamlit run app.py

------------------------------------------------------------------------

## Key Features

-   Processes 1.6M+ tweets
-   Real-time sentiment prediction
-   Interactive dashboard visualization
