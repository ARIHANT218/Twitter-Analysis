import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
try:
    from ntscraper import Nitter
except Exception:
    Nitter = None

@st.cache_resource
def load_stopwords():
    nltk.download('stopwords')
    return stopwords.words('english')


@st.cache_resource
def load_model_and_vectorizer():
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer


def predict_sentiment(text, model, vectorizer, stop_words):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in stop_words]
    processed = ' '.join(words)
    X = vectorizer.transform([processed])

    
    pred = model.predict(X)
    
    try:
        label = int(pred[0])
    except Exception:
       
        try:
            label = int(pred)
        except Exception:
            label = 1
    return "Negative" if label == 0 else "Positive"

@st.cache_resource
def initialize_scraper():
    if Nitter is None:
        return None
    try:
        return Nitter(log_level=1)
    except Exception:
        return None



def create_card(tweet_text, sentiment):
    color = "green" if sentiment == "Positive" else "red"
    card_html = f"""
    <div style="background-color: {color}; padding: 10px; border-radius: 5px; margin: 10px 0;">
        <h5 style="color: white;">{sentiment} Sentiment</h5>
        <p style="color: white;">{tweet_text}</p>
    </div>
    """
    return card_html

def main():
    st.title("Twitter Sentiment Analysis")

    stop_words = load_stopwords()
    model, vectorizer = load_model_and_vectorizer()
    scraper = initialize_scraper()

    option = st.selectbox("Choose an option", ["Input text", "Get tweets from user"])
    
    if option == "Input text":
        text_input = st.text_area("Enter text to analyze sentiment")
        if st.button("Analyze"):
            sentiment = predict_sentiment(text_input, model, vectorizer, stop_words)
            st.write(f"Sentiment: {sentiment}")

    elif option == "Get tweets from user":
        username = st.text_input("Enter Twitter username")
        if st.button("Fetch Tweets"):
            tweets_data = scraper.get_tweets(username, mode='user', number=5)
            if 'tweets' in tweets_data:  # Check if the 'tweets' key exists
                for tweet in tweets_data['tweets']:
                    tweet_text = tweet['text']  # Access the text of the tweet
                    sentiment = predict_sentiment(tweet_text, model, vectorizer, stop_words)  
                
                    card_html = create_card(tweet_text, sentiment)
                    st.markdown(card_html, unsafe_allow_html=True)
            else:
                st.write("No tweets found or an error occurred.")

if __name__ == "__main__":
    main()