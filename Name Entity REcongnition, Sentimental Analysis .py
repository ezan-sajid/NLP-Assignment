import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import spacy

# Sample dataset
data = {
    'State': ['AK', 'AK', 'AK', 'AK', 'AK'],
    'Gender': ['F', 'F', 'F', 'F', 'F'],
    'Year': [1980, 1980, 1980, 1980, 1980],
    'Name': ['Jessica', 'Jennifer', 'Sarah', 'Amanda', 'Melissa'],
    'Births': [116, 114, 82, 71, 65]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Initialize VADER sentiment analyzer
nltk.download('vader_lexicon')
vader_analyzer = SentimentIntensityAnalyzer()

# Function to get sentiment using VADER
def get_vader_sentiment(name):
    return vader_analyzer.polarity_scores(name)

# Function to get sentiment using TextBlob
def get_textblob_sentiment(name):
    return TextBlob(name).sentiment

# Apply sentiment analysis using VADER and TextBlob
df['VADER_Sentiment'] = df['Name'].apply(get_vader_sentiment)
df['TextBlob_Sentiment'] = df['Name'].apply(get_textblob_sentiment)

# Named Entity Recognition (NER) using SpaCy
nlp = spacy.load("en_core_web_sm")

def get_entities(name):
    doc = nlp(name)
    return [(ent.text, ent.label_) for ent in doc.ents]

# Apply NER
df['Named_Entities'] = df['Name'].apply(get_entities)

# Output results
print("Original DataFrame:\n", df[['State', 'Gender', 'Year', 'Name', 'Births']])
print("\nSentiment Analysis Results:\n", df[['Name', 'VADER_Sentiment', 'TextBlob_Sentiment']])
print("\nNamed Entity Recognition Results:\n", df[['Name', 'Named_Entities']])
