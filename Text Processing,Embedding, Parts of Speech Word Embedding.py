Text Preprocessing (Text Cleaning, Stemming / Lemmatization)
• Word Embedding (using an algorithm like Word2Vec, Glove, FastText)
• Encoding Techniques (Bag of Words, One – Hot)
• Parts of Speech tagging.

import pandas as pd
import nltk
import gensim
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
# Import OneHotEncoder from the correct module
from sklearn.preprocessing import OneHotEncoder 
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

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

# Text Cleaning and Preprocessing Function
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text.lower())
    
    # Stemming
    ps = PorterStemmer()
    stemmed = [ps.stem(token) for token in tokens]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(token) for token in stemmed]
    
    return ' '.join(lemmatized)

# Apply preprocessing on the 'Name' column
df['Processed_Name'] = df['Name'].apply(preprocess_text)

# Word Embedding using Word2Vec
model = Word2Vec(df['Processed_Name'].apply(lambda x: x.split()), vector_size=100, window=5, min_count=1, workers=4)

# Example of getting the vector for a name
name_vector = model.wv['jessica']  # Using the processed form for 'Jessica'

# Encoding Techniques: Bag of Words
vectorizer = CountVectorizer()
X_bow = vectorizer.fit_transform(df['Processed_Name']).toarray()

# Encoding Techniques: One-Hot Encoding
# Remove the 'sparse' argument or set it to True if using an older version
one_hot_encoder = OneHotEncoder(sparse_output=False)  # Use sparse_output instead of sparse
X_one_hot = one_hot_encoder.fit_transform(X_bow)

# Parts of Speech Tagging
pos_tags = [pos_tag(name.split()) for name in df['Processed_Name']]

# Output results
print("Original DataFrame:\n", df[['State', 'Gender', 'Year', 'Name', 'Births']])
print("\nPreprocessed Names:\n", df['Processed_Name'])
print("\nWord Vector for 'jessica':", name_vector)
print("\nBag of Words Representation:\n", X_bow)
print("\nOne-Hot Encoding Representation:\n", X_one_hot)
print("\nParts of Speech Tags:\n", pos_tags)
