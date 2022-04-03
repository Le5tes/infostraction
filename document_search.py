import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# documents should have a text, a title and an index
# query is simply the query text  
def search(documents, query):
  # preprocess documents and query - remove stopwords and create bag of words model
  docs = documents
  vectorizer = TfidfVectorizer()
  X = vectorizer.fit_transform()
  # perform inverse cosine 

  pass