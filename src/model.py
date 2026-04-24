from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

def build_model():
    vectorizer = CountVectorizer()
    model = LogisticRegression()
    return vectorizer, model