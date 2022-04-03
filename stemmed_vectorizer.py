from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer

class StemmedVectorizer(CountVectorizer):
  def build_analyzer(self):
    original_analyzer = super().build_analyzer()
    stem = SnowballStemmer('english').stem
    def new_analyser(item):
      return [stem(word) for word in original_analyzer(item) if not word.isnumeric()]
    return new_analyser