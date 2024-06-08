import os
import pandas as pd

import string
import nltk

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords

"""
ssl for nltk.download correctly downloads
"""

import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')


"""
FILE READING
"""
current_directory = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_directory, 'golden_dataset.csv')
df = pd.read_csv(file_path)

"""
WEIGHTS ASSIGNING TO THE TITLE
"""

def combine_title_text(row, title_weight=3):
    title = row['title']
    text = row['text']
    combined_text = (title + ' ') * title_weight + text
    return combined_text

df['text'] = df.apply(combine_title_text, axis=1)



"""
DATASET CLEANING
"""
def get_wordnet_pos(tag):
    ## This dict we need in order to make mapping of the wordnet auto detection of POS to WordNetLemmatizer POS
    tag = tag[0].upper()
    tag_dict = {
        'J': wordnet.ADJ,
        'N': wordnet.NOUN,
        'V': wordnet.VERB,
        'R': wordnet.ADV
    }
    return tag_dict.get(tag, wordnet.NOUN)


stop_words = set(stopwords.words('english'))
def preprocess(text):
    translator = str.maketrans('', '', string.punctuation + '“' + '’' + '–')  # we additionally added ’ – symbol, because translator module by default did not include it
    sentence = text.translate(translator)
    tokens = word_tokenize(sentence)
    tokens = [word.lower() for word in tokens]
    filtered_tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(word, get_wordnet_pos(pos_tag)) for word, pos_tag in nltk.pos_tag(filtered_tokens)]
    return lemmas

df['text_tokens'] = df['text'].apply(preprocess)
new_df = df[['text_tokens']].copy()
