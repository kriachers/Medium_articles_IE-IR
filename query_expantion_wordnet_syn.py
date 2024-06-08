from data_cleaning import new_df 
from inverted_index import inverted_index

import nltk
from nltk.corpus import wordnet
import os
import pandas as pd


def get_synonyms(word):
  synonyms  = set()

  for syn in wordnet.synsets(word):
    for lemma in syn.lemmas():
      synonyms.add(lemma.name().replace("-", " "))

  return synonyms


def expand_query(query):
  expanded_query = set(query.split())

  for word in query.split():
    synonyms = get_synonyms(word)

    expanded_query.update(synonyms)

  return " ".join(expanded_query)




def search(query, inverted_index):
    expanded_query = expand_query(query)
    expanded_query_tokens = expanded_query.split()
    relevant_documents = set()

    for token in expanded_query_tokens:
        if token in inverted_index:
            relevant_documents.update(inverted_index[token])

    return relevant_documents

query = input("Please enter your query here:")
relevant_documents = search(query, inverted_index)
print(expand_query(query))
print("Document for this querry:")
print(relevant_documents)