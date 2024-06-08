from data_cleaning import new_df 
from inverted_index import inverted_index
from data_cleaning import preprocess


import numpy as np
import pandas as pd
from collections import Counter
from collections import defaultdict
from itertools import combinations


tokens_text = list(word for word in new_df["text_tokens"] )
all_tokens = tokens_text


list_all_tokens = [word for sublist in all_tokens for word in sublist]
unique_words_corpus = list(set(list_all_tokens))
size_unique_words = len(unique_words_corpus)

window_size = 2

co_occurrence_matrix = pd.DataFrame(0, index=unique_words_corpus, columns=unique_words_corpus)

for sentence in all_tokens:
    for i, word in enumerate(sentence):
      start = max(0, i - window_size)
      end = min(len(sentence), i + window_size + 1)
      for context_word in sentence[start:i] + sentence[i+1:end]:
        co_occurrence_matrix.at[word, context_word] += 1

def expanded_query2(query, co_occurrence_matrix):
  query_terms = set(query.split())
  terms_to_add = set()
  for term in query_terms:
    if term in co_occurrence_matrix.index:
      co_occurrences = co_occurrence_matrix.loc[term].values
      top_co_occurrences = np.argsort(co_occurrences)[::-1][:5]
      terms_to_add.update(co_occurrence_matrix.index[top_co_occurrences])

  query_terms.update(terms_to_add)
  return query_terms


def search(query, inverted_index):
    query_tokens = expanded_query2(query, co_occurrence_matrix)

    document_word_count = defaultdict(int)

    for token in query_tokens:
        if token in inverted_index:
            for document in inverted_index[token]:
                document_word_count[document] += 1

    sorted_documents = sorted(document_word_count.items(), key=lambda x: x[1], reverse=True)

    # Select the top 10 documents
    top_documents = [doc for doc, _ in sorted_documents[:10]]

    return top_documents

query2 = input("Please enter your query here:")
query2 = " ".join(preprocess(query2))
relevant_doc = search(query2, inverted_index)
print(relevant_doc)