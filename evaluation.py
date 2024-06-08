from data_cleaning import preprocess
from data_cleaning import new_df 
from tf_idf_cos_sim import inverted_index_tf_idf
from inverted_index import search

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(new_df['text_tokens'].apply(' '.join))


# Functions to calculate metrics
def precision_at_k(relevant, retrieved, k):
    retrieved_at_k = retrieved[:k]
    relevant_at_k = [doc for doc in retrieved_at_k if doc in relevant]
    return len(relevant_at_k) / k

def recall_at_k(relevant, retrieved, k):
    retrieved_at_k = retrieved[:k]
    relevant_at_k = [doc for doc in retrieved_at_k if doc in relevant]
    return len(relevant_at_k) / len(relevant)

def f1_score(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

def ndcg_at_k(relevant, retrieved, k):
    retrieved_at_k = retrieved[:k]
    dcg = sum([1 / np.log2(i + 2) for i, doc in enumerate(retrieved_at_k) if doc in relevant])
    idcg = sum([1 / np.log2(i + 2) for i in range(min(len(relevant), k))])
    return dcg / idcg

# Interactive part
query = input("Please enter your query here: ")
query = " ".join(preprocess(query))

# Perform the search
relevant_documents = search(query, inverted_index_tf_idf, tfidf_vectorizer, tfidf_matrix)

# Display the search results
print("Retrieved documents:")
for rank, (doc_index, similarity) in enumerate(relevant_documents):
    print(f"Rank {rank + 1}: Document {doc_index}, Similarity: {similarity}")

# Get relevant documents from the user
relevant_input = input("Please enter the indices of relevant documents, separated by commas: ")
relevant_docs = list(map(int, relevant_input.split(',')))

# Evaluate the search results
retrieved_docs = [doc_index for doc_index, _ in relevant_documents]

# Specify the value of k - which measures the proportion of relevant documents among the top k retrieved documents
k = 4

# Calculate evaluation metrics
precision = precision_at_k(relevant_docs, retrieved_docs, k)
recall = recall_at_k(relevant_docs, retrieved_docs, k)
f1_score_val = f1_score(precision, recall)
ndcg = ndcg_at_k(relevant_docs, retrieved_docs, k)

# Display evaluation results
print(f"\nEvaluation Results at k={k}:")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1_score_val}")
print(f"nDCG: {ndcg}")
