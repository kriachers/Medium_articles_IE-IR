from data_cleaning import new_df 
from data_cleaning import preprocess

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(new_df['text_tokens'].apply(' '.join))

# Get feature names (terms)
terms = tfidf_vectorizer.get_feature_names_out()

def build_inverted_index_tf_idf(tfidf_matrix, terms):
    inverted_index = defaultdict(list)

    # Iterate through documents to populate the inverted index with TF-IDF scores
    for idx, tfidf_vec in enumerate(tfidf_matrix):
        for term_idx, tfidf_score in zip(tfidf_vec.indices, tfidf_vec.data):
            term = terms[term_idx]
            inverted_index[term].append((idx, tfidf_score))

    return inverted_index

inverted_index_tf_idf= build_inverted_index_tf_idf(tfidf_matrix, terms)

def write_inverted_index_to_file(inverted_index_tf_idf, filename):
    with open(filename, 'w') as file:
        for token, doc_numbers in inverted_index_tf_idf.items():
            file.write(f"{token}: {', '.join(map(str, doc_numbers))}\n")

write_inverted_index_to_file(inverted_index_tf_idf, 'inverted_index_tf_idf.txt')

"""
Cosine similarity and Ranking
"""

def search(query, inverted_index_tf_idf, tfidf_vectorizer, tfidf_matrix, limit=10):
    query_tfidf = tfidf_vectorizer.transform([query])

    # Retrieve relevant documents using the inverted index
    query_tokens = query.split()
    relevant_documents = set()
    for token in query_tokens:
        if token in inverted_index_tf_idf:
            relevant_documents.update(inverted_index_tf_idf[token])


    cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()

    ranked_indices = cosine_similarities.argsort()[::-1]
    ranked_documents = [(index, cosine_similarities[index]) for index in ranked_indices[:limit]]

    return ranked_documents

query = input("Please enter your query here:")
query = " ".join(preprocess(query))
relevant_documents = search(query, inverted_index_tf_idf, tfidf_vectorizer, tfidf_matrix)

for rank, (doc_index, similarity) in enumerate(relevant_documents):
    print(f" Article {doc_index}, Similarity: {similarity}")