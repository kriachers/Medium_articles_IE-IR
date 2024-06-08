from data_cleaning import new_df 

def build_inverted_index(df):
    inverted_index = {}

    for idx, row in df.iterrows():
        doc_number = idx
        for token in row['text_tokens']:
            if token in inverted_index:
                if doc_number not in inverted_index[token]:
                    inverted_index[token].append(doc_number)
            else:
                inverted_index[token] = [doc_number]
    return inverted_index

def print_inverted_index(inverted_index):
    for token, doc_numbers in inverted_index.items():
        print(f"{token}: {', '.join(map(str, doc_numbers))}")


inverted_index = build_inverted_index(new_df)

def write_inverted_index_to_file(inverted_index, filename):
    with open(filename, 'w') as file:
        for token, doc_numbers in inverted_index.items():
            file.write(f"{token}: {', '.join(map(str, doc_numbers))}\n")

write_inverted_index_to_file(inverted_index, 'inverted_index.txt')

def load_inverted_index(filename):
    inverted_index = {}
    with open(filename, 'r') as file:
        for line in file:
            token, doc_numbers_str = line.strip().split(':')
            doc_numbers = list(map(int, doc_numbers_str.split(',')))
            inverted_index[token] = doc_numbers
    return inverted_index

def search(query, inverted_index):
    query_tokens = query.split()
    relevant_documents = set()

    for token in query_tokens:
        if token in inverted_index:
            relevant_documents.update(inverted_index[token])

    return relevant_documents


inverted_index = load_inverted_index('inverted_index.txt')

# SEARCH EXAMPLE
query = "python"
relevant_documents = search(query, inverted_index)

print("Document for this querry:")
print(relevant_documents)
