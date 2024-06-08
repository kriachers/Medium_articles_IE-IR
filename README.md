# IE/IR project: Medium articles search system

## Project Content:

- **data_cleaning.py** - Loads and preprocess the data
- **inverted_index.py** - Creates inversted indexing
- **query_expantion_wordnet_syn.py** - Query expansion (wordnet - synonyms)
- **query_expansion_matrix.py** - Query expansion (co-occurrence matrix)
Improving our query results with the help of a co-occurrence matrix.
- **tf_idf_cos_sim.py** - Tf-idf Indexing and Cosine Similarity. This is the module with the best version of search
- **evaluation.py** - Evaluation (precision, recall, f1_score, nDCG)

## Installation and Usage

1. **Clone the repository:**

   ```bash
   git clone https://github.com/kriachers/Medium_articles_IE-IR
   cd Medium_articles_IE-IR
   ```
   

2. **Install the required packages:**

```
pip install -r requirements.txt
```

OR type following commands in the terminal

```
python pip install pandas
python pip install numpy
python pip install collections
python pip install itertools
python pip install sklearn
python pip install nltk

```

3. **Data Cleaning:**

Use data_cleaning.py to load the dataset:
```
python data_cleaning.py
```

4. **Inverted index**

Use inverted_index.py to load the dataset:
```
python inverted_index.py
```

5. **Query expansion (wordnet - synonyms)**

Use query_expantion_wordnet_syn.py to load the dataset:
```
python query_expantion_wordnet_syn.py
```

6. **Query expansion (co-occurrence matrix)**

Use query_expantion_wordnet_syn.py to load the dataset:
```
python query_expansion_matrix.py
```

6. **Tf-idf Indexing and Cosine Similarity**

Use tf_idf_cos_sim.py to load the dataset:
```
python tf_idf_cos_sim.py
```

6. **Evaluation**

Use evaluation.py to load the dataset:
```
python evaluation.py
```


