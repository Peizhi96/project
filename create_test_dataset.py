import pandas as pd
import numpy as np
import json
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

csv_path = "data/jobs_embed_clean.csv"
total_samples = len(pd.read_csv(csv_path))
sample_size = 30000
num_samples = random.sample(range(total_samples), sample_size)

def create_test_dataset(csv_path, num_samples=num_samples, id_column='id', num_queries=10000, relevant_docs_per_query=5):
    
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} records")
    
    
    if num_samples is not None and len(num_samples) < len(df):
        print(f"Using {len(num_samples)} samples instead of full dataset")
        df = df.iloc[num_samples].reset_index(drop=True)
        print(f"Reduced to {len(df)} records")
    
    text_column = 'jobDesc_after_spacy'
    
    print(f"Using text column: {text_column}")
    
    print(f"Using ID column: '{id_column}'")
    job_ids = df[id_column].tolist()
    
    # prepare text data
    texts = df[text_column].fillna('').tolist()
    
    # use TF-IDF to find important words
    print("Calculating TF-IDF...")
    tfidf = TfidfVectorizer(max_features=500, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(texts)
    
    # get important words
    feature_names = tfidf.get_feature_names_out()
    
    # use cosine similarity to find similar documents
    print("Calculating document similarities...")
    similarities = cosine_similarity(tfidf_matrix)
    
    # generate test queries and related documents
    test_queries = {}
    selected_seed_docs = set()  
    
    print(f"Generating {num_queries} test queries...")
    for _ in range(num_queries):
        # randomly select a document as a seed
        available_docs = set(range(len(texts))) - selected_seed_docs
        if not available_docs:
            print(" No more available seed documents")
            break
            
        seed_doc_idx = random.choice(list(available_docs))
        selected_seed_docs.add(seed_doc_idx)
        
        # get the most similar documents
        sim_scores = list(enumerate(similarities[seed_doc_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # select the top 15 similar documents as relevant documents
        similar_doc_indices = [idx for idx, _ in sim_scores[1:relevant_docs_per_query+1]]
        
        # add the seed document to the relevant documents
        relevant_docs = [job_ids[seed_doc_idx]] + [job_ids[idx] for idx in similar_doc_indices]
        
        job_desc = texts[seed_doc_idx]
        query_text = job_desc.strip()
        
        
        if not query_text:
            query_text = job_desc[:100].strip()
        
        # store query and relevant documents
        test_queries[query_text] = relevant_docs
    
    print(f"Generated {len(test_queries)} test queries")
    return test_queries

def main():
    output_path = "data/search_test_dataset.json"
    id_column = "id"  
    
    
    # create test dataset
    test_dataset = create_test_dataset(
        csv_path=csv_path,
        num_samples=num_samples,
        id_column=id_column,
        num_queries=10000,   
        relevant_docs_per_query=15
    )
    
    # save test dataset
    with open(output_path, 'w') as f:
        json.dump(test_dataset, f, indent=2)
    
    print(f"Test dataset saved to: {output_path}")
    print(f"Generated {len(test_dataset)} test queries")
    
    # print some example queries
    print("\nExample queries:")
    for i, (query, relevant_docs) in enumerate(list(test_dataset.items())[:5]):
        print(f"Query {i+1}: {query}")
        print(f"Job IDs: {relevant_docs[:3]}...")
        print("")

if __name__ == "__main__":
    main() 