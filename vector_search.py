import numpy as np
import pandas as pd
import time
import json
import random
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import os
from gensim.models import FastText
import spacy
from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection,
)
import os
from dotenv import load_dotenv

load_dotenv()
try:
    load_dotenv(override=True)
    MILVUS_HOST = os.getenv("MILVUS_HOST")
    MILVUS_PORT = os.getenv("MILVUS_PORT")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME")
except Exception as e:
    print(f"Error loading environment variables: {e}")
finally:
    connections.disconnect("default")

class VectorSearch:
    def __init__(self, collection_name=COLLECTION_NAME, vector_dim=100, host=MILVUS_HOST, port=MILVUS_PORT):
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.host = host
        self.port = port
        self.collection = None
        self.model = None
        
    def load_data(self, csv_path='data/jobs_embed_clean.csv'):
        try:
            df = pd.read_csv(csv_path)
            return df
        except Exception as e:
            print(f"Failed to load data from {csv_path}: {e}")
            return None
    
    # generate the job description vectors
    def generate_job_description_vectors(self, text):
        words = text.split() if isinstance(text, str) else []
        if not words:
            return np.zeros(self.vector_dim)
        
        words_vectors = [self.model.wv[word] for word in words if word in self.model.wv]
        if not words_vectors:
            return np.zeros(self.vector_dim)
        
        doc_vector = np.mean(words_vectors, axis=0)
        doc_vector = doc_vector / np.linalg.norm(doc_vector)
        return doc_vector.astype(np.float32)
    
    # alias for generate_job_description_vectors for compatibility
    def generate_embedding(self, text):
        return self.generate_job_description_vectors(text)
    
    # connect to milvus
    def connect_to_milvus(self):
        try:
            connections.connect("default", host=self.host, port=self.port)
            print(f"Successfully connected to Milvus server {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"Failed to connect to Milvus: {e}")
            return False
    
    # create the collection
    def create_collection(self, drop_existing=False):
        
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="job_vector", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim)
        ]
        
        schema = CollectionSchema(fields, description=f"Job vector collection, dimension={self.vector_dim}")
        if utility.has_collection(self.collection_name):
            self.collection = Collection(name=self.collection_name)
        else:
            self.collection = Collection(name=self.collection_name, schema=schema)
        print(f"Collection {self.collection_name} created successfully")
        
        return self.collection
    
    # create the index
    def create_index(self, index_type="HNSW", metric_type="L2"):
        if self.collection is None:
            print("Please create or connect to the collection first")
            return False
        
        index_params = {
            "metric_type": metric_type,
            "index_type": index_type,
            "params": {
                "M": 16,
                "efConstruction": 64
            }
        }
        
        try:
            self.collection.create_index("job_vector", index_params)
            print(f"Created {index_type} index for vector field, metric type: {metric_type}")
            return True
        except Exception as e:
            print(f"Failed to create index: {e}")
            return False
    
    # load the model
    def load_model(self, model_path):
        try:
            self.model = FastText.load(model_path)
            self.vector_dim = self.model.vector_size
            print(f"FastText model loaded, vector dimension: {self.vector_dim}")
            return True
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False
    
    # batch generate the embeddings
    def batch_generate_embeddings(self, texts, show_progress=True):
        if self.model is None:
            print("Please train or load the FastText model first")
            return None
        
        embeddings = []
        
        if show_progress:
            iterator = tqdm(texts, desc="Generating embeddings")
        else:
            iterator = texts
        
        for text in iterator:
            embedding = self.generate_embedding(text)
            embeddings.append(embedding)
        
        return np.array(embeddings, dtype=np.float32)
    
    # insert the vectors
    def insert_vectors(self, vectors, ids=None, batch_size=5000):
        if self.collection is None:
            print("Please create the collection first")
            return False
        
        total_vectors = len(vectors)
        
        if ids is None:
            ids = list(range(total_vectors))
        
        print(f"Inserting {total_vectors} vectors, batch size: {batch_size}")
        
        for i in range(0, total_vectors, batch_size):
            end_idx = min(i + batch_size, total_vectors)
            batch_vectors = vectors[i:end_idx]
            batch_ids = ids[i:end_idx]
            
            entities = [
                batch_ids,  
                batch_vectors.tolist()  
            ]
            
            try:
                self.collection.insert(entities)
                print(f"Inserted batch {i//batch_size + 1}/{(total_vectors-1)//batch_size + 1}, progress: {end_idx}/{total_vectors}")
            except Exception as e:
                print(f"Failed to insert batch {i//batch_size + 1}: {e}")
                return False
        
       
        self.collection.flush()
        print(f"Inserted {total_vectors} vectors, current collection entity count: {self.collection.num_entities}")
        
        return True
    
    # load the collection
    def load_collection(self):
        if self.collection is None:
            print("Please create the collection first")
            return False
        
        try:
            self.collection.load()
            print(f"Collection {self.collection_name} loaded into memory, containing {self.collection.num_entities} entities")
            return True
        except Exception as e:
            print(f"Failed to load collection: {e}")
            return False
    
    # search the collection
    def search(self, query_text=None, query_vector=None, top_k=15, nprobe=30):
        if self.collection is None:
            print("Please create and load the collection first")
            return None
        
        # determine the query vector
        if query_vector is None and query_text is not None:
            query_vector = self.generate_embedding(query_text)
            if query_vector is None or np.all(query_vector == 0):
                print(f"Warning: Generated empty vector for query '{query_text}'")
                return None
        
        if query_vector is None:
            print("Error: Both query_text and query_vector are None")
            return None
        
        # ensure the vector is a 2D array
        if isinstance(query_vector, np.ndarray):
            if len(query_vector.shape) == 1:
                query_vector = [query_vector.tolist()]
            else:
                query_vector = query_vector.tolist()
        elif not isinstance(query_vector, list):
            print(f"Error: Query vector has unexpected type {type(query_vector)}")
            return None
        elif not isinstance(query_vector[0], list) and not all(isinstance(x, (int, float)) for x in query_vector[0]):
            query_vector = [query_vector]
        
        # search parameters
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": nprobe}
        }
        
        try:
            results = self.collection.search(
                data=query_vector,  
                anns_field="job_vector",
                param=search_params,
                limit=top_k,
                output_fields=["id"]
            )
            
            return results
        except Exception as e:
            print(f"Search failed: {e}")
            return None
    
    # format the search results
    def format_search_results(self, results, df=None):
        if results is None:
            return []
        
        formatted_results = []
        
        for i, hits in enumerate(results):
            query_results = []
            
            for hit in hits:
                item = {
                    "id": hit.id,
                    "distance": hit.distance,
                    "score": 1.0 / (1.0 + hit.distance)  
                }
                
                query_results.append(item)
            
            formatted_results.append(query_results)
        
        return formatted_results
    
    # drop the collection
    def drop_collection(self):
        if utility.has_collection(self.collection_name):
            try:
                utility.drop_collection(self.collection_name)
                self.collection = None
                print(f" {self.collection_name} dropped")
                return True
            except Exception as e:
                print(f"Failed to drop collection: {e}")
                return False
        else:
            print(f" {self.collection_name} does not exist")
            return False
    
    # release the collection
    def release_collection(self):
        if self.collection is not None:
            try:
                self.collection.release()
                print(f" {self.collection_name} released")
                return True
            except Exception as e:
                print(f"Failed to release collection resources: {e}")
                return False
        return True
    
    # disconnect from milvus
    def disconnect(self):
        try:
            connections.disconnect("default")
            print("Disconnect successfully")
            return True
        except Exception as e:
            print(f"Disconnect failed: {e}")
            return False

    def clear_collection(self):
        if self.collection is not None:
            utility.drop_collection(self.collection_name)
            self.collection = None
            print(f"Collection {self.collection_name} cleared")
        else:
            print(f"Collection {self.collection_name} does not exist")

def analyze_similarity_thresholds(test_queries, vector_search,
                                csv_path='data/jobs_embed_clean.csv', 
                                id_column='id', 
                                text_column='jobDesc_after_spacy',
                                sample_pairs_limit=10000):
    
    print("Analyzing Similarity Thresholds (FastText Cosine Similarity on Test Subset)")
    
    # Check if model is loaded in vector_search object
    if not vector_search.model:
        print("Error: FastText model not loaded in the provided VectorSearch object.")
        return None

    # Extract unique IDs from test_queries
    print("Extracting unique IDs from test dataset...")
    unique_ids = set()
    # query_texts = list(test_queries.keys()) # Not strictly needed here
    for relevant_list in test_queries.values():
        unique_ids.update(relevant_list)
    
    if not unique_ids:
        print("Error: No relevant document IDs found in the test queries.")
        return None
    print(f"Found {len(unique_ids)} unique document IDs in the test set.")

    # Load main dataframe and filter by unique IDs
    print(f"Loading data from: {csv_path}")
    try:
        df_full = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading main data file {csv_path}: {e}")
        return None
        
    print(f"Filtering DataFrame to include only the {len(unique_ids)} unique IDs...")
    df_filtered = df_full[df_full[id_column].isin(unique_ids)].copy()
    df_filtered.reset_index(drop=True, inplace=True) 
    
    if df_filtered.empty:
        print("Error: No matching documents found in the main CSV for the IDs in the test set.")
        return None
    print(f"Filtered DataFrame contains {len(df_filtered)} records.")
    
    filtered_idx_to_id = pd.Series(df_filtered[id_column].values, index=df_filtered.index).to_dict()

    print(f"Using text column: {text_column}")
    texts_filtered = df_filtered[text_column].fillna('').tolist()
    
    print("Calculating FastText embeddings for the filtered subset...")
    fasttext_matrix_filtered = vector_search.batch_generate_embeddings(texts_filtered, show_progress=True)
    print(f"Generated FastText matrix of shape: {fasttext_matrix_filtered.shape}")

    n_docs_filtered = len(df_filtered)
    max_pairs = n_docs_filtered * (n_docs_filtered - 1) // 2
    actual_sample_size = min(sample_pairs_limit, max_pairs)
    print(f"Maximum possible pairs in subset: {max_pairs}")
    print(f"Sampling up to {actual_sample_size} pairs from the subset for analysis...")
        
    similarity_scores_list = [] 

    # Generate all possible pair indices (upper triangle)
    indices = np.triu_indices(n_docs_filtered, k=1)
    num_total_pairs_filtered = len(indices[0])

    # Randomly sample indices if needed
    if actual_sample_size < num_total_pairs_filtered:
        sampled_indices_flat = random.sample(range(num_total_pairs_filtered), actual_sample_size)
        row_indices = indices[0][sampled_indices_flat]
        col_indices = indices[1][sampled_indices_flat]
    else:
        row_indices = indices[0]
        col_indices = indices[1]
        actual_sample_size = num_total_pairs_filtered

    #Calculate Cosine Similarity for the sampled pairs
    print(f"Calculating cosine similarity for {len(row_indices)} sampled pairs...")
    vec1 = fasttext_matrix_filtered[row_indices]
    vec2 = fasttext_matrix_filtered[col_indices]
    # Cosine similarity for normalized vectors is just the dot product
    similarity_scores_list = np.sum(vec1 * vec2, axis=1).tolist()
    # Store pairs with their calculated scores (using filtered indices)
    pairs = list(zip(row_indices, col_indices, similarity_scores_list))
    print(f"Calculated similarities for {len(pairs)} pairs.")

    if not similarity_scores_list:
        print("Error: No similarity scores calculated for the sampled pairs.")
        return None

    # Perform Analysis (Distribution, Percentiles, Sample Inspection)
    plt.figure(figsize=(10, 6))
    plt.hist(similarity_scores_list, bins=50, alpha=0.75)
    plt.xlabel('FastText Cosine Similarity Score (Test Subset)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Similarity Scores (FastText Cosine - Test Subset)') 
    plt.grid(True)
    plt.savefig('fasttext_subset_similarity_distribution.png')
    plt.close()
    print(f"Subset similarity score distribution saved to fasttext_subset_similarity_distribution.png")


    percentiles = [50, 75, 90, 95, 97, 99]
    thresholds = []
    if similarity_scores_list:
         thresholds = [np.percentile(similarity_scores_list, p) for p in percentiles]
    
    print("\nPercentile Analysis (FastText Cosine - Test Subset):") # Updated Label
    if thresholds:
        for p, t in zip(percentiles, thresholds):
            print(f"{p}th percentile: {t:.4f}")
    else:
         print("Could not calculate percentiles.")
    
    ranges = [
        (0.95, 1.01, "Very High Similarity (0.95-1.0)"),
        (0.9, 0.95, "High Similarity (0.9-0.95)"),
        (0.8, 0.9, "Moderate Similarity (0.8-0.9)"),
        (0.7, 0.8, "Low Similarity (0.7-0.8)"),
        (0.5, 0.7, "Very Low Similarity (0.5-0.7)"),
        (0.0, 0.5, "Dissimilar (0.0-0.5)")
    ]
    
    print("\nSample Pairs Analysis (FastText Cosine - Test Subset):") 
    for min_sim, max_sim, label in ranges:
        range_pairs = [(i_filt, j_filt, sim) for i_filt, j_filt, sim in pairs if min_sim <= sim < max_sim]

            
        sampled_pairs_indices = random.sample(range_pairs, min(5, len(range_pairs)))
        
        print(f"\n{label} - Sample Size: {len(range_pairs)}")
        for i_filt, j_filt, sim in sampled_pairs_indices[:2]:
            original_id_i = filtered_idx_to_id.get(i_filt, 'N/A')
            original_id_j = filtered_idx_to_id.get(j_filt, 'N/A')

            
            print(f"Job 1 (ID {original_id_i})" + " ")
            print(f"Job 2 (ID {original_id_j})"+ " ")
            print(f"Similarity: {sim:.4f}")
            print(f"Suggestion: {'Likely Duplicate' if sim > 0.9 else ('Potentially Related' if sim > 0.8 else 'Likely Different')}")
            print("\n")
            
    # Determine optimal threshold based on analysis of the subset
    optimal_threshold = 0.9 
    if thresholds:
        p95_threshold = thresholds[percentiles.index(95)]
        if p95_threshold > 0.95:
            optimal_threshold = 0.95
        elif p95_threshold < 0.85:
            optimal_threshold = 0.9 
        else:
            optimal_threshold = p95_threshold
    
    print(f"\nRecommended FastText Cosine Similarity threshold (based on test subset): {optimal_threshold:.4f}")
    if thresholds:
        print(f"This threshold corresponds to roughly the {percentiles[np.argmin(np.abs(np.array(thresholds) - optimal_threshold))]}th percentile of similarities within the test subset.")
    else:
         print("Could not determine percentile correspondence.")

    return optimal_threshold

def evaluate_search_system(vector_search, test_queries, df=None, top_k=15):
    print(f"Starting evaluation with {len(test_queries)} test queries")

    precision_scores = [] 
    recall_scores = []
    ap_scores = []  
    error_count = 0
    
    query_iterator = tqdm(test_queries.items(), total=len(test_queries), desc="Evaluating Queries")

    for query_idx, (query_text, relevant_docs) in enumerate(query_iterator):
        if not relevant_docs:
            continue
            
        results = vector_search.search(query_text=query_text, top_k=top_k)
        if results is None:
            error_count += 1
            continue
        if not results: 
             continue
            
        if not results[0]:
            precision = 0
            recall = 0
            ap = 0
            precision_scores.append(precision)
            recall_scores.append(recall)
            ap_scores.append(ap)
            continue
            
        formatted_results = vector_search.format_search_results(results, df)
        if not formatted_results or not formatted_results[0]:
            error_count += 1
            continue
            
        retrieved_docs = [hit['id'] for hit in formatted_results[0]]
        
        relevant_docs_set = set(map(int, relevant_docs)) 
        retrieved_docs_set = set(map(int, retrieved_docs))

        relevant_retrieved = retrieved_docs_set & relevant_docs_set
        
        precision = len(relevant_retrieved) / len(retrieved_docs_set) if retrieved_docs_set else 0
        precision_scores.append(precision)
        
        recall = len(relevant_retrieved) / len(relevant_docs_set) if relevant_docs_set else 0
        recall_scores.append(recall)
        
        ap = 0
        hits = 0
        
        for i, doc_id in enumerate(retrieved_docs):
             doc_id_int = int(doc_id)
             if doc_id_int in relevant_docs_set:
                hits += 1
                precision_at_k = hits / (i + 1)  
                ap += precision_at_k
        
        ap = ap / len(relevant_docs_set) if relevant_docs_set else 0
        ap_scores.append(ap)
    
    if error_count > 0:
        print(f"Encountered {error_count} errors during search execution.")

    num_evaluated = len(precision_scores)
    if num_evaluated == 0:
         print("\nError: No queries were successfully evaluated. Cannot calculate metrics.")
         return {
             'precision': 0,
             'recall': 0,
             'map': 0,
             'f1_score': 0,
             'num_queries': len(test_queries),
             'queries_evaluated': 0
         }

    avg_precision = sum(precision_scores) / num_evaluated
    avg_recall = sum(recall_scores) / num_evaluated
    map_score = sum(ap_scores) / num_evaluated
    
    f1_score = 0
    if (avg_precision + avg_recall) > 0:
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
    
    return {
        'precision': avg_precision,
        'recall': avg_recall,
        'map': map_score,
        'f1_score': f1_score,
        'num_queries': len(test_queries),
        'queries_evaluated': num_evaluated
    }

def main():
    
    vs = VectorSearch(collection_name=COLLECTION_NAME) 

    test_queries_path = 'data/search_test_dataset.json'
    print(f"Loading Test Queries from {test_queries_path}")
    try:
        with open(test_queries_path, 'r') as f:
            test_queries = json.load(f)
        print(f"Loaded {len(test_queries)} test queries for analysis and evaluation.")
        if not test_queries:
             print("Warning: Test queries file is empty.")
             # Exit if no queries to analyze or evaluate
             return 
    except FileNotFoundError:
         print(f"Error: {test_queries_path} not found. Cannot proceed.")
         return
    except json.JSONDecodeError:
         print(f"Error: Could not decode {test_queries_path}. File might be corrupted.")
         return
    except Exception as e:
         print(f"An unexpected error occurred loading test queries: {e}")
         return

    
    model_path = "data/fasttext_model.bin"
    print(f"Loading FastText Model for Analysis from {model_path}")
    if not vs.load_model(model_path):
        print("Failed to load FastText model for threshold analysis, exiting.")
        return 

    
    cosine_threshold = analyze_similarity_thresholds(test_queries, vs)
    if cosine_threshold is not None:
        print(f"Similarity threshold analysis complete. Recommended threshold: {cosine_threshold:.4f}")
        config_filename = 'fasttext_cosine_subset_threshold_config.json'
        try:
            with open(config_filename, 'w') as f:
                json.dump({'fasttext_cosine_similarity_threshold_subset': float(cosine_threshold)}, f, indent=2)
            print(f"Saved FastText Cosine threshold configuration to {config_filename}")
        except Exception as e:
             print(f"Error saving FastText Cosine threshold config: {e}")
    else:
        print("Could not determine FastText Cosine threshold based on the test subset.")
        
    print("Starting Vector Search Evaluation")

    # Print sample queries again before evaluation part
    sample_queries = list(test_queries.items())[:3]
    for query_text, relevant_docs in sample_queries:
        query_preview = str(query_text)[:100] + '...' if isinstance(query_text, str) else '[Non-string query key]...'
        print(f"Sample query: '{query_preview}'") 
        print(f"  Has {len(relevant_docs)} relevant docs: {relevant_docs[:5]}...")
    
    # Connect to Milvus
    if not vs.connect_to_milvus():
        print("Failed to connect to Milvus, exiting evaluation.")
        return
        
        
    # Connect to existing Collection
    if not utility.has_collection(vs.collection_name):
         print(f"Collection '{vs.collection_name}' does not exist. Please run embedding generation first.")
         vs.disconnect()
         return
    else:
         try:
             vs.collection = Collection(name=vs.collection_name)
             print(f"Using existing collection: '{vs.collection_name}' with {vs.collection.num_entities} entities.")
         except Exception as e:
              print(f"Error connecting to existing collection '{vs.collection_name}': {e}")
              vs.disconnect()
              return

    # Load collection for searching
    if not vs.load_collection():
        print("Failed to load collection into memory, exiting evaluation.")
        vs.disconnect()
        return
        
    # Load dataframe for result formatting
    df = vs.load_data() 
    if df is None:
        print("Could not load DataFrame")
        df = pd.DataFrame()

    # Perform evaluation
    if test_queries:
         evaluation_results = evaluate_search_system(vs, test_queries, df, top_k=20)
         print("\nSearch system evaluation results:")
         print(f"Precision: {evaluation_results['precision']:.4f}")
         print(f"Recall: {evaluation_results['recall']:.4f}")
         print(f"MAP: {evaluation_results['map']:.4f}")
         print(f"F1 score: {evaluation_results['f1_score']:.4f}")
         print(f"Evaluated queries: {evaluation_results['queries_evaluated']}/{evaluation_results['num_queries']}")
    else:
         print("\nSkipping evaluation as no test queries were loaded or available.")
    
    # Clean up Milvus resources
    vs.release_collection() 
    vs.disconnect()
    print("\nEvaluation complete.")

if __name__ == "__main__":
    main()
    



