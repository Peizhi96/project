# project

## Set up environment
```pip install -r requirements.txt```

## Set up Milvus
1. Create a docker-compose.yml
2. Run ```docker compose up -d```
3. Create Milvus container

## EDA and embedding notebook
1. Do basic EDA
2. Clean the job description colum
3. Fasttext embedding model
4. Connect milvus, create colletion, create index, insert data, load collection, search.

## Vector_search script
```python3 vector_search.py```

## Sample output 
```
Job 1 (ID 4423) 
Job 2 (ID 60739) 
Similarity: 0.9936


Job 1 (ID 33916) 
Job 2 (ID 46447) 
Similarity: 0.9527


Job 1 (ID 14173) 
Job 2 (ID 80946) 
Similarity: 0.9223


Job 1 (ID 4860) 
Job 2 (ID 28946) 
Similarity: 0.9084


Job 1 (ID 29989) 
Job 2 (ID 69884) 
Similarity: 0.8251

Job 1 (ID 11810) 
Job 2 (ID 66429) 
Similarity: 0.8695


Job 1 (ID 56314) 
Job 2 (ID 91585) 
Similarity: 0.7754


Job 1 (ID 63092) 
Job 2 (ID 63278) 
Similarity: 0.7324


Job 1 (ID 38799) 
Job 2 (ID 86570) 
Similarity: 0.6551


Job 1 (ID 64013) 
Job 2 (ID 78020) 
Similarity: 0.6562


Job 1 (ID 81296) 
Job 2 (ID 93286) 
Similarity: 0.1184

Job 1 (ID 61453) 
Job 2 (ID 65900) 
Similarity: 0.4867
```

