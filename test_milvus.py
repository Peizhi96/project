import os
from dotenv import load_dotenv
from pymilvus import connections

load_dotenv()
try:
    load_dotenv(override=True)
    MILVUS_HOST = os.getenv("MILVUS_HOST")
    MILVUS_PORT = os.getenv("MILVUS_PORT")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME")

    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    print("Successfully connected to Milvus.")

except Exception as e:
    print(f"Error connecting to Milvus: {e}")
finally:
    connections.disconnect("default")  
