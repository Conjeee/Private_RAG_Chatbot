import os
from langchain_community.vectorstores import LanceDB
from langchain_huggingface import HuggingFaceEmbeddings

class VectorDB:
    def __init__(self, db_path="./storage"):
        # Define the local path for the embedding model
        embedding_path = "./models/all-MiniLM-L6-v2"
        
        # Fallback logic: If you forgot to run setup.py, it tries to download from internet
        if not os.path.exists(embedding_path):
            print(f"⚠️ Local model not found at {embedding_path}. Downloading from Hub...")
            model_source = "sentence-transformers/all-MiniLM-L6-v2"
        else:
            print(f"📂 Loading Embedding Model from local disk: {embedding_path}")
            model_source = embedding_path

        # 1. Load the Embedding Model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=model_source,  # Now points to local folder
            model_kwargs={'device': 'cpu'} 
        )

        # 2. Connect to LanceDB
        self.vector_store = LanceDB(
            uri=db_path,
            embedding=self.embedding_model,
        )
        print("✅ Vector Database Connected!")

    def add_documents(self, docs):
        print(f"📥 Adding {len(docs)} document chunks to database...")
        self.vector_store.add_documents(docs)
        print("✅ Documents saved!")

    def search(self, query: str, k: int = 3):
        print(f"🔎 Searching for: '{query}'")
        results = self.vector_store.similarity_search(query, k=k)
        return results