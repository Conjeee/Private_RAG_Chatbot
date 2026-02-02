import os
from langchain_community.vectorstores import LanceDB
from langchain_huggingface import HuggingFaceEmbeddings

class VectorDB:
    def __init__(self, db_path="./storage"):
        """
        Initializes the local Vector Database (LanceDB).
        """
        print("⏳ Loading Embedding Model (all-MiniLM-L6-v2)...")
        
        # 1. Load the Embedding Model
        # This converts text into numbers. It runs locally on CPU/GPU.
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'} 
        )

        # 2. Connect to LanceDB
        # This creates a table in your './storage' folder
        self.vector_store = LanceDB(
            uri=db_path,
            embedding=self.embedding_model,
        )
        print("✅ Vector Database Connected!")

    def add_documents(self, docs):
        """
        Takes a list of LangChain Documents and saves them to the DB.
        """
        print(f"📥 Adding {len(docs)} document chunks to database...")
        self.vector_store.add_documents(docs)
        print("✅ Documents saved!")

    def search(self, query: str, k: int = 3):
        """
        Searches for the most similar documents to the query.
        k = number of results to return
        """
        print(f"🔎 Searching for: '{query}'")
        results = self.vector_store.similarity_search(query, k=k)
        return results

# --- Test Block ---
if __name__ == "__main__":
    # --- FIX: New Import Path for Document ---
    from langchain_core.documents import Document
    # -----------------------------------------
    
    try:
        db = VectorDB()
        
        # Test Data
        test_docs = [
            Document(page_content="The secret password is 'Blueberry'."),
            Document(page_content="The sun rises in the east."),
        ]
        
        # 1. Test Saving
        db.add_documents(test_docs)
        
        # 2. Test Searching
        results = db.search("What is the password?")
        print("\n--- Search Result ---")
        print(results[0].page_content)
        print("---------------------")
        
    except Exception as e:
        print(f"❌ Error: {e}")