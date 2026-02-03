import os
import sys
# --- FIX: New Import Path for Text Splitters ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
# -----------------------------------------------
from langchain_community.document_loaders import PyPDFLoader
from src.vector_db import VectorDB

class Ingester:
    def __init__(self):
        self.data_path = "./static" 
        
        self.chunk_size = 1000
        self.chunk_overlap = 100
        self.db = VectorDB()

    def run(self):
        # 1. Check for PDFs
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
            print("Please put your PDF files in ./static and run again.")
            return

        files = [f for f in os.listdir(self.data_path) if f.endswith(".pdf")]
        if not files:
            print("❌ No PDF files found in ./static folder.")
            return

        print(f"📖 Found {len(files)} PDFs. Starting ingestion...")
        
        raw_docs = []
        for file in files:
            file_path = os.path.join(self.data_path, file)
            print(f"   -> Loading: {file}...")
            loader = PyPDFLoader(file_path)
            raw_docs.extend(loader.load())

        # 2. Split Text
        print("✂️  Splitting documents...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = text_splitter.split_documents(raw_docs)
        print(f"🧩 Created {len(chunks)} chunks.")

        # 3. Store in DB
        self.db.add_documents(chunks)
        print("🎉 Ingestion Complete!")

if __name__ == "__main__":
    ingester = Ingester()
    ingester.run()