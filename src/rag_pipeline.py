from vector_db import VectorDB
from llm_engine import LLMEngine

class RAGPipeline:
    def __init__(self):
        # Initialize the two components we built earlier
        self.db = VectorDB()
        self.llm = LLMEngine()

    def get_answer(self, query: str):
        """
        The Core Loop: Search -> Prompt -> Answer
        """
        # 1. Search for relevant context
        print(f"🔎 Retrieving context for: '{query}'")
        docs = self.db.search(query, k=3) # Get top 3 matching chunks
        
        if not docs:
            return "I couldn't find any information about that in the documents."

        # 2. Build the Context String
        # We combine the text from the top 3 results
        context_text = "\n\n---\n\n".join([doc.page_content for doc in docs])
        
        # 3. Construct the Prompt
        # This specific format helps the model distinguish instructions from data
        prompt = f"""
        You are a helpful assistant. Use the following pieces of context to answer the question at the end.
        If the answer is not in the context, say that you don't know. Don't make up facts.

        CONTEXT:
        {context_text}

        QUESTION:
        {query}

        ANSWER:
        """

        # 4. Generate Answer
        print("🤖 Generating answer...")
        response = self.llm.generate(prompt)
        
        return response

if __name__ == "__main__":
    # Test the full pipeline
    rag = RAGPipeline()
    
    # Change this question to something actually inside your PDF!
    question = "What is the main topic of this document?"
    
    print("\n\nUser: " + question)
    answer = rag.get_answer(question)
    print("\n\nAI: " + answer)