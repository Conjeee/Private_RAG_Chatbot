import os
from . import VectorDB
from . import LLMEngine

class RAGPipeline:
    def __init__(self):
        self.db = VectorDB()
        self.llm = LLMEngine()

    def get_answer(self, query: str):
        """
        Returns a tuple: (answer_string, source_documents)
        """
        print(f"🔎 Retrieving context for: '{query}'")
        docs = self.db.search(query, k=5)
        
        if not docs:
            return "I couldn't find any information about that in the documents.", []

        # 1. Build Context
        context_text = "\n\n---\n\n".join([doc.page_content for doc in docs])
        
        # 2. Build Prompt
        # Improved Prompt Engineering
        prompt = f"""
        <|system|>
        You are an expert Research Assistant. You answer questions strictly based on the provided context.
        
        Rules:
        1. Use ONLY the context below. Do not use outside knowledge.
        2. If the answer is not in the context, say: "I cannot find that information in the documents."
        3. Keep the answer concise and professional.
        
        CONTEXT:
        {context_text}
        <|end|>
        
        <|user|>
        {query}
        <|end|>
        """

        # 3. Generate Answer
        print("🤖 Generating answer...")
        response = self.llm.generate(prompt)
        
        # 4. Return both the Answer AND the Source Documents
        return response, docs

if __name__ == "__main__":
    rag = RAGPipeline()
    answer, sources = rag.get_answer("What is the main topic?")
    print("\nAI:", answer)
    print("\nSources:", sources)