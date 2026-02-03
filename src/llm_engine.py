import os
import sys
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler

class LLMEngine:
    def __init__(self, model_path="./models/Phi-3-mini-4k-instruct-q4.gguf"):
        """
        Initializes the local LLM using LlamaCPP optimized for Apple Silicon.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model not found at {model_path}. Did you run setup.py?")

        print(f"⏳ Loading LLM from {model_path}... (This fits in your 8GB RAM)")

        # Callback manager handles the streaming output (text appearing piece by piece)
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

        # Initialize the model
        self.llm = LlamaCpp(
            model_path=model_path,
            n_gpu_layers=-1,      # -1 = Offload ALL layers to Mac GPU (Metal)
            n_ctx=4096,           # Context window size
            temperature=0.1,      # Low temp = more factual
            callback_manager=callback_manager,
            verbose=True,         
            f16_kv=True,          # Metal optimization
        )
        print("✅ LLM Loaded successfully!")

    def generate(self, prompt: str) -> str:
        """
        Sends a prompt to the model and returns the response.
        """
        return self.llm.invoke(prompt)

if __name__ == "__main__":
    try:
        engine = LLMEngine()
        print("\n--- Test Response ---")
        engine.generate("Explain in one sentence why local privacy matters.")
        print("\n---------------------")
    except Exception as e:
        print(f"Error: {e}")