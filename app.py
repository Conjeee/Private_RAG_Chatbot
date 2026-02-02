import streamlit as st
from src.rag_pipeline import RAGPipeline

st.set_page_config(page_title="Private RAG Chatbot", page_icon="🤖")

st.title("🤖 Private Doc Chatbot")

# --- 1. Load the Brain (Once) ---
@st.cache_resource
def load_pipeline():
    """
    Initialize the RAG pipeline only once to save RAM/Time.
    """
    return RAGPipeline()

# Show a spinner while loading (only happens on first run)
with st.spinner("⏳ Loading Brain & Memory... (This takes a few seconds)"):
    rag = load_pipeline()

# --- 2. Initialize Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 3. Display Previous Messages ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 4. Handle User Input ---
if prompt := st.chat_input("Ask a question about your documents..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag.get_answer(prompt)
            st.markdown(response)
    
    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": response})