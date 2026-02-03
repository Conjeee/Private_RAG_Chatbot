import os
import streamlit as st
from src import RAGPipeline

st.set_page_config(page_title="Private RAG Chatbot", page_icon="🤖")

st.title("🤖 Private Doc Chatbot")

@st.cache_resource
def load_pipeline():
    return RAGPipeline()

with st.spinner("⏳ Loading Brain & Memory..."):
    rag = load_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📚 View Sources"):
                for source in message["sources"]:
                    # Create a clickable link
                    # Streamlit serves files from 'static' at the URL path 'app/static/'
                    file_name = source["source"]
                    page_num = source["page"] + 1
                    url = f"app/static/{file_name}#page={page_num}"
                    
                    st.markdown(f"[[Open PDF on Page {page_num}]]({url})")
                    st.markdown(f"> *{source['content'][:150]}...*")

# Handle input
if prompt := st.chat_input("Ask a question..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response, docs = rag.get_answer(prompt)
            st.markdown(response)
            
            source_data = []
            if docs:
                with st.expander("📚 View Sources"):
                    for doc in docs:
                        # Extract filename only (removes directory path)
                        full_path = doc.metadata.get("source", "Unknown")
                        file_name = os.path.basename(full_path)
                        page_num = doc.metadata.get("page", 0) + 1
                        snippet = doc.page_content
                        
                        # --- THE MAGIC LINK ---
                        # We point to 'app/static/' which Streamlit maps to your local 'static' folder
                        link = f"app/static/{file_name}#page={page_num}"
                        
                        # Render Link + Snippet
                        st.markdown(f"📄 **[{file_name} (Page {page_num})]({link})**")
                        st.markdown(f"> *{snippet[:150]}...*")
                        
                        source_data.append({
                            "source": file_name,
                            "page": doc.metadata.get("page", 0),
                            "content": snippet
                        })

    st.session_state.messages.append({
        "role": "assistant", 
        "content": response,
        "sources": source_data
    })