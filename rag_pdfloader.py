import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
import tempfile
import os

# Application configuration and layout setup
st.set_page_config(page_title="🔍 Multi-Model PDF Intelligence Assistant", page_icon="📚")
st.header("🔍 Multi-Model PDF Intelligence Assistant")
st.subheader("Transform your documents into interactive conversations using multiple AI models")

# Model selection
model_option = st.selectbox(
    "🤖 Select AI Model",
    ["GPT-4 (Most Capable)", "GPT-3.5-Turbo (Faster)", "GPT-4-Turbo (Balanced)"]
)

# File upload interface
document_file = st.file_uploader("📁 Select your PDF document", type=["pdf"])

if document_file:
    # Create temporary file for processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(document_file.read())
        file_path = temp_file.name

    st.success("🎉 Document processed and ready for interaction!")

    # Document processing and vector database creation
    with st.spinner("⚡ Analyzing document content and building knowledge base..."):
        # Load PDF content
        pdf_loader = PyPDFLoader(file_path)
        raw_documents = pdf_loader.load()

        # Enhanced text splitting with recursive approach
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=250,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        document_chunks = text_splitter.split_documents(raw_documents)

        # Create embeddings using text-embedding-3-small for better semantic search
        embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_database = Chroma.from_documents(document_chunks, embeddings_model)
        document_retriever = vector_database.as_retriever(search_kwargs={"k": 8})

        # Initialize language model based on selection
        model_map = {
            "GPT-4 (Most Capable)": "gpt-4",
            "GPT-3.5-Turbo (Faster)": "gpt-3.5-turbo",
            "GPT-4-Turbo (Balanced)": "gpt-4-turbo-preview"
        }
        selected_model = model_map[model_option]
        
        language_model = ChatOpenAI(
            model=selected_model,
            temperature=0.1,
            streaming=True
        )
        
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=language_model,
            retriever=document_retriever,
            return_source_documents=True
        )

    # Session state initialization for conversation history
    if "conversation_memory" not in st.session_state:
        st.session_state.conversation_memory = []

    # User input interface
    user_query = st.text_input("💭 What would you like to know about your document?")

    if user_query:
        with st.spinner("🤔 Processing your question..."):
            chain_response = conversation_chain.invoke({
                "question": user_query,
                "chat_history": st.session_state.conversation_memory
            })
            st.session_state.conversation_memory.append((user_query, chain_response["answer"]))

    # Conversation display with enhanced formatting
    if st.session_state.conversation_memory:
        st.markdown("### 💬 Conversation History")
        for query, response in reversed(st.session_state.conversation_memory):
            with st.container():
                st.markdown(f"**👤 Your Question:** {query}")
                st.markdown(f"**🤖 Assistant Response:** {response}")
                st.divider()

    # Cleanup temporary resources
    os.remove(file_path)
else:
    st.info("📤 Please upload a PDF document to start the conversation.")