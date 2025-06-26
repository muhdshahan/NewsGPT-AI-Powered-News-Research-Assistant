import streamlit as st 
import pickle
import time
from langchain_community.llms import Together 
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS  # FAISS is the vector store (for similarity search)
from secret_key import togetherapi_key
from langchain.embeddings import HuggingFaceEmbeddings

# Set Together API key in environment variables
import os
os.environ['TOGETHER_API_KEY'] = togetherapi_key

st.title("NewsGPT: Your AI-Powered News Research Assistant 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)
    
process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_together.pkl"  # file_path to save the FAISS vectorstore locally

main_placeholder = st.empty()  # Placeholder for dynamic messages on main screen

# Mistral 7B Instruct model from Together for Q&A
llm = Together(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    temperature=0.7,
    max_tokens=128
)

if process_url_clicked:
    # load the webpage content using UnstructuredURLLoader
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    docs = text_splitter.split_documents(data)
    # create embeddings and save it to FAISS index
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Create FAISS vector index from the document chunks
    vectorstore_mistral = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)
    
    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_mistral, f)
        
query = main_placeholder.text_input("Question: ")
if query:
    # Check if FAISS index file exits
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            # Create retrieval-based QA chain using the LLM + vector retriever
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            # Run query through the chain
            result = chain({"question": query}, return_only_outputs=True)
            # {"answer": "", "sources": []}
            st.header("Answer")
            st.write(result["answer"])
            
            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  
                for source in sources_list:
                    st.write(source)
                    