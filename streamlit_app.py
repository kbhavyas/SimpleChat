import streamlit as st
import os
import numpy as np
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from rank_bm25 import BM25Okapi  # BM25 retrieval
from langchain_cohere import CohereRerank  # Re-ranking
from sentence_transformers import SentenceTransformer
import faiss

import getpass
import os


if "COHERE_API_KEY" not in os.environ:
    os.environ["COHERE_API_KEY"] = "ffrkCXytXzNJ0XoA1LqJCSmk0ZhAhY9aq33q5msI"


# Initialize Embeddings & LLM
EMBEDDING_MODEL = "all-minilm:latest"
LLM_MODEL = "llama3:latest"
BASE_URL = "http://localhost:11434"

def process_pdf(pdf_path):
    """Loads and splits a PDF into text chunks."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return text_splitter.split_documents(documents)

def create_vector_db(chunks):
    """Stores document embeddings in ChromaDB."""
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [doc.page_content for doc in chunks]  # Ensure doc.page_content is a string

# Convert texts to embeddings
    embeddings = np.array(embedding_model.encode(texts), dtype=np.float32)  # Convert to float32

# Create FAISS index
    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 (Euclidean) distance
    faiss_index.add(embeddings)  # Add vectors to index
    return faiss_index, embedding_model

def initialize_bm25(chunks):
    """Initializes BM25 with text chunks."""
    tokenized_corpus = [doc.page_content.split() for doc in chunks]
    return BM25Okapi(tokenized_corpus), chunks

def retrieve_documents(user_query, vector_db, bm25, bm25_chunks):
    """Performs hybrid retrieval (BM25 + embeddings) and re-ranks results."""
    bm25_scores = bm25.get_scores(user_query.split())
    bm25_top_k = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)[:50]
    bm25_docs = [bm25_chunks[idx].page_content for idx, _ in bm25_top_k]
    

#    embedding_results = vector_db.similarity_search(user_query, k=50)
    embedding_results = vector_db.search(user_query, k=50)
    dense_docs = [doc.page_content for doc in embedding_results]

    
    combined_docs = list(set(bm25_docs + dense_docs))  # Merge results
    
    reranker = CohereRerank(model='rerank-english-v3.0')
    reranked_results = reranker.rerank(combined_docs, user_query)

    return [combined_docs[dict['index']] for dict in reranked_results][:5] # Return top 5 re-ranked docs

def ask_question(user_query, vector_db, bm25, bm25_chunks, llm):
    """Retrieves and generates a response for the given question."""
    retrieved_docs = retrieve_documents(user_query, vector_db, bm25, bm25_chunks)
    print("ret-----", retrieved_docs)
    context = "\n\n".join(retrieved_docs)
    
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="{context} ---- Given the context above, answer the following question with given context only : {question}"
    )
    final_prompt = prompt_template.format(context=context, question=user_query)
    
    return llm(final_prompt)



st.title("🔍 ChromaQA: Financial Document Search")
pdf_path = "10-Q4-2024-As-Filed.pdf"




st.info("📖 Processing document...")
chunks = process_pdf(pdf_path)
vector_db, embedding_model = create_vector_db(chunks)
bm25, bm25_chunks = initialize_bm25(chunks)
llm = Ollama(model=LLM_MODEL, base_url=BASE_URL)

st.success("✅ Data successfully ingested into ChromaDB!")
st.subheader("💬 Ask a Question")
user_query = st.text_input("Type your question here...")

if st.button("Get Answer"):
    if user_query:
        with st.spinner("🔍 Searching for the best answer..."):
            answer = ask_question(user_query, vector_db, bm25, bm25_chunks, llm)
        st.success("✅ Answer retrieved!")
        st.write("**🔹 Answer:**", answer)
    else:
        st.warning("⚠️ Please enter a question.")
