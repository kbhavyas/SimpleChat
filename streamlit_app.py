import streamlit as st
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

#uploaded = files.upload()
file_path =  "financial_data_21_23.xlsx" # Get the uploaded file name

def load_and_clean_data(file_path):
    # Load Excel file
    df = pd.read_excel(file_path)

    # Standardizing column names (removing spaces, converting to lowercase)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # Converting date column to datetime format
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Removing commas and converting numeric columns to float
    numeric_cols = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
    for col in numeric_cols:
        df[col] = df[col].astype(str).str.replace(',', '')  # Remove commas
        df[col] = pd.to_numeric(df[col], errors='coerce')   # Convert to float

    # Drop rows with missing values
    df = df.dropna()

    # Sorting by date (most recent first)
    df = df.sort_values(by='date', ascending=False)

    return df

# Load and clean the data
df_cleaned = load_and_clean_data(file_path)

# Display first few rows
df_cleaned.head()

# Convert financial data to text chunks
text_chunks = df_cleaned.astype(str).apply(lambda row: ' '.join(row), axis=1).tolist()

# Load pre-trained embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
embeddings = np.array(model.encode(text_chunks))

# Initialize FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)


def load_llm():
    model_name = "distilbert/distilgpt2"  # Change this for other models
    token = "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"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token = token, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_name, token = token)
    return model, tokenizer
#Rerank combined results
def rerank_results(query, results):
    pairs = [(query, doc) for doc in results]
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    scores = reranker.predict(pairs)
    ranked_results = [doc for _, doc in sorted(zip(scores, results), reverse=True)]
    return ranked_results

# Function to retrieve similar documents
def retrieve_similar(query, top_k=5):
    query_embedding = np.array(model.encode([query]))
    distances, indices = index.search(query_embedding, top_k)
    return [text_chunks[i] for i in indices[0]]


# Initialize BM25 for keyword-based search
tokenized_chunks = [chunk.split() for chunk in text_chunks]
bm25 = BM25Okapi(tokenized_chunks)

def retrieve_similar(query, top_k=5):
    # Embedding-based retrieval
    query_embedding = np.array(model.encode([query]))
    distances, indices = index.search(query_embedding, top_k)
    embedding_results = [text_chunks[i] for i in indices[0]]

    # BM25 retrieval
    bm25_scores = bm25.get_scores(query.split())
    bm25_top_indices = np.argsort(bm25_scores)[-top_k:][::-1]
    bm25_results = [text_chunks[i] for i in bm25_top_indices]

    # Combine and rerank results
    combined_results = list(sorted(embedding_results))
    return rerank_results(query, combined_results)[:top_k]



st.title("🔍 Document Retrieval App")
query = st.text_input("Enter your search query:")
llm_model, tokenizer = load_llm()

if query:
    with st.spinner("Retrieving relevant documents..."):
        retrieved_docs = retrieve_similar(query, top_k=5)
        context = "\n\n".join(retrieved_docs)  # Combine retrieved text

        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
        
        with torch.no_grad():
            outputs = llm_model.generate(**inputs, max_length=300)

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.write(response)
