# Embedding Support
import pandas as pd

from src.constants import OPENAI_KEY
# from langchain.embeddings import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer

# def generate_embeddings_openai(txt_series):
#     embedder = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)
#     na_filled = txt_series.fillna("", inplace=False) 
#     # Generate embeddings for the text column
#     return embedder.embed_documents(na_filled.tolist())

import streamlit as st

def generate_embeddings_free(txt_series):
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    na_filled = txt_series.fillna("", inplace=False) 
    # Generate embeddings for the text column
    return embedder.encode(na_filled.tolist())
    

@st.cache_data(show_spinner=False)
def embed_reviews(df, column):
    df[f'{column}_embeddings'] = pd.Series(list(generate_embeddings_free(df[column])))
   
    return df