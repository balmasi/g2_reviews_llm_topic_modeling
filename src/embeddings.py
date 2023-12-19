# Embedding Support

import umap
import pandas as pd
import numpy as np
import streamlit as st

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

@st.cache_data(show_spinner=False)
def get_pca_components_for_variance(embeddings, variance_threshold=0.8):
    """
    Compute the number of principal components needed to explain a given variance threshold.

    Parameters:
        embeddings (numpy.ndarray): The input embeddings to perform PCA on.
        variance_threshold (float, optional): The desired threshold of explained variance.
            Defaults to 0.80.

    Returns:
        int: The number of principal components that explain the given variance threshold.
    """
    pca = PCA()
    pca.fit(embeddings)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # Determine the number of components for the given variance threshold
    n_components = np.where(cumulative_variance >= variance_threshold)[0][0] + 1
    print('Number of components derived from PCA:', n_components)
    return n_components

def generate_embeddings_free(txt_series):
    """
    Generate embeddings for the given text series using a pre-trained SentenceTransformer model.

    Parameters:
        txt_series (pandas.Series): A pandas series containing text data.

    Returns:
        numpy.ndarray: An array of embeddings generated for the text series.
    """
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    na_filled = txt_series.fillna("", inplace=False) 
    # Generate embeddings for the text column
    return embedder.encode(na_filled.tolist())
    

@st.cache_data(show_spinner=False)
def embed_reviews(df, column):
    """
    Caches the result of the function and returns a DataFrame with an additional column that contains the embeddings of the reviews in the specified column.

    Parameters:
    - df (DataFrame): The DataFrame containing the reviews.
    - column (str): The name of the column containing the reviews.

    Returns:
    - df (DataFrame): The modified DataFrame with an additional column containing the embeddings.
    """
    df[f'{column}_embeddings'] = pd.Series(list(generate_embeddings_free(df[column])))
   
    return df

@st.cache_data(show_spinner=False)
def reduce_dimensions_append_array(df, vector_col, num_dimensions=2, dim_col_name="dims"):
    """
    Reduces the dimensions of a DataFrame by applying UMAP algorithm to the specified vector column.

    Parameters:
        df (pandas.DataFrame): The input DataFrame.
        vector_col (str): The name of the column containing the vector data.
        num_dimensions (int, optional): The number of dimensions to reduce the data to. Defaults to 2.
        dim_col_name (str, optional): The name of the column to store the reduced dimensions. Defaults to "dimensions".

    Returns:
        pandas.DataFrame: The DataFrame with the reduced dimensions column added.
    """
    df = df.copy()

    # Extract embeddings from DataFrame
    embeddings = np.array(df[vector_col].tolist())

    # Apply UMAP
    reducer = umap.UMAP(n_components=num_dimensions, random_state=42)
    embeddings_reduced = reducer.fit_transform(embeddings)

    # Assign the reduced dimensions to a new column as an array
    df[dim_col_name] = list(embeddings_reduced)

    return df