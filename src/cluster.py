from sklearn.cluster import KMeans
import hdbscan
import numpy as np
import umap
import streamlit as st


def cluster_column_embeddings(column_embeddings):
    # Initialize HDBSCAN with minimum cluster size
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=5, min_samples=3, cluster_selection_epsilon=0.5
    )

    # Fit the HDBSCAN clusterer
    clusterer.fit(column_embeddings)

    return clusterer.labels_


def cluster_column_embeddings_umap(column_embeddings, n_components=None):
    """
    Generates a UMAP embedding for the given column embeddings and performs clustering using HDBSCAN.

    Args:
        column_embeddings (list): A list of column embeddings.
        n_components (int, optional): The number of dimensions in the UMAP embedding. If not provided, it will be set to the length of column_embeddings.

    Returns:
        list: A list of cluster labels for each column embedding.
    """
    n_components = len(column_embeddings) if n_components is None else n_components
    umap_model = umap.UMAP(n_components, random_state=42)
    umap_embeddings = umap_model.fit_transform(column_embeddings)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3)
    clusterer.fit(umap_embeddings)
    return clusterer.labels_


@st.cache_data(show_spinner=False)
def cluster_and_append(df_original, embeddings_column, n_components=None):
    """
    A function that clusters the embeddings in a DataFrame and appends the cluster labels as a new column.

    Parameters:
    - df_original: The original DataFrame containing the embeddings.
    - embeddings_column: The name of the column in the DataFrame that contains the embeddings.
    - n_components: The number of dimensions to reduce the embeddings to using UMAP. If None, no reduction is performed.

    Returns:
    - df: A copy of the original DataFrame with an additional column containing the cluster labels.
    """
    df = df_original.copy()

    # Convert lists of embeddings to a 2D NumPy array
    embeddings_array = np.array(df[embeddings_column].tolist())

    # Ensure embeddings_array is 2D
    if len(embeddings_array.shape) == 1:
        # If the array is 1D, it means each embedding is a single number
        # Reshape it to be 2D with one column
        embeddings_array = embeddings_array.reshape(-1, 1)

    # Get cluster labels
    cluster_labels = cluster_column_embeddings_umap(embeddings_array, n_components)

    # Create new column name for cluster labels
    new_column_name = f"{embeddings_column}_cluster_id"

    # Assign cluster labels to new column in DataFrame
    df[new_column_name] = cluster_labels

    return df  # Optional: Return the modified DataFrame


def find_closest_to_centroid(df, top_N, embeddings_column, cluster_column, text_column):
    """
    Find the closest embeddings to the centroid of each cluster in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        top_N (int): The number of closest embeddings to find for each cluster.
        embeddings_column (str): The name of the column containing the embeddings.
        cluster_column (str): The name of the column containing the cluster IDs.
        text_column (str): The name of the column containing the corresponding text.

    Returns:
        dict: A dictionary where the keys are the cluster IDs and the values are dictionaries
              with keys 'indices' and 'texts'. 'indices' contains the indices of the closest
              embeddings to the centroid, and 'texts' contains the corresponding text.
    """
    closest_data = {}
    unique_clusters = df[cluster_column].unique()
    for cluster_id in unique_clusters:
        # Filter the data for the current cluster
        cluster_data = df[df[cluster_column] == cluster_id]
        cluster_embeddings = np.array(cluster_data[embeddings_column].tolist())

        # Compute the centroid of the current cluster
        centroid = np.mean(cluster_embeddings, axis=0)

        # Compute the distances from the centroid to each embedding in the cluster
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)

        # Get the indices of the N closest embeddings to the centroid
        closest_indices = np.argsort(distances)[:top_N]

        # Extract the corresponding text from the DataFrame
        closest_texts = cluster_data.iloc[closest_indices][text_column].tolist()

        # Store the closest data
        closest_data[cluster_id] = {"indices": closest_indices, "texts": closest_texts}

    return closest_data
