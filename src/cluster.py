from sklearn.cluster import KMeans
import hdbscan
import numpy as np
import umap
import streamlit as st

def cluster_column_embeddings(column_embeddings):
    # Initialize HDBSCAN with minimum cluster size
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3, cluster_selection_epsilon=0.5) #gen_min_span_tree=True)
    
    # Fit the HDBSCAN clusterer
    clusterer.fit(column_embeddings)
    
    return clusterer.labels_

def cluster_column_embeddings_umap(column_embeddings):
    # Reduce dimensions with UMAP
    umap_model = umap.UMAP(n_components=10, random_state=42)
    umap_embeddings = umap_model.fit_transform(column_embeddings)
    
    # Initialize HDBSCAN with adjusted parameters
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3)
    
    # Fit the HDBSCAN clusterer on the UMAP-reduced data
    clusterer.fit(umap_embeddings)
    
    return clusterer.labels_

def cluster_column_embeddings_kmeans(column_embeddings, name=''):
    # Using the elbow method to find the optimal number of clusters
    wcss = []  # Within-cluster sum of squares
    max_clusters = 10  # You might want to set this to a different maximum number of clusters

    # Calculating the within-cluster sum of squares for different numbers of clusters
    for i in range(1, max_clusters+1):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
        kmeans.fit(column_embeddings)
        wcss.append(kmeans.inertia_)  # inertia_ is the within-cluster sum-of-squares

    # Plotting the results to visualize the 'elbow'
    plt.figure(figsize=(10,5))
    plt.plot(range(1, max_clusters+1), wcss, marker='o', linestyle='--')
    plt.title(f'{name} Elbow Method For Optimal k')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

    # You need to choose the optimal number of clusters manually by looking at the plot
    # Let's say you chose the number of clusters to be 'n', you'd then proceed with:
    optimal_n = 4  # Replace with the number of clusters you've determined to be optimal
    kmeans = KMeans(n_clusters=optimal_n, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(column_embeddings)

    return kmeans.labels_


@st.cache_data(show_spinner=False)
def cluster_and_append(df_original, embeddings_column):
    df = df_original.copy()

    # Convert lists of embeddings to a 2D NumPy array
    embeddings_array = np.array(df[embeddings_column].tolist())
    
    # Ensure embeddings_array is 2D
    if len(embeddings_array.shape) == 1:
        # If the array is 1D, it means each embedding is a single number
        # Reshape it to be 2D with one column
        embeddings_array = embeddings_array.reshape(-1, 1)
    
    # Get cluster labels
    cluster_labels = cluster_column_embeddings_umap(embeddings_array)
    
    # Create new column name for cluster labels
    new_column_name = f'{embeddings_column}_cluster_id'
    
    # Assign cluster labels to new column in DataFrame
    df[new_column_name] = cluster_labels
    
    return df  # Optional: Return the modified DataFrame

def find_closest_to_centroid(df, top_N, embeddings_column, cluster_column, text_column):
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
        closest_data[cluster_id] = {
            'indices': closest_indices,
            'texts': closest_texts
        }
    
    return closest_data