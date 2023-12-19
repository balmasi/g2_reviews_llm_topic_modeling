import asyncio
from src.extract_topic import summarize_parallel, summarize_sequential

import streamlit as st
import pandas as pd
import numpy as np

from src.preprocess import preprocess_data
from src.text_utils import split_into_sentences
from src.embeddings import embed_reviews, get_pca_components_for_variance, reduce_dimensions_append_array
from src.cluster import cluster_and_append, find_closest_to_centroid
from src.visualize import visualize_embeddings, plot_over_time
from src.ui import radio_filter, range_filter


REVIEW_COL = 'review_text'


# Define a function to process each DataFrame
@st.cache_data(show_spinner=False)
def explode_reviews(df, column_name):
    df = df.copy()
    # Split reviews into sentences
    df[column_name] = df[column_name].astype(str).apply(split_into_sentences)
    
    # Explode the DataFrame and reset the index
    return df.explode(column_name).reset_index(drop=True).dropna(subset=[column_name])


def select_reviews_of_type(df, review_type):
    if review_type == 'Likes':
        return df[['id', 'likes']].rename(columns = {'likes':REVIEW_COL})
    elif review_type == 'Dislikes':
        return df[['id', 'dislikes']].rename(columns = {'dislikes':REVIEW_COL})
    elif review_type == 'Use-case':
        return df[['id', 'usecase']].rename(columns = {'usecase':REVIEW_COL})
    else:
        raise ValueError('Unexpected review type')


df_cleaned = preprocess_data('./data/g2_reviews.json')
base_df = df_cleaned[[
    'id', 'url', 'product.slug', 'name', 'type', 'helpful', 'score', 'segment', 'role', 'title', 'source.type', 'country', 'region', 'date_submitted', 'date_published', 'industry'
]]

# Set page to wide mode
st.set_page_config(layout="wide")
sb = st.sidebar

## Select a company
company_counts = base_df['product.slug'].value_counts()
companies_with_counts = { f"{company} ({count})": company for company, count in company_counts.items() }
selected_company_label = sb.selectbox('Company', companies_with_counts.keys())
selected_company = companies_with_counts[selected_company_label]

## Select a review type
review_type = sb.radio('Review Type', ['Likes', 'Dislikes', 'Use-case'])    

df_of_type = select_reviews_of_type(df_cleaned, review_type)

# Explode the sentences of that review type
with st.spinner('Parsing review sentences...'):
    xpl_df =  explode_reviews(df_of_type, REVIEW_COL)

# Embed reviews
with st.spinner('Vectorizing Reviews...'):
    embedded_df = embed_reviews(xpl_df, REVIEW_COL)


# Filter to selected company
company_df = base_df[base_df['product.slug'] == selected_company].merge(
    embedded_df, on='id'
)

with st.spinner('Clustering Reviews...'):
    clustered_df = cluster_and_append(company_df, f'{REVIEW_COL}_embeddings', n_components=50)

N = 30

top_cluster_docs = find_closest_to_centroid(
    clustered_df,
    N,
    f'{REVIEW_COL}_embeddings',
    f'{REVIEW_COL}_embeddings_cluster_id',
    REVIEW_COL
)

top_cluster_docs = summarize_sequential(top_cluster_docs)
top_cluster_map = {cluster_id: data["cluster_label"] for cluster_id, data in top_cluster_docs.items()}
clustered_df['cluster_label'] = clustered_df[f'{REVIEW_COL}_embeddings_cluster_id'].map(top_cluster_map)

## Reduce the embedding space to 2D for visualization
reduce_dim_df = reduce_dimensions_append_array(clustered_df, f'{REVIEW_COL}_embeddings', num_dimensions=2, dim_col_name='dims_2d')


#### FILTERS
filtered_df = radio_filter('Source', sb, reduce_dim_df, 'source.type')
filtered_df = radio_filter('Segment', sb, filtered_df, 'segment')
filtered_df = range_filter('Review Date', sb, filtered_df, 'date_published')

### Colour Selector
colour_by_selected = st.radio('Colour by', options=['Cluster', 'Segment', 'Source'], index=0, horizontal=True)
colour_by_col = {'Segment': 'segment', 'Source': 'source.type', 'Cluster': 'cluster_label'}[colour_by_selected]


fig_clusters = visualize_embeddings(
    filtered_df,
    coords_col='dims_2d',
    review_text_column=REVIEW_COL,
    colour_by_column=colour_by_col
)


st.plotly_chart(fig_clusters, use_container_width=True)


fig_publish_dates = plot_over_time(filtered_df, 'date_published')

st.plotly_chart(fig_publish_dates, use_container_width=True)