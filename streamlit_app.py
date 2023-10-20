import asyncio
from os import environ
from src.extract_topic import summarize_parallel, summarize_sequential

import streamlit as st
import pandas as pd

from src.text_utils import split_into_sentences
from src.embeddings import embed_reviews, reduce_dimensions_append_x_y
from src.cluster import cluster_and_append, find_closest_to_centroid
from src.visualize import visualize_embeddings, plot_over_time
from src.ui import radio_filter, range_filter


REVIEW_COL = 'review_text'

def extract_answers(df_original):
    df = df_original.copy()
    # Define the regular expressions for each question
    regex_like_best = r"What do you like best about .+?\?\n\n(.*?)(?:\n\n|$)"
    regex_dislike = r"What do you dislike about .+?\?\n\n(.*?)(?:\n\n|$)"
    regex_problems_solving = r"What problems is .+? solving and how is that benefiting you\?\n\n(.*?)(?:\n\n|$)"

    # Use str.extract method to extract the answers and create new columns
    df['like_best'] = df['Review body'].str.extract(regex_like_best, expand=False)
    df['dislike'] = df['Review body'].str.extract(regex_dislike, expand=False)
    df['problems_solving'] = df['Review body'].str.extract(regex_problems_solving, expand=False)
    df['Published date'] = df['Published date'].dt.date.astype(str)
    df['Original date'] = df['Original date'].dt.date.astype(str)

    return df


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
        return df[['ID', 'like_best']].rename(columns = {'like_best':REVIEW_COL})
    elif review_type == 'Dislikes':
        return df[['ID', 'dislike']].rename(columns = {'dislike':REVIEW_COL})
    elif review_type == 'Use-case':
        return df[['ID', 'problems_solving']].rename(columns = {'problems_solving':REVIEW_COL})
    else:
        raise ValueError('Unexpected review type')


df_reviews = pd.read_csv('./data/g2.csv', parse_dates=['Published date', 'Original date'])
df_cleaned = extract_answers(df_reviews)
base_df = df_cleaned[['ID', 'Name', 'Review slug', 'Company name', 'Competitor type',
       'Review rating', 'Review Link', 'Reviewer Type', 'Reviewer Title',
       'Review title', 'Business Partner Review?', 'Validated Reviewer?',
       'Verified Current User?', 'Incentivized Review', 'Review Source',
       'Published date', 'Original date', 'Published date == Original date',
       ]]

# Set page to wide mode
st.set_page_config(layout="wide")
sb = st.sidebar

## Select a company
company_counts = base_df['Company name'].value_counts()
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
company_df = base_df[base_df['Company name'] == selected_company].merge(
    embedded_df, on='ID'
)


with st.spinner('Clustering Reviews...'):
    clustered_df = cluster_and_append(company_df, f'{REVIEW_COL}_embeddings')

# #Define the two columns
# col1, col2 = st.columns(2)

# # Get the input from the user
# rating_min = col1.number_input('Rating Min', min_value=1, max_value=10, value=1)
# rating_max = col2.number_input('Rating Max', min_value=1, max_value=10, value=10)

# # Validate the input
# if rating_min > rating_max:
#     st.error('The minimum rating should not be higher than the maximum rating.')
# elif rating_max < rating_min:
#     st.error('The maximum rating should not be lower than the minimum rating.')
             
# # # Filter the DataFrame based on selected ratings
# filtered_df = clustered_df[
#     (clustered_df['Review rating']>= rating_min) & (clustered_df['Review rating']<= rating_max)
# ]

N = 30

top_cluster_docs = find_closest_to_centroid(
    clustered_df,
    N,
    f'{REVIEW_COL}_embeddings',
    f'{REVIEW_COL}_embeddings_cluster_id',
    REVIEW_COL
)

# filtered_df = clustered_df

top_cluster_docs = summarize_sequential(top_cluster_docs)
top_cluster_map = {cluster_id: data["cluster_label"] for cluster_id, data in top_cluster_docs.items()}
clustered_df['cluster_label'] = clustered_df[f'{REVIEW_COL}_embeddings_cluster_id'].map(top_cluster_map)

## Reduce the embedding space to 2D
reduce_dim_df = reduce_dimensions_append_x_y(clustered_df, f'{REVIEW_COL}_embeddings')


#### FILTERS

# Review Source 
filtered_df = radio_filter('Source', sb, reduce_dim_df, 'Review Source')
filtered_df = radio_filter('Segment', sb, filtered_df, 'Reviewer Type')
filtered_df = range_filter('Review Date', sb, filtered_df, 'Published date')

fig_clusters = visualize_embeddings(
    filtered_df,
    x_col='x', y_col='y',
    cluster_column='cluster_label',
    review_text_column=REVIEW_COL,
    colour_by_column='cluster_label'
)


st.plotly_chart(fig_clusters, use_container_width=True)


fig_publish_dates = plot_over_time(filtered_df, 'Published date')

st.plotly_chart(fig_publish_dates, use_container_width=True)