import umap
import numpy as np
import pandas as pd
import plotly.express as px


def visualize_embeddings(df, embeddings_column, cluster_column, review_text_column, colour_by_column):
    # Extract embeddings, cluster labels, and hover text from DataFrame
    embeddings = np.array(df[embeddings_column].tolist())
    cluster_labels = df[cluster_column].astype(str)
    review_text = df[review_text_column]

    # Apply UMAP
    reducer = umap.UMAP(random_state=42)
    embeddings_2d = reducer.fit_transform(embeddings)

    # Create a new DataFrame for the 2D embeddings
    df_2d = pd.DataFrame(embeddings_2d, columns=['x', 'y'])
    df_2d[cluster_column] = cluster_labels  # Ensure that the cluster labels are treated as categorical data
    df_2d['review_text'] = review_text
    df_2d[colour_by_column] = df[colour_by_column].astype(str)

    # Create the interactive plot
    fig = px.scatter(df_2d, x='x', y='y', color=colour_by_column,
                     hover_data={
                         'x': False,  # hide the x-coordinate
                         'y': False,  # hide the y-coordinate
                         cluster_column: False,  # hide the cluster_column
                         'review_text': True  # display the hover_text
                     },
                     title=f'Reviews coloured by {colour_by_column}'
    )

    fig.update_layout(legend_title_text='Clusters')

    # Customize the layout
    fig.update_traces(marker=dict(size=5,
                                  line=dict(width=2,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers+text'))

    # Hide cluster id -1 by default (noise)
    for trace in fig.data:
        if trace.legendgroup == 'Uncategorized':  # or trace.name == '-1' depending on your data
            trace.visible = 'legendonly'

     # Remove x and y axis labels (set title to an empty string) and grid lines (set showgrid to False)
    fig.update_xaxes(title='', showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(title='', showgrid=False, zeroline=False, showticklabels=False)

    # Show the plot
    return fig