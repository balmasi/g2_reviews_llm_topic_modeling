import umap
import numpy as np
import pandas as pd
import plotly.express as px



def visualize_embeddings(df, coords_col, review_text_column, colour_by_column):
    # Ensure each entry in coords_col has exactly 2 elements
    if any(len(coords) != 2 for coords in df[coords_col]):
        raise ValueError(f"Each entry in '{coords_col}' must have exactly 2 elements (x and y coordinates)")

    
    # Create a temporary DataFrame for plotting
    temp_df = df.copy()
    temp_df[['x', 'y']] = temp_df[coords_col].to_list()

    # Create the interactive plot
    fig = px.scatter(
        temp_df,
        x='x',
        y='y',
        color=colour_by_column,
        hover_data={
            'x': False,  # hide the x-coordinate
            'y': False,  # hide the y-coordinate
            review_text_column: True  # display the hover_text
        }
    )

    fig.update_layout(
        legend_title_text=None
        # , height=1300
    )

    # Customize the layout
    fig.update_traces(
        marker=dict(size=5, line=dict(width=2, color='DarkSlateGrey')),
        selector=dict(mode='markers+text')
    )

    # Hide cluster id -1 by default (noise)
    for trace in fig.data:
        if trace.legendgroup == 'Uncategorized':  # or trace.name == '-1' depending on your data
            trace.visible = 'legendonly'

     # Remove x and y axis labels (set title to an empty string) and grid lines (set showgrid to False)
    fig.update_xaxes(title='', showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(title='', showgrid=False, zeroline=False, showticklabels=False)

    # Show the plot
    return fig


def plot_over_time(df, date_col):
    daily_counts = (
        pd.to_datetime(df[date_col])
        .dt.floor('D')
        .to_frame()
        .groupby(date_col)
        .size()
        .reset_index(name='count')
    )

    # Create a bar chart with Plotly
    fig = px.bar(daily_counts, x=date_col, y='count')
    fig.update_yaxes(title='', showgrid=False, zeroline=False, showticklabels=False)
    return fig