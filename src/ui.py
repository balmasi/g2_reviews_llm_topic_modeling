def radio_filter(label, st, df, col):
    options = ["All", *df[col].unique().tolist()]

    choice = st.radio(label, options, index=0)
    if choice == "All":
        return df

    return df[df[col] == choice]


def range_filter(label, st, df, col):
    # Filter out None or NaN values and sort the unique values
    options = sorted(df[col].dropna().unique().tolist())
    min_item = min(options)
    max_item = max(options)

    start, end = st.select_slider(label, options=options, value=(min_item, max_item))

    return df[(df[col] >= start) & (df[col] <= end)]
