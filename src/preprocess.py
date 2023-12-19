import pandas as pd
import streamlit as st
import json

from src.text_utils import split_into_sentences


def load_json_data(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


def normalize_data(data):
    return pd.json_normalize(data)


def rename_columns(df):
    column_mapping = {
        "location.country": "country",
        "location.region": "region",
        "location.primary": "location_primary",
        "date.submitted": "date_submitted",
        "date.published": "date_published",
        "date.updated": "date_updated",
    }
    return df.rename(columns=column_mapping)


def parse_dates(df, date_columns):
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], utc=True).dt.date
        # Convert to date, handling NaT values by converting them to None
        df[col] = df[col].apply(lambda x: x if pd.notna(x) else None)
    return df


def extract_answers(df):
    def extract_answer_element(answers, index):
        if isinstance(answers, list) and len(answers) > index:
            return answers[index]
        return None

    df["likes"] = df["answers"].apply(lambda x: extract_answer_element(x, 0))
    df["dislikes"] = df["answers"].apply(lambda x: extract_answer_element(x, 1))
    df["recommendations"] = df["answers"].apply(lambda x: extract_answer_element(x, 2))
    df["usecase"] = df["answers"].apply(lambda x: extract_answer_element(x, 3))

    return df.drop("answers", axis=1)


def drop_rows_with_answers_raw(df):
    # Drop rows where 'answers_raw' is not NaN (i.e., has a value)
    # This is because splitting this raw text is beyond the scope of this project (for now)
    return df[df["answers_raw"].isna()]


@st.cache_data(show_spinner=False)
def explode_reviews(df, column_name):
    """
    A function that explodes the reviews with multiple sentences into multiple rows with 1 sentence each.

    Parameters:
    - df: A pandas DataFrame. The DataFrame containing the reviews.
    - column_name: A string. The name of the column containing the reviews.

    Returns:
    - df: A pandas DataFrame. The DataFrame with exploded sentences.
    """
    df = df.copy()
    # Split reviews into sentences
    df[column_name] = df[column_name].astype(str).apply(split_into_sentences)

    # Explode the DataFrame and reset the index
    return df.explode(column_name).reset_index(drop=True).dropna(subset=[column_name])


def preprocess_data(path_to_file):
    data = load_json_data(path_to_file)
    df = normalize_data(data)
    df = rename_columns(df)
    df = parse_dates(df, ["date_submitted", "date_published", "date_updated"])
    df = extract_answers(df)
    return drop_rows_with_answers_raw(df)
