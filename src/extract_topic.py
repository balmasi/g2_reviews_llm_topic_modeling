import textwrap
import asyncio

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.globals import set_llm_cache
from langchain.cache import SQLiteCache

from src.constants import OPENAI_KEY

set_llm_cache(SQLiteCache(database_path=".langchain.db"))

llm3 = OpenAI(
    temperature=0, 
    openai_api_key=OPENAI_KEY,
    model='gpt-3.5-turbo-instruct',
    cache=True,
    max_tokens=100

)

llm4 = OpenAI(
    temperature=0, 
    openai_api_key=OPENAI_KEY,
    model='gpt-4',
    cache=True,
    max_tokens=100

)


@st.cache_data(show_spinner=False)
def summarize_cluster(texts, _llm):
    # Use a cheaper model for the map part

    summarize_one_prompt = textwrap.dedent(
        '''
        You are an expert summarizer with the ability to summarize a set of documents into a single concise label.
        Provide at most 3 labels that encapsulate the topics all documents have in common. The documents are enclosed in triple backticks (```).
        The label(s) you provide should not be longer than a few words. Do not include anything else.

        DOCUMENTS:
        ```{review_text}```

        LABEL:
        ''')
    summarize_one_prompt_template = PromptTemplate(template=summarize_one_prompt, input_variables=["review_text"])
    summarize_one_chain = LLMChain(
        llm=_llm,
        prompt=summarize_one_prompt_template
    )

    stuffed_reviews_txt = '\n'.join([f'Review {i}: {txt}' for i, txt in enumerate(texts)])
    
    # Assuming summarize_one_chain.run() is an async function
    return summarize_one_chain.run(stuffed_reviews_txt)


@st.cache_resource
async def summarize_parallel(top_n_cluster):
    # Create a mapping from coroutine to cluster_id
    coro_to_cluster_id = {
        summarize_cluster(val['texts'], llm3): cluster_id
        for cluster_id, val in top_n_cluster.items()
        if cluster_id != -1
    }
    coroutines = list(coro_to_cluster_id.keys())

    # Gather results from all coroutines
    results = await asyncio.gather(*coroutines)

    # Pair each result with its corresponding cluster ID
    for coro, result in zip(coroutines, results):
        top_n_cluster[coro_to_cluster_id[coro]]['cluster_label'] = result
    
    top_n_cluster[-1]['cluster_label'] = 'Uncategorized'

    return top_n_cluster

def summarize_sequential(top_n_cluster):
    """
    This function receives a list of documents and summarizes each document sequentially.
    """

    # Creating a progress bar
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    num_clusters = len(top_n_cluster)
    for i, (cluster_id, val) in enumerate(top_n_cluster.items()):
        if cluster_id == -1:
            top_n_cluster[-1]['cluster_label'] = 'Uncategorized'
        else:
            top_n_cluster[cluster_id]['cluster_label'] = summarize_cluster(val['texts'], llm3)
        
        progress = (i + 1) / num_clusters
        progress_bar.progress(progress)
        progress_text.text(f'Processing document {i + 1}/{num_clusters}')

    # Ensure the progress bar is full upon completion
    progress_bar.progress(1.0)

    return top_n_cluster