import textwrap
import asyncio

import streamlit as st

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.globals import set_llm_cache
from langchain.cache import SQLiteCache
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

from src.constants import OPENAI_KEY

set_llm_cache(SQLiteCache(database_path=".langchain.db"))

llm3 = ChatOpenAI(
    temperature=0, 
    openai_api_key=OPENAI_KEY,
    model='gpt-3.5-turbo-1106',
    cache=True,
    max_tokens=500

)

llm4 = ChatOpenAI(
    temperature=0, 
    openai_api_key=OPENAI_KEY,
    model='gpt-4-1106-preview',
    cache=True,
    max_tokens=500

)


@st.cache_data(show_spinner=False)
def summarize_cluster(texts, _llm):
    # Use a cheaper model for the map part

    summarize_one_prompt = textwrap.dedent(
        '''
        You are an expert summarizer with the ability to find patterns in a set of customer reviews and summarize them into a single concise label.
        Provide a single short (3-10 words) label that encapsulate the key points the reviews have in common.
        The label(s) you provide should not be longer than a few words.
        Ensure the label generated is not too vague (e.g. Do not include anything else.

        The reviews are enclosed in triple backticks (```).

        ---
        EXAMPLE 1

        REVIEWS:
        ```
        Review 1: The UI seems to be a little buggy and slow to respond, but it's been getting better
        Review 2: I think they could use more integrations. The user interface also could use some love. It's finicky and and confusing.
        Review 3: The app user experience needs to be improved. It's extremely hard to use.
        ```

        LABEL: UI is hard to use

        ---
        EXAMPLE 2

        REVIEWS:
        ```
        Review 1: The initial price point is pretty high.
        Review 2: Licensing can be a pain in the neck.
        Review 3: Pricing can be lower to favor lower market segments.
        Review 4: The pricing model needs to be simplified.
        ```

        LABEL: Expensive and Confusing Pricing Model
        ---

        REVIEWS:
        ```
        {review_text}
        ```

        LABEL:
        ''')
    prompt = ChatPromptTemplate.from_template(summarize_one_prompt)
    stuffed_reviews_txt = '\n\n'.join([f'Review {i}: {txt}' for i, txt in enumerate(texts)])

    chain = prompt | llm3 | StrOutputParser()
    
    return chain.invoke({ "review_text": stuffed_reviews_txt })


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
        progress_text.text(f'Naming cluster {i + 1}/{num_clusters}')

    # Ensure the progress bar is full upon completion
    progress_bar.empty()
    progress_text.empty()


    return top_n_cluster