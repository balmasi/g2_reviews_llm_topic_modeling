# G2 Review clustering using LLMs

This project is a proof-of-concept demonstraing how you can use LLMs to perform competitive intelligence on customer reviews and feedback.

In this scenario, we're taking G2 reviews and performing topic modelling in a simple streamlit app.

## Getting Started

### Prerequisites

Python (version 3.10)


### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/balmasi/topic_modeling_llm

2. Install the required dependencies:

```
pip install -r requirements.txt
```


### Getting the G2 Company reviews
1. Browse to your target G2 profiles to grab the slug from the url. For example `https://www.g2.com/products/vena/reviews` would be `vena`
2. Place each target company on a line in the `data/slugs-to-fetch.txt` file. 
3. Set the `APIFY_API_TOKEN` in the .env file to your Apify API token
4. run the create_dataset.py command using `python data/create_dataset.py`


### Running the App

To run the app, execute the following command:

```
streamlit run streamlit_app.py
```

This will start the Streamlit server and launch the app in your default web browser.