import os
from apify_client import ApifyClient
from os import environ
import json

if 'APIFY_API_TOKEN' not in environ:
    raise KeyError('Please set the APIFY_API_TOKEN environment variable to your Apify API token before continuing. Find/create an API key at https://console.apify.com/account/integrations')

# Initialize the ApifyClient with your API token
client = ApifyClient(environ['APIFY_API_TOKEN'])

# Function to read the entire content of the slugs file
def read_slugs(file_path):
    with open(file_path, 'r') as file:
        return file.read().strip()

# Determine the directory where the script is located
script_dir = os.path.dirname(os.path.realpath(__file__))

# Construct paths relative to the script directory
slugs_file_path = os.path.join(script_dir, 'slugs-to-fetch.txt')
json_file_path = os.path.join(script_dir, 'g2_reviews.json')

# Read slugs from the file
slugs_query = read_slugs(slugs_file_path)

# Prepare the Actor input with the slugs query
run_input = {
    "query": slugs_query,
    "mode": "review",
    "limit": 5000,
}

print(f"Fetching reviews for G2 products with the following slugs:\n\n{slugs_query}")
print('\nPlease wait while the extractor runs...')
# Run the Actor and wait for it to finish
run = client.actor("4KGEKf1EUjmYW76dd").call(run_input=run_input)

# Fetch Actor results from the run's dataset and save to a JSON file
with open(json_file_path, 'w', encoding='utf-8') as file:
    reviews = [item for item in client.dataset(run["defaultDatasetId"]).iterate_items()]
    json.dump(reviews, file, indent=4)

print(f"Finished fetching and saving reviews to {json_file_path}")