import os
from os.path import join, dirname
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), '..', '.env')
load_dotenv(dotenv_path)

if 'OPENAI_KEY' not in os.environ:
    raise ValueError('Please create a .env file (not tracked by git) and add your OpenAI Key like so \nOPENAI_KEY="[Your Key Here]" \n\nYou can find/create a key at https://platform.openai.com/api-keys')

OPENAI_KEY = os.environ['OPENAI_KEY']