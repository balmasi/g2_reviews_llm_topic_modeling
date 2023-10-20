import os
from os.path import join, dirname
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), '..', '.env')
load_dotenv(dotenv_path)

NOT_SET = 'SECRET NOT SET!'

OPENAI_KEY = os.environ.get('OPENAI_KEY', NOT_SET)