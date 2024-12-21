import os
import pathlib

from dotenv import load_dotenv
load_dotenv()


PROJECT_DIR = pathlib.Path('README.md').resolve().parent
API_KEY = os.getenv('API_KEY')
