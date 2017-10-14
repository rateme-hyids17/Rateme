import configparser
from src.redditparser import *
import pandas as pd

# This is the main script to manage all the other elements

# TODO: Provide command line argument parsing

# Read the config
config = configparser.RawConfigParser(allow_no_value=True)
# Production code
# config.read('reddit_api.cfg')
# Test code
config.read('test_bot.cfg')

# Read the values into the variables
client_id = config.get("redditapi", "client_id")
client_secret = config.get("redditapi", "client_secret")
password = config.get("redditapi", "password")
user_agent = config.get("redditapi", "user_agent")
username = config.get("redditapi", "username")

# Test our reddit parser
reddit = RedditParser(client_id, client_secret, password, user_agent, username)
reddit.parse_rateme(query_level=reddit.query_level('1week'))
