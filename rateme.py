import argparse
import configparser
from src.redditparser import *


if __name__ == "__main__":
    # Defaults
    query_level = "1year"
    mode = "scrap"
    # TODO: add optional values for other modes: train, predict, live
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-q", "--query_level",
                        help="The query level for scraping submissions from reddit/r/Rateme. (DEFAULT: -q 1year)\n"
                             "Available values: 1day, 1week, 1month, 3months, 6months, 1year, 3years.\n\n")
    parser.add_argument("-m", "--application_mode",
                        help="Choose application mode to perform. (DEFAULT: -m scrap)\n"
                             "Available values: scrap, train, predict, live.\n\n")
    # Parse arguments
    args = parser.parse_args()

    # Ensure they are in correct format
    if args.query_level and args.query_level in ["1day", "1week", "1month", "3months", "6months", "1year", "3years"]:
        query_level = args.query_level
    if args.application_mode and args.application_mode in ["scrap", "train", "predict", "live"]:
        mode = args.application_mode

    # Run according to the application modes
    if mode == "scrap":
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
        # Fail that if it is empty
        if not (client_id and client_secret and password and user_agent and username):
            raise Exception("Please provide a reddit API related fields in reddit_api.cfg")

        # Test our reddit parser
        try:
            reddit = RedditParser(client_id, client_secret, password, user_agent, username)
            reddit.parse_rateme(query_level)
        except:
            raise Exception("Check your reddit_api.cfg. Reddit agent failed")
