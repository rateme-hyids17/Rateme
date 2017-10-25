import argparse
import configparser

from mpmath.functions.functions import im
from src.redditparser import *
from src.NN_classifier import *
from src.tpot_classification import *

if __name__ == "__main__":
    # Defaults
    query_level = "1year"
    mode = "tpot"
    image_path = ""
    train_from_scratch = False

    # Add argument parsers
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    # Query level for the redditparser.py
    parser.add_argument("-q", "--query_level",
                        help="The query level for scraping submissions from reddit/r/Rateme. (DEFAULT: -q 1year)\n"
                             "Available values: 1day, 1week, 1month, 3months, 6months, 1year, 3years.\n\n")
    # Application modes
    parser.add_argument("-a", "--application_mode",
                        help="Choose application mode to perform. (DEFAULT: -a scrap)\n"
                             "Available values: scrap, tpot, nn.\n\n")
    # Method selection for the train and predict
    parser.add_argument("-i", "--image",
                        help="Image path to be predicted.\n\n")

    parser.add_argument("-t", "--train_scratch", action='store_true',
                        help="Train tpot from scratch.\n\n")
    # Parse arguments
    args = parser.parse_args()

    # Ensure they are in correct format
    if args.query_level and args.query_level in ["1day", "1week", "1month", "3months", "6months", "1year", "3years"]:
        query_level = args.query_level
    if args.application_mode and args.application_mode in ["scrap", "tpot", "nn"]:
        mode = args.application_mode
    if args.image:
        image_path = args.image
    if args.train_scratch:
        train_from_scratch = True

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
    elif mode == "tpot":
        # Tpot training here
        classifier = TpotClassifier('data/')
        # Check the train from scratch argument
        if train_from_scratch:
            # Train the model
            if not os.path.exists('data/faces'):
                print("Creating face images...")
                TpotClassifier.create_data('data/')
            print("Trying to find best regressor. This process can take long time.")
            acc, wthn_1_acc, mean_sqrt_error = classifier.train()
            print('Accuracy: ' + str(acc))
            print('Within 1 accuracy: ' + str(wthn_1_acc))
            print('Average distance: ' + str(np.sqrt(mean_sqrt_error)))
        else:
            classifier.load()

        # Predict an image
        image = cv2.imread(image_path)
        prediction = classifier.predict(image)
        if prediction is None:
            print('Face not found.')
        else:
            print('Prediction: ' + str(prediction))
    elif mode == "nn":
        try:
            # Predict using NN
            classifier = NeuralClassifier(network_path='src/network/output_labels.txt',
                                          label_path='src/network/neural_network.pb')
            if os.path.exists(image_path):
                image = cv2.imread(image_path)
                label, scores = classifier.predict(image)
                print('Prediction: ' + str(label))
                for label, score in scores:
                    print('{} (score = {})'.format(label, score))
        except:
            raise Exception("Error in neural network classifier")
    else:
        raise Exception("Please select available application modes")
