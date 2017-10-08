import praw


class RedditParser:
    def __init__(self, client_id, client_secret, password, user_agent, username):
        # Authenticate to API
        self.reddit = praw.Reddit(client_id = client_id,
                             client_secret = client_secret,
                             password = password,
                             user_agent = user_agent,
                             username = username)

    def test(self):
        """
        Prints the reddit username
        :return:
        """
        print("Hello I am {}".format(self.reddit.user.me()))

    def test2(self):
        """
        This is a test for PRAW module.
        :return: Prints top level comments of a specific thread
        """
        submission = self.reddit.submission(url='https://www.reddit.com/r/funny/comments/3g1jfi/buttons/')
        submission = self.reddit.submission(id='3g1jfi')
        # Print the top level comments
        submission.comments.replace_more(limit=0)
        for top_level_comment in submission.comments:
            print(top_level_comment.body)