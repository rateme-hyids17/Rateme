import pandas as pd
import praw
import time
import re
import math
import urllib.request
import os
import shutil


class RedditParser:
    def __init__(self, client_id, client_secret, password, user_agent, username):
        # Authenticate to API
        self.reddit = praw.Reddit(client_id = client_id,
                             client_secret = client_secret,
                             password = password,
                             user_agent = user_agent,
                             username = username)
        # Main data structure
        self.users = []
        # Create directory
        self.setup_imagepath()

    def query_level(self, level):
        """
        Returns the query level - epoch correspondence

        :param level: query level (str) available levels:
            1day, 1week, 1month, 3months, 6months, 1year, 3year
        :return: Epoch time of query level
        """
        # Init epoch
        epoch = 0
        # Get epoch-level mapping
        if level == '1day':
            epoch = 86400 # 1 day epoch
        elif level == '1week':
            epoch = 604800  # 1 week epoch
        elif level == '1month':
            epoch = 2678400  # 1 month epoch
        elif level == '3months':
            epoch = 8035200  # 3 months epoch
        elif level == '6months':
            epoch = 16070400  # 6 months epoch
        elif level == '1year':
            epoch = 31536000  # 1 year epoch
        elif level == '3years':
            epoch = 94608000  # 3 year epoch
        else:
            epoch = 31536000  # (DEFAULT) 1 year epoch
        # Give the epoch correspondance
        return epoch

    def setup_imagepath(self, path='data/images'):
        """
        Creates directory if it does not exist, and removes everything

        :param path: Full path to be created
        :return:
        """
        # Remove everything under the images data/image folder
        if os.path.exists(os.path.abspath(path)):
            shutil.rmtree(os.path.abspath(path))
        # Create directory if it does not exist
        if not os.path.exists(os.path.abspath(path)):
            os.makedirs(os.path.abspath(path))

    def store_media(self, submission, fullpath):
        """
        Stores the preview image under RateMe/data/image without any optimization

        :param submission: Submission object of PRAW
        :param fullpath: Fullpath of image to be saved
        :return: Returns True on successful operation
        """
        # Image optimization later
        # https://cloudinary.com/blog/image_optimization_in_python
        if hasattr(submission, 'preview'):
            submission_preview = submission.preview
        else:
            return False

        if submission_preview:
            media = submission_preview['images'][0]
            url = media['source']['url'] # reddit provide different resize options, pick medium option
            if url:
                try:
                    urllib.request.urlretrieve(url, fullpath)
                    return True
                except:
                    return False
            else:
                return False
        else:
            return False

    def get_aboutuser(self, username):
        """
        Returns the user additional information about user using about.json of the user

        :param username: Redditor username (str)
        :return: Returns tuple of (created, comment_karma, link_karma, verified, is_gold, is_mod, is_employee)
        """
        # Get the about.json of the user
        data = self.reddit.get('user/{}/about.json'.format(username))
        created, comment_karma, link_karma, verified_email, is_gold, is_mod, is_employee = data.created_utc,\
                                                                                     data.comment_karma,\
                                                                                     data.link_karma,\
                                                                                     data.has_verified_email,\
                                                                                     data.is_gold,\
                                                                                     data.is_mod,\
                                                                                     data.is_employee
        return created, comment_karma, link_karma, verified_email, is_gold, is_mod, is_employee

    def get_agegender(self, title):
        """
        Get age and gender from the title if given

        :param title: Submission.title object (str)
        :return: Returns age, gender (str tuple)
        """
        age, gender = '', ''
        # Catch the first 2-digit number
        res = re.findall('(\d{2})', title)
        # Check if there is a match
        if res:
            age = res[0] # get the first occurence
        else:
            # Return immediately
            return '', ''
        # Get gender, check char by char
        for c in title.upper():
            gender = c
            if c == 'M' or c == 'F':
                break
            else:
                gender = ''
        # Return the values
        return age, gender

    def get_score(self, comments):
        """
        Gets the average attractiveness score from submission

        :param comments: Comments from submission
        :return: Returns tuple of Score (float), Number of Comment count (int)
        """
        score = 0.0
        comment_count = 0
        # Print the top level comments
        comments.replace_more(limit=0)
        for top_c in comments:
            # Skip bot message
            if 'Hi there, thank you for your submission! To be rated on /r/Rateme,' in top_c.body:
                continue
            # Match the rating
            res = re.findall('(\d*\.*\d+)(/10)', top_c.body)
            if res:
                match = res[0] # get the first one, I dont care the rest
                num, denum = res[0]
                denum = denum[1:] # Get rid of slash: /10 -> 10
                # This is just a hack for case .....5/10 -> .
                if ".." in num:
                    num = num.replace('.','')
                    # Regex doesn't catch this: 6....5/10
                    # This is also another hack, get the significand: 6, in this case
                    if len(num) > 1:
                        num = num[0]

                # print('{}   {}'.format(num, denum))
                # Sometimes people say 'you are 13/10', this is for that
                if float(num) < float(denum):
                    comment_count += 1
                    score += float(num)
                    # print(top_c.body)
                    # print('{}   {}'.format(num, denum))
                    # print('------------------------------------------------------------------------------')

        # If there is no comment yet, just give below average
        if comment_count == 0:
            return 0.0, 0
        else:
            return score/comment_count, comment_count

    def parse_rateme(self, query_level):
        """
        Main runner method for reddit parsing

        :return: Returns number of the people has been parsed (int)
        """
        subreddit = self.reddit.subreddit('Rateme')
        id = 1
        # TODO: Give optional dates in argument
        now = int(time.time())
        for submission in subreddit.submissions(now - query_level, now):
            # Get age and gender from the submisson title
            age, gender = self.get_agegender(submission.title)
            if age == '' or gender == '':
                continue
            # Get the attractiveness score from the comments
            score, comment_count = self.get_score(submission.comments)
            if score == 0.0:
                continue
            # This could have some bugs, I havent checked thoroughly
            try:
                created, comment_karma, link_karma,\
                verified_email, is_gold, is_mod, is_employee = self.get_aboutuser(submission.author)
            except:
                continue
            # Store the media into a folder
            fullpath = os.path.join(os.path.abspath('data/images'), str(id) + '.jpg')
            res = self.store_media(submission, fullpath)
            if not res:
                continue
            # Show some result
            print('{} {} {} {}/10 comment_karma:{}, link_karma:{}'.format(submission.author,
                                                                          age,
                                                                          gender,
                                                                          math.ceil(score),
                                                                          comment_karma,
                                                                          link_karma))
            # Create a dictionary for user data
            user = {'id': id, 'name': submission.author, 'age': age, 'comment_count': comment_count,
                    'score': format(score, '.3f'), 'gender': gender, 'created_epoch': created,
                    'comment_karma': comment_karma, 'link_karma': link_karma, 'is_gold': is_gold,
                    'is_mod': is_mod, 'is_employee': is_employee, 'verified_email': verified_email,
                    'image_path' : fullpath}
            # Save it into the list
            self.users.append(user)
            id += 1
        # Convert into pandas dataframe to save in csv format
        df = pd.DataFrame.from_dict(self.users)
        df.to_csv('data/users.csv',index=False,
                  columns=['id', 'name', 'age', 'gender', 'score', 'comment_count',
                           'created_epoch', 'comment_karma', 'link_karma', 'verified_email',
                           'is_gold', 'is_mod', 'is_employee', 'image_path'])
        print('Total of {} people parsed'.format(id - 1))
        # Return number of people parsed
        return id - 1