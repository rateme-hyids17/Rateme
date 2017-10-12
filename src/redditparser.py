import praw
import time
import re
import math
import urllib.request

class RedditParser:
    def __init__(self, client_id, client_secret, password, user_agent, username):
        # Authenticate to API
        self.reddit = praw.Reddit(client_id = client_id,
                             client_secret = client_secret,
                             password = password,
                             user_agent = user_agent,
                             username = username)
        # so lazy to implement date manipulation
        self.year_epoch = 31536000 # use in real system, get 1 year of reddit submissions
        self.day_epoch = 86400 # use for testing purposes

    def store_media(self, submission_media, filename, path):
        if submission_media:
            # TODO: Handle imgur gallery format as well
            # But this requires a scraper package to get first image from the gallery
            if submission_media['oembed']['provider_name'] == 'Imgur':
                url = submission_media['oembed']['thumbnail_url']
                url = url[:-3] # remove this: ?fb
                # TODO: urllib.error.HTTPError: HTTP Error 404: Not Found
                # Throws an exception if there is no link found
                # You can try catch here
                urllib.request.urlretrieve(url, path+filename+'.jpg')
                return True
        else:
            return False

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
                # Regex doesn't catch this: 6....5/10
                # Sometimes people say 'you are 13/10', this is for that
                # print('{}   {}'.format(num, denum))
                if num.startswith('.'):
                    num = num.replace('.','')

                if float(num) < float(denum):
                    comment_count += 1
                    score += float(num)
                    # print(top_c.body)
                    # print('{}   {}'.format(num, denum))
                    # print('------------------------------------------------------------------------------')

        # If there is no comment yet, just give below average
        if comment_count == 0:
            return 0.0
        else:
            return score/comment_count

    def parse_rateme(self):
        subreddit = self.reddit.subreddit('Rateme')
        id = 1
        # TODO: Give optional dates in argument
        now = int(time.time())
        for submission in subreddit.submissions(now - self.year_epoch, now):
            # Get age and gender from the submisson title
            age, gender = self.get_agegender(submission.title)
            if age == '' or gender == '':
                continue
            # Get the attractiveness score from the comments
            score = self.get_score(submission.comments)
            if score == 0.0:
                continue
            print('{} {} {} {}/10'.format(submission.author, age, gender, math.ceil(score)))
            # Store the media into a folder
            # TODO: give absolute path also for main data structure-> image_path
            res = self.store_media(submission.media, str(id), 'C:\\Users\\eozer\\Documents\\GitHub\\Rateme\\data\\')

            if not res:
                continue
                # TODO: check output allow if only everything saved
            # Store the data into a data structure, we will later save it into csv
            # TODO: Create the data structure first
                # | id | sender | image_path | gender | age | score
            id += 1
        print('Total of {} people parsed'.format(id))