"""Takes the images and divides them into folders based on score for neural network"""
import os
import shutil
import pandas as pd
import numpy as np
import cv2


if __name__ == '__main__':
    # Delete old folders and create new
    folder = '../data/nn'
    if os.path.exists(folder):
        shutil.rmtree(folder, ignore_errors=True)
    os.mkdir(folder)

    for i in range(10):
        os.mkdir(os.path.join(folder, str(i+1)))

    # Copy images to right score folders
    df = pd.read_csv('../data/users.csv')
    for index, row in df.iterrows():
        if row.gender == 'M':
            score = int(np.round(row.score))
            im = cv2.imread(row.image_path)
            # gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            fname = os.path.split(row.image_path)[1]
            shutil.copyfile('../' + row.image_path, os.path.join(folder, str(score), fname))
            # cv2.imwrite(os.path.join(folder, str(score), fname), gray)
