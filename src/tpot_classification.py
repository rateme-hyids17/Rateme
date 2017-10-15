import cv2
from src.cascade_classifcation import find_face, create_folder
import pandas as pd
import os
from tpot import TPOTRegressor
import numpy as np


def build_filters():
    """ returns a list of kernels in several orientations
    """
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 32):
        params = {'ksize':(ksize, ksize), 'sigma':1.0, 'theta':theta, 'lambd':15.0,
                  'gamma':0.02, 'psi':0, 'ktype':cv2.CV_32F}
        kern = cv2.getGaborKernel(**params)
        kern /= 1.5*kern.sum()
        filters.append((kern,params))
    return filters

def process(img, filters):
    """ returns the img filtered by the filter list
    """
    accum = np.zeros_like(img)
    for kern,params in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum


def create_data():
    """Makes a 64x64 grayscale image dataset."""
    create_folder('../data/faces')
    scores = []
    image_paths = []
    df = pd.read_csv('../data/users.csv')
    for index, row in df.iterrows():
        if row.gender == 'M':
            im = cv2.imread(row.image_path)
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

            x, y, w, h = find_face(gray)
            if x is None:
                continue
            roi_gray = gray[y:y + h, x:x + w]
            resized = cv2.resize(roi_gray, (64, 64))
            filename = os.path.split(row.image_path)[1]
            path = os.path.join('../data/faces', filename)
            cv2.imwrite(path, resized)

            image_paths.append(path)
            scores.append(row.score)

    new_df = pd.DataFrame({'image_path': image_paths, 'score': scores})
    new_df.to_csv('../data/faces.csv')


if __name__ == '__main__':
    #create_data()
    use_gabor = True

    df = pd.read_csv('../data/faces.csv')

    imgs = []
    labels = []

    if use_gabor:
        filters = build_filters()

    for index, row in df.iterrows():
        img = cv2.imread(row.image_path, cv2.IMREAD_GRAYSCALE)

        if use_gabor:
            img = process(img, filters)

        imgs.append(np.array(img.flatten()))
        labels.append(row.score)

    imgs = np.array(imgs)
    labels = np.array(labels)

    train_len = int(len(imgs) * 0.8)
    train_data = imgs[:train_len]
    train_labels = labels[:train_len]
    test_data = imgs[train_len:]
    test_labels = labels[train_len:]

    pipeline_optimizer = TPOTRegressor(generations=5, population_size=20, verbosity=2, config_dict='TPOT light')
    pipeline_optimizer.fit(train_data, train_labels)

    #pipeline_optimizer.export('tpot_pipeline.py')

    print('Score: ' + str(pipeline_optimizer.score(test_data, test_labels)))

    rounded_labels = np.round(test_labels)
    preds = pipeline_optimizer.predict(test_data)
    rounded_preds = np.round(preds)
    print('Accuracy: ' + str(np.sum(rounded_labels == rounded_preds) / len(rounded_labels)))

    # Count close ones
    dists = np.abs(test_labels - preds)
    close = np.sum(dists <= 1)
    print('Close {}/{} ({})'.format(close, len(dists), close / len(dists)))

