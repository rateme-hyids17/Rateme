import cv2
from src.cascade_classifcation import find_face, create_folder
import pandas as pd
import os
from tpot import TPOTRegressor
import numpy as np
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


class TpotClassifier():
    def __init__(self, data_path, gabor=False, reduction_method=None):
        """
        :param data_path:
            Path to the folder that has the image folder and users.csv
        :param make_data_set:
            Takes the images from the data folder, finds faces in them and transforms those into 64x64 grayscale images.
            Making the data set takes a long time.
        :param gabor:
            This determines if gabor filter is used in training.
        :param reduction_method:
            Which method to use for dimensionality reduction. Supported types are 'pca' and 'lda'.
        """
        self.regressor = TPOTRegressor(generations=5, population_size=20, verbosity=2, config_dict='TPOT light')
        self.gabor = gabor
        self.data_path = data_path
        self.filters = None
        if self.gabor:
            self.filters = self.build_filters()
        self.reduction_method = reduction_method
        self.pca = PCA(n_components=2048)
        self.lda = LDA()

    @staticmethod
    def create_data(data_path):
        """Makes a 64x64 grayscale image dataset of faces.
        :param data_path:
            Path to the folder that has the image folder and users.csv
        """
        create_folder(os.path.join(data_path, 'faces'))
        scores = []
        image_paths = []
        df = pd.read_csv(os.path.join(data_path, 'users.csv'))
        for index, row in df.iterrows():
            if row.gender == 'M':
                filename = os.path.split(row.image_path)[1]
                im = cv2.imread(os.path.join(data_path, 'images', filename))
                gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

                x, y, w, h = find_face(gray)
                if x is None:
                    continue
                roi_gray = gray[y:y + h, x:x + w]
                resized = cv2.resize(roi_gray, (64, 64))
                path = os.path.join(os.path.join(data_path, 'faces'), filename)
                cv2.imwrite(path, resized)

                image_paths.append(path)
                scores.append(row.score)

        new_df = pd.DataFrame({'image_path': image_paths, 'score': scores})
        new_df.to_csv(os.path.join(data_path, 'faces.csv'))

    def train(self):
        df = pd.read_csv(os.path.join(self.data_path, 'faces.csv'))

        imgs = []
        labels = []

        for index, row in df.iterrows():
            img = cv2.imread(row.image_path, cv2.IMREAD_GRAYSCALE)

            if self.gabor:
                img = self.process(img, self.filters)

            imgs.append(np.array(img.flatten()))
            labels.append(row.score)

        imgs = np.array(imgs)
        labels = np.array(labels)

        imgs, labels = shuffle(imgs, labels)

        train_len = int(len(imgs) * 0.8)
        train_data = imgs[:train_len]
        train_labels = labels[:train_len]
        test_data = imgs[train_len:]
        test_labels = labels[train_len:]

        if self.reduction_method == 'pca':
            self.pca.fit(train_data)
            train_data = self.pca.transform(train_data)
            test_data = self.pca.transform(test_data)
        elif self.reduction_method == 'lda':
            self.lda.fit(train_data, np.round(train_labels))
            train_data = self.lda.transform(train_data)
            test_data = self.lda.transform(test_data)

        self.regressor.fit(train_data, train_labels)

        # Calculate test accuracy
        rounded_labels = np.round(test_labels)
        preds = self.regressor.predict(test_data)
        rounded_preds = np.round(preds)
        accuracy = np.sum(rounded_labels == rounded_preds) / len(rounded_labels)


        # Calculate within 1 accuracy
        dists = np.abs(test_labels - preds)
        close = np.sum(dists <= 1)
        within_1_accuracy = close / len(dists)

        mean_sqrt_error = self.regressor.score(test_data, test_labels)

        return accuracy, within_1_accuracy, mean_sqrt_error

    def predict(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        x, y, w, h = find_face(gray)
        if x is None:
            return None

        if self.gabor:
            gray = self.process(gray, self.filters)

        roi_gray = gray[y:y + h, x:x + w]
        resized = cv2.resize(roi_gray, (64, 64)).flatten()

        if self.reduction_method == 'pca':
            resized = self.pca.transform([resized])[0]
        elif self.reduction_method == 'lda':
            resized = self.lda.transform([resized])[0]

        return self.regressor.predict(np.array([resized]))[0]

    def build_filters(self):
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

    def process(self, img, filters):
        """ returns the img filtered by the filter list
        """
        accum = np.zeros_like(img)
        for kern,params in filters:
            fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
            np.maximum(accum, fimg, accum)
        return accum


if __name__ == '__main__':
    data_path = '../data'

    # Create data and classifier
    if not os.path.exists(os.path.join(data_path, 'faces')):
        TpotClassifier.create_data(data_path)
    classifier = TpotClassifier(data_path, reduction_method='pca')

    # Train the model
    acc, wthn_1_acc, mean_sqrt_error = classifier.train()
    print('Accuracy: ' + str(acc))
    print('Within 1 accuracy: ' + str(wthn_1_acc))
    print('Average distance: ' + str(np.sqrt(mean_sqrt_error)))

    # Predict an image
    image = cv2.imread('/home/ottohant/test.jpg')
    prediction = classifier.predict(image)
    if prediction is None:
        print('Face not found.')
    else:
        print('Prediction: ' + str (prediction))