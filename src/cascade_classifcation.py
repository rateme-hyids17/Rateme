import cv2
import pandas as pd
import os


if __name__ == '__main__':
    df = pd.read_csv('../data/users.csv')
    smile_cascade = cv2.CascadeClassifier('haar_xml/haarcascade_smile.xml')
    for index, row in df[:1].iterrows():
        im = cv2.imread(row.image_path)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        smiles = smile_cascade.detectMultiScale(
            gray,
            scaleFactor= 1.7,
            minNeighbors=22,
            minSize=(25, 25),
            flags=cv2.CASCADE_SCALE_IMAGE
            )
        for (x, y, w, h) in smiles:
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = im[y:y + h, x:x + w]
        resized = cv2.resize(im, (512, 512))
        cv2.imwrite(os.path.split(row.image_path)[1], im)
        print(len(smiles))
