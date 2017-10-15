import cv2
import pandas as pd
import os
import shutil


def create_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder, ignore_errors=True)
    os.mkdir(folder)

def find_face(image):
    """Image has to be grayscale. Returns largest face found."""
    face_cascade = cv2.CascadeClassifier('haar_xml/haarcascade_frontalface_default.xml')

    # Detect face
    faces = face_cascade.detectMultiScale(
        image,
        scaleFactor=1.05,
        minNeighbors=8,
        minSize=(55, 55),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) == 0:
        return None, None, None, None

    # Find largest detected face
    largest = faces[0]
    largest_size = largest[2] * largest[3]
    for (x, y, w, h) in faces[1:]:
        size = w * h
        if size > largest_size:
            largest = (x, y, w, h)
            largest_size = size

    x, y, w, h = largest
    return x, y, w, h


if __name__ == '__main__':
    df = pd.read_csv('../data/users.csv')

    # Create cascade classifiers
    feature_cascade = cv2.CascadeClassifier('haar_xml/haarcascade_smile.xml')
    # feature_cascade = cv2.CascadeClassifier('haar_xml/haarcascade_eye_tree_eyeglasses.xml')


    # Create folders for images based on if feature is detected
    create_folder('../data/haar_testing')
    feature_folder = '../data/haar_testing/detected'
    frown_folder = '../data/haar_testing/not_detected'
    create_folder(feature_folder)
    create_folder(frown_folder)

    for index, row in df[:10].iterrows():
        # Read image and convert to grayscale
        im = cv2.imread(row.image_path)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        x, y, w, h = find_face(gray)
        if x is None:
            continue

        # Draw rectangle around face
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Get face area
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = im[y:y + h, x:x + w]

        # Detect feature in face area
        featuers = feature_cascade.detectMultiScale(
            roi_gray,
            scaleFactor= 2.1,
            minNeighbors=22,
            minSize=(25, 25),
            flags=cv2.CASCADE_SCALE_IMAGE
            )

        # Draw smiles
        for (x2, y2, w, h) in featuers:
            im_x = x + x2
            im_y = y + y2
            cv2.rectangle(im, (im_x, im_y), (im_x + w, im_y + h), (255, 0, 0), 1)

        # resized = cv2.resize(im, (512, 512))

        if len(featuers) > 0:
            cv2.imwrite(os.path.join(feature_folder, os.path.split(row.image_path)[1]), im)
        else:
            cv2.imwrite(os.path.join(frown_folder, os.path.split(row.image_path)[1]), im)
