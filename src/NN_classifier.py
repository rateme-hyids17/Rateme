import tensorflow as tf
from tensorflow.python.platform import gfile
import cv2

class NeuralClassifier():
    def __init__(self, network_path, label_path):
        """
        :param network_path:
            Path to the saved neural network.
        :param label_path:
            Path to the label text file.
        """
        # Get labels
        self.labels = [line.rstrip() for line in tf.gfile.GFile(network_path)]

        # Load the network
        model_filename = label_path
        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            g_in = tf.import_graph_def(graph_def, name='')
            self.sess = tf.Session()


    def predict(self, image):
        """Reads an image and returns a list of labels and scores in descending order"""

        # Run classification
        with tf.Session() as sess:
            # Feed the image_data as input to the graph.
            #   predictions  will contain a two-dimensional array, where one
            #   dimension represents the input image count, and the other has
            #   predictions per class
            softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
            predictions, = sess.run(softmax_tensor, {'DecodeJpeg:0': image})

            # Sort to show labels in order of confidence
            labels_and_preds = []
            for node_id in reversed(predictions.argsort()):
                labels_and_preds.append((self.labels[node_id], predictions[node_id]))
            return labels_and_preds[0][0], labels_and_preds


if __name__ == '__main__':
    classifier = NeuralClassifier(network_path='network/output_labels.txt', label_path='network/neural_network.pb')
    image = cv2.imread('/home/osboxes/Rateme/data/images/1.jpg')
    label, scores = classifier.predict(image)
    print('Prediction: ' + str(label))
    for label, score in scores:
        print('{} (score = {})'.format(label, score))