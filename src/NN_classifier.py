import tensorflow as tf
from tensorflow.python.platform import gfile



def classify_image(img_path):
    """Reads an image and returns a list of labels and scores in descending order"""

    # Read image
    image_data = tf.gfile.FastGFile(img_path, 'rb').read()

    # Get labels
    labels = [line.rstrip() for line in tf.gfile.GFile('network/output_labels.txt')]

    # Load the network
    model_filename = 'network/neural_network.pb'
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def, name='')

    # Run classification
    with tf.Session() as sess:
        # Feed the image_data as input to the graph.
        #   predictions  will contain a two-dimensional array, where one
        #   dimension represents the input image count, and the other has
        #   predictions per class
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        predictions, = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})

        # Sort to show labels in order of confidence
        labels_and_preds = []
        for node_id in reversed(predictions.argsort()):
            labels_and_preds.append((labels[node_id], predictions[node_id]))
        return labels_and_preds


if __name__ == '__main__':
    scores = classify_image('/home/ottohant/test.jpg')
    for label, score in scores:
        print('{} (score = {})'.format(label, score))