# test.py
# written by Sami Khan 02/12/2020 (ALL RIGHTS RESERVED)
import os
import numpy as np
import tensorflow as tf
import unnormalize
import cv2
from tensorflow import ConfigProto
from tensorflow import InteractiveSession
from random import randint

class MODEL:
    def __init__(self, image_np, segment_graph):
        self.graph = segment_graph
        self.image = image_np


    def predict(self):
        self.config = ConfigProto()
        self.config.gpu_options.allow_growth = True
        with self.graph.as_default():
            with tf.compat.v1.Session(graph = self.graph, config = self.config) as sess:
                # Definite input and output Tensors for segment_graph
                image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                detection_boxes = self.graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                detection_scores = self.graph.get_tensor_by_name('detection_scores:0')
                detection_classes = self.graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.graph.get_tensor_by_name('num_detections:0')
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(self.image, axis=0)
                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes,
                     num_detections], feed_dict={image_tensor: image_np_expanded})
        return [boxes, scores, classes, num]


def get_segments(image_np, segment_graph):
    model = MODEL(image_np, segment_graph)
    boxes,scores,classes,num = model.predict()

    ##Remove single-dimensional entries from the shape of arrays
    s = np.squeeze(scores)
    c = np.squeeze(classes)
    b = np.squeeze(boxes)

    ## selecting only the boxes with confidence scores are threshold value
    good_scores = [score for score in s if score >= 0]
    ## b and c are intrisincally sorted by the scores
    new_boxes = b[:len(good_scores)]
    new_classes = c[:len(good_scores)]

    return new_boxes, good_scores, new_classes


def load_graph(FROZEN_INFERENCE_GRAPH_LOC):
    # load a (frozen) TensorFlow model into memory
    ## https://stackoverflow.com/questions/47059848/difference-between-tensorflows-graph-and-graphdef
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        ## tf.gfile.GFile is tensorflows file I/O wrapper
        with tf.io.gfile.GFile(FROZEN_INFERENCE_GRAPH_LOC, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            ##Imports the graph from graph_def into the current default Graph
            tf.import_graph_def(od_graph_def, name='')

    return detection_graph



def get_regions(ATTRIBUTES, SINGLE_ATTRIBUTES_LIST, DOUBLE_ATTRIBUTES_DIC,
                 image, detection_graph):
    boxes, scores, classes = get_segments(image, detection_graph)
    indicies = {}
    j = 0
    for attributes in ATTRIBUTES.keys():
        if attributes in DOUBLE_ATTRIBUTES_DIC:
            for value in DOUBLE_ATTRIBUTES_DIC.values():
                indicies[attributes] = [i for i, x in enumerate(classes)
                                        if x == j + 1 and scores[i] > value]
        else:
            indicies[attributes] = [i for i, x in enumerate(classes)
                                     if x == j + 1] 
        j += 1   

    height, width = image.shape[:2]
    unnormalize_dic = {}
    for attributes in ATTRIBUTES.keys():
        unnormalize_dic[attributes] = (unnormalize.get_boundaries(indicies[attributes],
                                         height, width, boxes))

    predicted_dic = {}
    for attribute in ATTRIBUTES.keys():
        if attribute in DOUBLE_ATTRIBUTES_DIC:
            predicted_dic[attribute.capitalize()] = (unnormalize_dic[attribute][0:2]
                                                     if len(attribute) else None)
        else:
            predicted_dic[attribute.capitalize()] = ([unnormalize_dic[attribute][0]]
                                                     if len(attribute) else None)

    print('PREDICTED RESULTS: ', predicted_dic)
    return predicted_dic

def draw_boxes(ATTRIBUTES, SINGLE_ATTRIBUTES_LIST, DOUBLE_ATTRIBUTES_DIC,
                 temp_image, file_dirpath, predicted_dic):
    """Draw boundary boxes in the image for the predicted coordinates """
    for attributes, coordinates in predicted_dic.items():
        if attributes not in DOUBLE_ATTRIBUTES_DIC:
            ymax = int(coordinates[0][0])
            xmin = int(coordinates[0][1])
            ymin = int(coordinates[0][2])
            xmax = int(coordinates[0][3])
            color = (randint(0,255), randint(0,255), randint(0,255))
            cv2.rectangle(temp_image, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(temp_image, attributes, (xmin, ymax-randint(8,30)),
                         cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        else:
            for j, box in enumerate(coordinates[0:2]):
                ymax = int(box[0])
                xmin = int(box[1])
                ymin = int(box[2])
                xmax = int(box[3])
                color = (randint(0,255), randint(0,255), randint(0,255))
                cv2.rectangle(temp_image, (xmin, ymin), (xmax, ymax), color, 2)
                if j == 0:
                    cv2.putText(temp_image, attributes + str(j + 1),
                     (xmin-randint(8,30), ymax-randint(8,30)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                else:
                    cv2.putText(temp_image, attributes + str(j + 2),
                     (xmin-randint(8,30), ymax-randint(8,30)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imwrite(file_dirpath,temp_image)
