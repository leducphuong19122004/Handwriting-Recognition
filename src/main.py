from loadDataset import DataLoader
from pathlib import Path
from preprocess import Preprocess
from model import CRNN_CTCModel, CNN_and_RNN
import matplotlib as plt
import cv2
import numpy as np
import os
import keras.api.ops as ops
import tensorflow as tf


def get_value(x):
    if not tf.is_tensor(x):
        return x
    if tf.executing_eagerly() or isinstance(x, tf.__internal__.EagerTensor):
        return x.numpy()
    if not getattr(x, "_in_graph_mode", True):
        # This is a variable which was created in an eager context, but is being
        # evaluated from a Graph.
        with tf.__internal__.eager_context.eager_mode():
            return x.numpy()
    with tf.init_scope():
        return x.numpy()

def main():
    # CRNN_CTCModel().train() 
    # We can not use our training model because it also requires labels as input and at test time
    # we can not have labels. So to test the model we will use ” act_model ” that we have created earlier which takes only one input: test images.
    act_model = CNN_and_RNN().act_model
    act_model.load_weights('RCNN_model.keras')

    # load and preprocess test images
    dir_path = "D:\Handwritten Extraction Project\\test_data"
    test_img = []
    gt_text = []
    for img_filename in os.listdir(dir_path):
        # create list of image
        img_path = dir_path + "\\" + img_filename
        img = Preprocess().preprocess(img_path)
        test_img.append(img)
        # create list of ground truth text
        gt_text.append(img_filename.split('.')[0])
    test_img = np.array(test_img)
    # prediction outputs on test image
    prediction = act_model.predict(test_img)
    print(prediction.shape)
    # use CTC decoder
    out = get_value(ops.ctc_decode(prediction, sequence_lengths=np.ones(prediction.shape[0]) * prediction.shape[1], strategy='greedy')[0][0])
    # see the results
    i = 0
    for x in out:
        print("original_text =  ", gt_text[i])
        print(x)
        print("predicted text = ", end = '')
        for p in x:  
            if int(p) != -1 and int(p) < 79:
                print(Preprocess().char_list[int(p)], end = '')       
        print('\n')
        i+=1


if __name__ == "__main__":
    main()