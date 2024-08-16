from loadDataset import DataLoader
from pathlib import Path
from preprocess import Preprocess
from model import CRNN_CTCModel, CNN_and_RNN
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import keras.api.ops as ops
import tensorflow as tf
import keras.api.backend as backend


def ctc_decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1):
    """DEPRECATED."""
    input_shape = tf.shape(y_pred)
    num_samples, num_steps = input_shape[0], input_shape[1]
    y_pred = tf.math.log(
        tf.transpose(y_pred, perm=[1, 0, 2]) + backend.epsilon()
    )
    input_length = tf.cast(input_length, tf.int32)

    if greedy:
        (decoded, log_prob) = tf.nn.ctc_greedy_decoder(
            inputs=y_pred, sequence_length=input_length
        )
    else:
        (decoded, log_prob) = tf.compat.v1.nn.ctc_beam_search_decoder(
            inputs=y_pred,
            sequence_length=input_length,
            beam_width=beam_width,
            top_paths=top_paths,
        )
    decoded_dense = []
    for st in decoded:
        st = tf.SparseTensor(st.indices, st.values, (num_samples, num_steps))
        decoded_dense.append(tf.sparse.to_dense(sp_input=st, default_value=-1))
    return (decoded_dense, log_prob)

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
    CRNN_CTCModel().train() # master

    # We can not use our training model because it also requires labels as input and at test time
    # we can not have labels. So to test the model we will use ” act_model ” that we have created earlier which takes only one input: test images.
    act_model = CNN_and_RNN().act_model
    # act_model.summary()
    act_model.load_weights('CRNN_model.weights.h5')

    # # load and preprocess test images
    dir_path = "D:\Handwritten Extraction Project\\test_data"
    test_img = []
    gt_text = []
    for img_filename in os.listdir(dir_path):
        # create list of image
        img_path = dir_path + "\\" + img_filename
        img = Preprocess().preprocess(img_path)
        # print("==========")
        # print(img)
        test_img.append(img)
        # create list of ground truth text
        gt_text.append(img_filename.split('.')[0])
    test_img = np.array(test_img)
    # prediction outputs on test image
    prediction = act_model.predict(test_img) 
    # use CTC decoder
    out = get_value(ctc_decode(prediction, input_length=np.ones(prediction.shape[0]) * prediction.shape[1], greedy=True)[0][0])
    # see the results
    text_prediction = []
    i = 0
    for x in out:
        char = []
        for p in x:  
            if int(p) != -1:
                char.append(Preprocess().char_list[int(p)]) 
        text = " ".join(char)  
        text_prediction.append(text)  
        i+=1
    # show image and text prediction
    train_data_fig, ax = plt.subplots(2, 4, figsize=(15, 10))
    train_data_fig.suptitle('Training data', weight='bold', size=18)
    for i, img in enumerate(test_img):
        text = text_prediction[i]
        ax[i // 4, i % 4].imshow(img[:, :, 0], cmap="gray")
        ax[i // 4, i % 4].set_title(text)
        ax[i // 4, i % 4].axis("off")
    plt.show()

if __name__ == "__main__":
    main()