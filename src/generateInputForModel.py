from loadDataset import DataLoader
import random
from preprocess import Preprocess
import numpy as np

class InputGenerator():
    def __init__(self, data_set, img_w=128, img_h=64, batch_size=64, input_length=30, max_text_length=19):
        self.img_w = img_w
        self.img_h = img_h
        self.batch_size = batch_size
        self.max_text_length = max_text_length
        self.input_length = input_length
        self.data_set = data_set # [[image_path, gt_text], [],....]
        # training_data_set: 42334
        # validation_data_set: 2229
        self.n = len(self.data_set) 
        self.current_index = 0
        self.indexes = list(range(self.n))
        self.images_and_labels = []

    def build_data(self):
        for index, [img_path, gt_text] in enumerate(self.data_set):
            # if index == 2: 
            #     break
            img = Preprocess().preprocess(img_path)
            label = Preprocess().encode_gttext_to_label(gt_text) # not padded
            self.images_and_labels.append([img, label])

    def next_sample(self):
        self.current_index += 1
        if self.current_index >= self.n:
            self.current_index = 0
            random.shuffle(self.indexes) # shuffle data set 
        return self.images_and_labels[self.indexes[self.current_index]]
    
    def next_batch(self):
        while True:
            X_data = np.ones([self.batch_size, self.img_h, self.img_w, 1])
            Y_data = np.zeros([self.batch_size, self.max_text_length])
            input_length = np.ones((self.batch_size, 1)) * self.input_length
            label_length = np.zeros((self.batch_size, 1))
            
            for i in range(self.batch_size):
                img, label = self.next_sample()
                X_data[i] = img
                Y_data[i, :len(label)] = label
                label_length[i] = len(label)
            
            inputs = [X_data, Y_data, input_length, label_length]
            outputs = np.zeros([self.batch_size])
            yield inputs, outputs


    


    
