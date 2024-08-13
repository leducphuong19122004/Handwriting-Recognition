import cv2
import numpy as np

class Preprocess:  
    char_list = [' ', '!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
           '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?',
           'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
           'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
     
    def __init__(self) -> None:
       pass


    def add_padding(self):
        padd_h = int((self.target_h - self.img_h) / 2)
        padd_w = int((self.target_w - self.img_w) / 2)
        h1, h2 = padd_h, int(padd_h + self.img_h)
        w1, w2 = padd_w, int(padd_w + self.img_w)
        img_pad = np.ones([self.target_h, self.target_w, 3]) * 255
        img_pad[h1:h2, w1:w2, :] = self.image
        self.image = img_pad

    def fix_size(self):
        if self.img_h < self.target_h and self.img_w < self.target_w:
            self.add_padding()
        elif self.img_h < self.target_h and self.img_w >= self.target_w:
            # for example:to convert image size from (28, 256) to target size (32, 128), firstly we need resize image to (14, 128) and then add padding to image
            new_w = self.target_w
            self.img_w = new_w
            new_h = int(self.img_h * self.target_w / self.img_w)
            self.img_h = new_h
            self.image = cv2.resize(self.image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            self.add_padding()
        elif self.img_h >= self.target_h and self.img_w < self.target_w:
            new_h = self.target_h
            self.img_h = new_h
            new_w = int(self.img_w * self.target_h / self.img_h)
            self.img_w = new_w
            self.image = cv2.resize(self.image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            self.add_padding()
        else:
            # w >= target_w and h >= target_h
            ratio = max((self.img_w / self.target_w), (self.img_h / self.target_h))
            new_w = max(min(self.target_w, int(self.img_w/ratio)), 1)
            self.img_w = new_w
            new_h = max(min(self.target_h, int(self.img_h / ratio)), 1)
            self.img_h = new_h
            self.image = cv2.resize(self.image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            self.add_padding()
    
    def preprocess(self, image_path, target_size: tuple=(64,128)):
        self.image = cv2.imread(image_path)
        self.img_h, self.img_w = self.image.shape[:2]
        self.target_h = target_size[0]
        self.target_w = target_size[1]
        self.fix_size()
        """ Pre-processing image for predicting """
        self.image = np.clip(self.image, 0, 255) #  is used to Clip (limit) the values in an array.
        self.image = np.uint8(self.image)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image = self.image.astype(np.float32)
        self.image = np.expand_dims(self.image, axis=2)
        self.image /= 255 # Normalize imself.image
        return self.image # images have shape (64, 128, 1)
    
    def encode_gttext_to_label(self, gt_text):
        # "more" -> [65 67 70 57]  
        digits_list = []
        for index, char in enumerate(gt_text):
            try:
                digits_list.append(self.char_list.index(char))
            except:
                print(char)
        return digits_list

    def decode_label_to_gttext(self, label):
        pass

        


