# Handwriting recognition 
> [!WARNING]  
> This project is meeting a overfit problem
This is a project that was developed to recognize handwriting by using deep learning. In detail, I trained a CRNN model using "iam handwriting word" database. 
## CRNN Model
For more details you can visit this [paper](https://arxiv.org/pdf/1507.05717)
| Type                   | Configurations                            |
|------------------------|-------------------------------------------|
| Transcription          | -                                         |
| Bidirectional-LSTM      | #hidden units: 256                        |
| Bidirectional-LSTM      | #hidden units: 256                        |
| Map-to-Sequence        | -                                         |
| Convolution            | #maps: 512, k: 2 × 2, s: 1, p: 0          |
| MaxPooling             | Window: 1 × 2, s: 2                       |
| BatchNormalization     | -                                         |
| Convolution            | #maps: 512, k: 3 × 3, s: 1, p: 1          |
| BatchNormalization     | -                                         |
| Convolution            | #maps: 512, k: 3 × 3, s: 1, p: 1          |
| MaxPooling             | Window: 1 × 2, s: 2                       |
| Convolution            | #maps: 256, k: 3 × 3, s: 1, p: 1          |
| Convolution            | #maps: 256, k: 3 × 3, s: 1, p: 1          |
| MaxPooling             | Window: 2 × 2, s: 2                       |
| Convolution            | #maps: 128, k: 3 × 3, s: 1, p: 1          |
| MaxPooling             | Window: 2 × 2, s: 2                       |
| Convolution            | #maps: 64, k: 3 × 3, s: 1, p: 1           |
| Input                  | 128 × 64 gray-scale image                   |
## Dataset
You can download IAM dataset from this [link](https://www.kaggle.com/datasets/nibinv23/iam-handwriting-word-database)
## Steps
This project can be divided into 4 steps:
1. Load image dataset from iam_words/words folder and coresponding lable from words.txt
2. Preprocess image (padding, resize, cvt color,...) and label (padding)
3. Convert preprocessed image and label to model's input format
4. Build model and train it with given inputs
5. Load test image from "test_data" folder and make prediction for those image and show result