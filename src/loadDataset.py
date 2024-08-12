class DataLoader:
    max_label_length = 0

    def __init__(self, data_dir='D:\Handwritten Extraction Project\iam_words', data_split: float=0.95) -> None:
        self.paths_and_gts = [] # this list contains image's path and its ground truth text

        bad_samples_reference = ['a01-117-05-02', 'r06-022-03-05']  # this is 2 error images in dataset
        f = open(data_dir + '/words.txt')
        print("Loading dataset ...")
        for line in f:
            # ignore empty and comment line
            line.strip()
            if not line or line[0] == "#":
                continue
            # line sample: a01-000u-00-00 ok 154 408 768 27 51 AT A
            line_split = line.split()
            assert len(line_split) >= 9
            # extract description of line 
            file_name_split = line_split[0].split('-') # [a01, 000u, 00, 00]
            subfolder_1_name = file_name_split[0] # a01
            subfolder_2_name = subfolder_1_name + '-' + file_name_split[1] # a01-000u
            image_file_name = line_split[0] + '.png'
            image_path = data_dir + "\\words" + "\\" + subfolder_1_name + "\\" + subfolder_2_name + "\\" + image_file_name

            if line_split[0] in bad_samples_reference:
                print("Ignore broken image")
                continue
            # in string like "a01-000u-00-00 ok 154 408 768 27 51 AT A", text from 9th split is the ground truth text
            gt_text = ' '.join(line_split[8:])
            # caculate max label length
            if len(gt_text) > self.max_label_length:
                self.max_label_length = len(gt_text)

            self.paths_and_gts.append([image_path, gt_text])
            # split dataset into traning set and validation set (95% and 5%)
            split_index = int(data_split * len(self.paths_and_gts))
            self.train_data_set = self.paths_and_gts[:split_index]
            self.validation_data_set = self.paths_and_gts[split_index:]
        print("Finished loading dataset !")

def get_max_text_length(self):
        return self.max_label_length # 19




    



