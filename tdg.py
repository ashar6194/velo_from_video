import math
import glob
import pickle
import numpy as np
import os
import keras
import cv2
import time
# Change argument source file based on the dataset
from io_args import args
from utils import provide_shuffle_idx, opticalFlowDense

class DataGenerator(keras.utils.Sequence):

    def __init__(self, list_ids, batch_size=10, dim=(10, 16, 2),
                 dim_visual=(10, 16, 565), num_actions=3, shuffle=True, flag_data='base_depth'):

        self.dim = dim
        pkl_filename = os.path.join('/home/ashar/Documents/comma_ai/speed_challenge_2017/data/train_images',
                                    'vel_labels.pkl')
        self.labels = pickle.load(open(pkl_filename, 'rb'))

        self.dim_visual = dim_visual
        self.batch_size = batch_size
        self.list_ids = list_ids
        self.shuffle = shuffle
        self.indexes = []
        self.flag_data = flag_data
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_ids) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = [], []

        # Find list of IDs
        list_IDs_temp = [self.list_ids[k] for k in indexes]

        # Generate data
        if self.flag_data == 'base_depth':
            X, y = self.__data_generation_kron(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation_kron(self, list_ids_temp):
        X = np.empty((self.batch_size, args.input_size1, args.input_size2, 3), dtype=np.float16)
        y = np.empty((self.batch_size, 1), dtype=np.float16)
        # y = [[None]] * self.batch_size

        for idx, id_name in enumerate(list_ids_temp):

          name_id = int(id_name.split('/')[-1].split('.')[0].split('_')[-1])

          if name_id == 20399:
              id_name2 = id_name
          else:
              name_id2 = name_id+1
              parent_dir = '/'.join(id_name.split('/')[:-1])
              id_name2 = os.path.join(parent_dir, 'frame_%07d.png' % name_id2)

          img = cv2.imread(id_name)[200:360, 150:500]
          img2 = cv2.imread(id_name2)[200:360, 150:500]

          img = cv2.resize(img, (args.input_size1, args.input_size2))
          img2 = cv2.resize(img2, (args.input_size1, args.input_size2))

          rgb_flow = opticalFlowDense(img, img2)
          # print(np.min(rgb_flow), np.max(rgb_flow))

          dst = np.zeros(shape=(5, 2))
          rgb_flow = cv2.normalize(rgb_flow, dst, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
          # print(np.min(rgb_flow), np.max(rgb_flow))
          # cv2.imshow('flow_vis', rgb_flow)
          # cv2.waitKey(100)

          # print time.time() - aa

          X[idx, ] = rgb_flow.astype(np.float32)
          label = self.labels[name_id]
          y[idx, ] = label

        return X, y


if __name__ == '__main__':
    vid_dir = '/home/ashar/Documents/comma_ai/speed_challenge_2017/data'
    train_vid = os.path.join(vid_dir, 'train.mp4')
    train_img_folder = os.path.join(vid_dir, 'train_images')
    img_list = sorted(glob.glob('%s/*.png' % train_img_folder))
    #
    # np.random.seed(0)
    #
    # train_ids, test_ids = np.random.

    train_idx = provide_shuffle_idx(len(img_list), ratio=0.75, data_mode='train')
    test_idx = provide_shuffle_idx(len(img_list), ratio=0.75, data_mode='test')
    train_list = [img_list[id1] for id1 in train_idx]
    test_list = [img_list[id2] for id2 in test_idx]

    # print(len(train_list), len(test_list))

    qwe = DataGenerator(img_list, flag_data='base_depth', batch_size=32)
    a, b = qwe.__getitem__(0)
    print(a.shape, np.min(a), np.max(a), b.shape, b)

  # img_list = sorted(glob.glob('%s*/images/depthRender/Cam1/*.png' % args.root_dir))

