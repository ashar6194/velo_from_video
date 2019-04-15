import os
import numpy as np
import glob
import cv2
import pickle

from keras.models import load_model
from tqdm import tqdm
from io_args import args
from model_set import compile_network, build_ddp_vgg, build_model_elu
from utils import opticalFlowDense


def eval_results(args, directory):

  im_list = sorted(glob.glob('%s/*.png' % directory))
  pkl_filename = os.path.join('/home/ashar/Documents/comma_ai/speed_challenge_2017/data/train_images', 'vel_labels.pkl')
  gt_labels = pickle.load(open(pkl_filename, 'rb'))

  full_pred_list = np.array([])
  full_gt_list = gt_labels
  mov_avg_idx = 3

  for idx, id_name in tqdm(enumerate(im_list)):

      name_id = int(id_name.split('/')[-1].split('.')[0].split('_')[-1])
      # 20399
      if name_id == 10797:
          id_name2 = id_name
      else:
          name_id2 = name_id + 1
          parent_dir = '/'.join(id_name.split('/')[:-1])
          id_name2 = os.path.join(parent_dir, 'frame_%07d.png' % name_id2)

      img = cv2.imread(id_name)[200:360, 150:500]
      img2 = cv2.imread(id_name2)[200:360, 150:500]

      img = cv2.resize(img, (args.input_size2, args.input_size1))
      img2 = cv2.resize(img2, (args.input_size2, args.input_size1))

      rgb_flow = opticalFlowDense(img, img2)

      dst = np.zeros(shape=(5, 2))
      rgb_flow = cv2.normalize(rgb_flow, dst, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
      rgb_flow -= 0.5
      rgb_flow *= 2

      X = np.expand_dims(rgb_flow.astype(np.float32), axis=0)
      final_pred = np.squeeze(model.predict(X))

      if idx > mov_avg_idx:
          final_pred = (final_pred + np.sum(full_pred_list[idx - mov_avg_idx:idx])) / np.float(mov_avg_idx + 1)

      full_pred_list = np.append(full_pred_list, final_pred)

  # print (np.mean(error))
  # print (np.median(error))
  pickle.dump(full_pred_list, open('test_results_avg_3.pkl', 'wb'))



if __name__ == '__main__':

  ckpt_dir = args.data_root_dir + 'keras_models/'
  train_img_folder = os.path.join(args.data_root_dir, 'train_images')
  test_img_folder = os.path.join(args.data_root_dir, 'test_images')

  model_name = ckpt_dir + 'basic_ELU_04_14.h5'
  inp_shape = (args.input_size1, args.input_size2, 3)
  model = build_model_elu(inp_shape)
  model.load_weights(model_name)
  # infer_outputs(args, ddp_model)
  eval_results(args, test_img_folder)