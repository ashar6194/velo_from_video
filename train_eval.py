import math
import os
import h5py
import glob
import datetime

from io_args import args
from model_set import build_model_basic, compile_network, build_model_elu
from tdg import DataGenerator
from utils import provide_shuffle_idx
from keras.callbacks import TensorBoard, LearningRateScheduler, ModelCheckpoint
from multiprocessing import cpu_count


def step_decay(epoch):
  initial_lrate = 0.001
  drop = 0.1
  epochs_drop = 2.0
  num_epoch = epoch if epoch < 100 else 100
  lrate = initial_lrate * math.pow(drop, math.floor((1+num_epoch)/epochs_drop))
  return lrate


if __name__ == '__main__':

  ckpt_dir = args.data_root_dir + 'keras_models/'
  logs_dir = args.data_root_dir + 'keras_logs/'

  if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

  if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

  inp_shape = (args.input_size1, args.input_size2, 3)
  train_img_folder = os.path.join(args.data_root_dir, 'train_images')

  img_list = sorted(glob.glob('%s/*.png' % train_img_folder))

  train_idx = provide_shuffle_idx(len(img_list), ratio=0.75, data_mode='train')
  test_idx = provide_shuffle_idx(len(img_list), ratio=0.75, data_mode='test')
  train_list = [img_list[id1] for id1 in train_idx]
  test_list = [img_list[id2] for id2 in test_idx]

  train_dg = DataGenerator(train_list, batch_size=args.batch_size)
  test_dg = DataGenerator(test_list, batch_size=args.batch_size)

  model = build_model_elu(inp_shape)
  compile_network(model)

  filepath = ckpt_dir + 'weights_%03d.h5' % args.num_epochs
  checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
  tensorboard = TensorBoard(log_dir=logs_dir, batch_size=args.batch_size)
  lrate = LearningRateScheduler(step_decay)
  callbacks_list = [tensorboard]  # lrate
  model.fit_generator(generator=train_dg, epochs=args.num_epochs, verbose=1, validation_data=test_dg,
                      use_multiprocessing=True, workers=cpu_count(), callbacks=callbacks_list)

  model_name = ckpt_dir + '%s_ELU_%s.h5' % (args.model_name, datetime.datetime.now().strftime("%m_%d"))
  model.save(model_name)