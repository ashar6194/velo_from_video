import cv2
import glob
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os

from io_args import args


def extract_frames(vid_name, vid_folder):
    vid = cv2.VideoCapture(vid_name)
    flag, image = vid.read()
    frame_id = 0

    while flag:
        out_path = os.path.join(vid_folder, 'frame_%07d.png' % frame_id)
        cv2.imwrite(out_path, image)
        flag, image = vid.read()
        frame_id += 1

    print('Extraction Completed')


def display_cropped_test(vid_folder, label_file_name):
    img_list = sorted(glob.glob('%s/*.png' % vid_folder))

    vel = np.squeeze(pickle.load(open(label_file_name, 'rb')))

    for idx, img_file in enumerate(img_list):
        img = cv2.imread(img_file)
        bbox = cv2.rectangle(img, (150, 250), (500, 360), (255, 0, 0), 2)
        cropped_img = bbox[200:360, 150:500]
        print(idx)
        bbox = cv2.putText(bbox, 'Speed = %f mph' % vel[idx], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0, 255), 2)

        if idx < 5:
            cv2.imwrite('frame_%07d.png' % idx, bbox)

        cv2.imshow('see_crop', bbox)
        cv2.imshow('sep_crop', cropped_img)
        cv2.waitKey(50)


def display_cropped_imgs(vid_folder, label_file_name):
    img_list = sorted(glob.glob('%s/*.png' % vid_folder))
    vel = []
    f = open(label_file_name, 'rb')
    for row in f:
        vel.append(float(row))
    pkl_filename = os.path.join(vid_folder, 'vel_labels.pkl')

    if not os.path.exists(pkl_filename):
        pickle.dump(vel, open(pkl_filename, 'wb'))

    for idx, img_file in enumerate(img_list):
        img = cv2.imread(img_file)
        bbox = cv2.rectangle(img, (150, 250), (500, 360), (255, 0, 0), 2)
        cropped_img = bbox[200:360, 150:500]
        print(cropped_img.shape)
        bbox = cv2.putText(bbox, 'Speed = %f' % vel[idx], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0 ,0, 255), 2)

        cv2.imshow('see_crop', bbox)
        cv2.imshow('sep_crop' , cropped_img)
        cv2.waitKey(5)


def plot_train_speeds(file_name):
    vel = []

    f = open(file_name, 'rb')

    for row in f:
        vel.append(float(row))

    plt.figure(1)
    plt.plot(vel)
    plt.show()


if __name__ == '__main__':
    a = 1

    vid_dir = args.data_root_dir
    train_vid = os.path.join(vid_dir, 'train.mp4')
    test_vid = os.path.join(vid_dir, 'test.mp4')
    velocity_file = os.path.join(vid_dir, 'train.txt')
    test_velocity_file = 'test_results.pkl'

    train_img_folder = os.path.join(vid_dir, 'train_images')
    test_img_folder = os.path.join(vid_dir, 'test_images')

    if not os.path.exists(train_img_folder):
        os.makedirs(train_img_folder)
    if not os.path.exists(test_img_folder):
        os.makedirs(test_img_folder)

    # plot_train_speeds(velocity_file)
    # display_cropped_imgs(test_img_folder, velocity_file)
    display_cropped_test(test_img_folder, test_velocity_file)
    # extract_frames(train_vid, train_img_folder)
    # extract_frames(test_vid, test_img_folder)
