import math
import cv2
import numpy as np
import glob
import os


def change_brightness(image, bright_factor):
    """
    Augments the brightness of the image by multiplying the saturation by a uniform random variable
    Input: image (RGB)
    returns: image with brightness augmentation
    """

    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # perform brightness augmentation only on the second channel
    hsv_image[:, :, 2] = hsv_image[:, :, 2] * bright_factor

    # change back to RGB
    image_rgb = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    return image_rgb


def opticalFlowDense(image_current, image_next):
    """
    Borrowed From -> https://github.com/JonathanCMitchell/speedChallenge

    input: image_current, image_next (RGB images)
    calculates optical flow magnitude and angle and places it into HSV image
    * Set the saturation to the saturation value of image_next
    * Set the hue to the angles returned from computing the flow params
    * set the value to the magnitude returned from computing the flow params
    * Convert from HSV to RGB and return RGB image with same size as original image
    """
    gray_current = cv2.cvtColor(image_current, cv2.COLOR_RGB2GRAY)
    gray_next = cv2.cvtColor(image_next, cv2.COLOR_RGB2GRAY)

    hsv = np.zeros(image_current.shape)
    # set saturation
    hsv[:, :, 1] = cv2.cvtColor(image_next, cv2.COLOR_RGB2HSV)[:, :, 1]

    # Flow Parameters
    # flow_mat = cv2.CV_32FC2
    flow_mat = None
    image_scale = 0.5
    nb_images = 1
    win_size = 15
    nb_iterations = 2
    deg_expansion = 5
    STD = 1.3
    extra = 0

    # obtain dense optical flow paramters
    flow = cv2.calcOpticalFlowFarneback(gray_current, gray_next,
                                        flow_mat,
                                        image_scale,
                                        nb_images,
                                        win_size,
                                        nb_iterations,
                                        deg_expansion,
                                        STD,
                                        0)

    # convert from cartesian to polar
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # hue corresponds to direction
    hsv[:, :, 0] = ang * (180 / np.pi / 2)

    # value corresponds to magnitude
    hsv[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # convert HSV to float32's
    hsv = np.asarray(hsv, dtype=np.float32)
    rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    return rgb_flow


def provide_shuffle_idx(shape_0, ratio=0.75, data_mode='train'):
    """ Shuffle data and labels.
        Input:
          List of indices
        Return:
          Shuffled Train/Test indices based on requiremets
    """
    np.random.seed(0)
    idx = np.random.choice(shape_0, shape_0, replace=False)
    if data_mode == 'train':
        idx_list = idx[0:int(math.floor(ratio * shape_0))]
    else:
        idx_list = idx[int(math.floor(ratio * shape_0)):]

    return idx_list


if __name__ == '__main__':
    vid_dir = '/home/ashar/Documents/comma_ai/speed_challenge_2017/data'
    train_vid = os.path.join(vid_dir, 'train.mp4')
    train_img_folder = os.path.join(vid_dir, 'train_images')
    img1 = cv2.imread(sorted(glob.glob('%s/*.png' % train_img_folder))[0])
    img2 = cv2.imread(sorted(glob.glob('%s/*.png' % train_img_folder))[1])
    # print(img1.shape, img2.shape)

    flow = opticalFlowDense(img1, img2)
    cv2.imshow('flow_image', flow)
    cv2.waitKey(0)

    print(flow.shape)

