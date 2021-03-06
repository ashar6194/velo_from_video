import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_size1', type=int, default=66)
parser.add_argument('--input_size2', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--model_name', type=str, default="basic")
parser.add_argument('--data_root_dir', type=str, default="/home/ashar/Documents/comma_ai/speed_challenge_2017/data/")
parser.add_argument('--gt_pkl_filename', type=str, default='/home/ashar/Documents/comma_ai/speed_challenge_2017/data/train_images/vel_labels.pkl')

args = parser.parse_args()