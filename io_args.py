import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_size1', type=int, default=66)
parser.add_argument('--input_size2', type=int, default=200)
parser.add_argument('--model_name', type=str, default="basic")
parser.add_argument('--data_root_dir', type=str, default="/home/ashar/Documents/comma_ai/speed_challenge_2017/data/")

args = parser.parse_args()