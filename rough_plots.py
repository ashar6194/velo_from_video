import os
import matplotlib.pyplot as plt
import pickle
import numpy as np
from utils import provide_shuffle_idx

pkl_filename = os.path.join('/home/ashar/Documents/comma_ai/speed_challenge_2017/data/train_images', 'vel_labels.pkl')
gt_labels = np.squeeze(pickle.load(open(pkl_filename, 'rb')))
pred_labels = np.squeeze(pickle.load(open('results.pkl', 'rb')))

test_set_predictions = np.squeeze(pickle.load(open('test_results.pkl', 'rb')))

new_labels = pred_labels.copy()
mov_avg_idx = 5
for idx in range(new_labels.shape[0]):
    if idx > mov_avg_idx:
        new_labels[idx] = (new_labels[idx] + np.sum(new_labels[idx-mov_avg_idx:idx]))/ np.float(mov_avg_idx+1)

test_idx = provide_shuffle_idx(pred_labels.shape[0], ratio=0.75, data_mode='test')

test_gt = gt_labels[test_idx]
test_pred = pred_labels[test_idx]
new_test_pred = new_labels[test_idx]

print(np.mean((test_pred - test_gt)**2))
print(np.mean((new_test_pred - test_gt)**2))

plt.figure(1)
# plt.hold()
plt.grid()
# plt.plot(pred_labels, c='b')
plt.plot(new_labels, c='g')
plt.plot(gt_labels, c='r')


plt.figure(2)
# plt.hold()
plt.grid()
plt.plot(test_set_predictions, c='b')
plt.show()