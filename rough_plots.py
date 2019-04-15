# To Plot and analyze different results obtained by the model

import os
import matplotlib.pyplot as plt
import pickle
import numpy as np
from utils import provide_shuffle_idx
from io_args import args

pkl_filename = args.gt_pkl_filename
gt_labels = np.squeeze(pickle.load(open(pkl_filename, 'rb')))
pred_labels = np.squeeze(pickle.load(open('results.pkl', 'rb')))
test_set_predictions = np.squeeze(pickle.load(open('test_results.pkl', 'rb')))

f = open('test.txt', 'w')
f2 = open('train.txt', 'w')

for elem in test_set_predictions:
    f.write("%7f\n" % elem)
f.close()

new_labels = pred_labels.copy()
mov_avg_idx = 5
for idx in range(new_labels.shape[0]):
    if idx > mov_avg_idx:
        new_labels[idx] = (new_labels[idx] + np.sum(new_labels[idx-mov_avg_idx:idx])) / np.float(mov_avg_idx+1)

for elem in new_labels:
    f2.write("%7f\n" % elem)
f2.close()

test_idx = provide_shuffle_idx(pred_labels.shape[0], ratio=0.75, data_mode='test')

test_gt = gt_labels[test_idx]
test_pred = pred_labels[test_idx]
new_test_pred = new_labels[test_idx]

print(np.mean((gt_labels - pred_labels)**2))
print(np.mean((gt_labels - new_labels)**2))
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