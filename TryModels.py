from MortgageData import train_data, valid_data
from Models.GAN import get_new_acgan
import numpy as np
from sklearn.metrics import roc_auc_score


import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
# train_data = train_data[:2000, :]
train_data[np.isnan(train_data)] = 0
valid_data[np.isnan(valid_data)] = 0

max_aucs = []
for i in range(500):
    steps = 2000
    batch_size = 32
    cur_index = 0
    np.random.shuffle(train_data)
    aucs = []
    acgan = get_new_acgan()
    for i in range(steps):
        if cur_index + batch_size >= train_data.shape[0]:
            np.random.shuffle(train_data)
            cur_index = 0
        losses = acgan.train_on_batch(train_data[cur_index:cur_index + batch_size, 1:],
                                      train_data[cur_index:cur_index + batch_size, :1])
        if i % 30 == 1:
            # print("Round:", i)
            # print(losses)
            y_pred = acgan.pred_real(valid_data[:, 1:])[:, 0]
            auc_score = roc_auc_score(valid_data[:, 0], y_pred)
            # print("AUC test:", auc_score)
            aucs.append(auc_score)
    aucs = np.array(aucs)
    print(aucs.max())
    max_aucs.append(aucs.max())


import matplotlib.pyplot as plt

plt.hist(max_aucs, bin=50)
plt.show()