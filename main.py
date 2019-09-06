import pandas as pd
import numpy as np
import tensorflow as tf
import Models.DNN as dnn
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
keras = tf.keras
from LoanData import train_data, valid_data

train_data = train_data[:2000, :]
replica_neg = train_data[train_data[:, 0]==1, :]
replica_neg = np.repeat(replica_neg, 9, axis=0)
train_data = np.vstack((train_data, replica_neg))
train_x = train_data[:, 1:]
train_y = train_data[:, 0]
valid_x = valid_data[:, 1:]
valid_y = valid_data[:, 0]


def auc(y_true, y_pred):
    return roc_auc_score(y_true.reshape([-1]), y_pred.reshape([-1]))


print("Input shape:", train_x.shape)
print("Output shape:", train_y.shape)
hparams = dnn.default_hy_params
hparams["layers"] = [(64, keras.activations.sigmoid)]
hparams["dropouts"] = 0.0
model_config = dnn.default_model_config
model_config["input_shape"] = (train_x.shape[1],)
model_config["output_shape"] = (1,)
model_config["metrics"] = [keras.metrics.binary_accuracy]
dnn_model = dnn.get_model(_hy_params=hparams, _model_config=model_config)

report = open("reports/report.txt", "a")
# report.write("\nLR\npostloanoverduedays\n")
report.write("\nLR\n")


class CheckAUCCallBack(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        pred_train = self.model.predict(train_x)
        pred_y = self.model.predict(valid_x)
#        for i in range(pred_y.shape[0]):
#            print(pred_y[i, 0], valid_y.values[i, 0])
        auc_test = auc(valid_y, pred_y)
        auc_train = auc(train_y, pred_train)
        print("\nAuc score:", auc_test, auc_train)
        report.write("Test:{:.4f} Train:{:.4f}\n".format(auc_test, auc_train))
        # print(self.model.get_weights())


train_x[np.isnan(train_x)] = 0
valid_x[np.isnan(valid_x)] = 0
dnn_model.fit(train_x, train_y,
              batch_size=32,
              epochs=2000,
              verbose=1,
              validation_data=(valid_x, valid_y),
              callbacks=[CheckAUCCallBack()])


'''
xgb_params = {
    'max_depth': 4,
    'eta': 0.3,
    'col_subsample': 0.6,
    'objective': 'binary:logistic',
    'tree_method': 'gpu_hist',  # In the laptop, it should be hist, otherwise gpu will have outOfMemory and quit,
    "metrics": ["auc"]
}
xgb_model = xgb.train(xgb_params, xgb.DMatrix(train_x, train_y), 42)
pred_ys = xgb_model.predict(xgb.DMatrix(valid_x))
print("Xgboost result AUC:", auc(valid_y, pred_ys))
# cv = xgb.cv(xgb_params, xgb.DMatrix(train_x, train_y), 100, metrics=["auc"], nfold=5)
# print(cv)
'''