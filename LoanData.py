import numpy as np
import pandas as pd


mortgages = pd.read_csv("房抵贷数据/fangdidai_origin.csv")
mortgages.columns = mortgages.columns.str.lower()
loan_train = mortgages[(mortgages['ordertimestamp'] <= 1525017600) | (mortgages['ordertimestamp'] == 1561359511)]
loan_test = mortgages[(mortgages['ordertimestamp'] > 1525017600) & (mortgages['ordertimestamp'] < 1561359511)]
'''
loan_train_file = open("xiaofei_data/xiaofei_train.csv")
loan_test_file = open("xiaofei_data/xiaofei_test.csv")
loan_train = pd.read_csv(loan_train_file)
loan_test = pd.read_csv(loan_test_file)
'''
loan_all = pd.concat([loan_train, loan_test])
loan_train[loan_train==-1] = np.nan
nan_threshold = 0.5
nan_cols = [col for col in loan_train.columns if loan_train[col].isna().mean() > nan_threshold]
# 丢弃能够直接推导出标签的列
drop_cols = nan_cols + \
            [
                'idcard_sha',
                'orderid',
                'ordertimestamp',
                'orderplatformtype',
                'postloanoverduedays',
                'postloanrepaymentamount',
                'postloanrepaymentstatus',
                'orderperiods',
                'max_overdue_times', 'phone'
            ] + \
            [col for col in loan_all.columns if col.startswith("xc_")]

# 对样本每一列进行归一处理
for col in loan_all.columns:
    if col not in drop_cols:
        loan_train[col] = (loan_train[col] - loan_all[col].min())/(loan_all[col].max() - loan_all[col].min())
for col in loan_all.columns:
    if col not in drop_cols:
        loan_test[col] = (loan_test[col] - loan_all[col].min())/(loan_all[col].max() - loan_all[col].min())


loan_train = loan_train.drop(columns=drop_cols)
loan_test = loan_test.drop(columns=drop_cols)
train_data = loan_train.iloc[:, :].values
valid_data = loan_test.iloc[:, :].values