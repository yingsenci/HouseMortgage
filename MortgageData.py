# Load data
import pandas as pd
mortgage_file = open("房抵贷数据/fangdidai_origin.csv")
mortgages = pd.read_csv(mortgage_file)
# 这两列可以直接推断出label，因此需要去除
mortgages = mortgages.drop(columns=["postLoanOverdueDays", "postLoanRepaymentStatus", "idcard_sha", "orderid"])
nan_threshold = 0.5
mortgages = mortgages.drop(columns=[col for col in mortgages.columns
                                    if mortgages[col].isna().mean() > nan_threshold])
mortgages = mortgages.fillna(0)
mortgages = mortgages.drop(columns=[col for col in mortgages.columns
                                    if mortgages[col].min() == mortgages[col].max()])
# 对样本每一列进行归一化
for col in mortgages.columns:
    if col != "orderTimestamp":
        mortgages[col] = (mortgages[col] - mortgages[col].min())/(mortgages[col].max() - mortgages[col].min())

train_set = mortgages[(mortgages['orderTimestamp'] <= 1525017600) | (mortgages['orderTimestamp'] == 1561359511)]
valid_set = mortgages[(mortgages['orderTimestamp'] > 1525017600) & (mortgages['orderTimestamp'] < 1561359511)]
print(valid_set["label"].sum())
train_set = train_set.drop(columns=['orderTimestamp'])
valid_set = valid_set.drop(columns=['orderTimestamp'])

train_data = train_set.iloc[:, :].values
valid_data = valid_set.iloc[:, :].values

