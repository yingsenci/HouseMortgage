import matplotlib.pyplot as plt

auc_score = []
with open('gan_res.txt', 'r') as f:
    while True:
        auc = f.readline()
        if not auc:
            break
        auc_score.append(float(auc.strip('\n')))
print(auc_score)
plt.hist(auc_score, bins=40)
plt.show()


