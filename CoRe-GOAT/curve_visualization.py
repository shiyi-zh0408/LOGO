import os
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

log_path = 'logs'
file_name = 'LOG_MTL_res_1012_0117.log'
state = '[TEST]'
best_ep = 165
f_path = os.path.join(log_path, file_name)
loss_list = []
rl2_list = []

# show plt of rho
with open(f_path, 'r') as f:
    for line in f:
        if line.startswith(f'{state} EPOCH:') and line.split(',')[1].startswith(' correlation:'):
            loss_list.append(float(line.split(',')[1].split(':')[1].split(' ')[1]))
            rl2_list.append(float(line.split(',')[3].split(':')[1].split(' ')[1]))
    f.close()
s = pd.Series(loss_list, name='rho')
sns.lineplot(data=s)
title = state + 'rho'
plt.title(title)
plt.show()

s = pd.Series(rl2_list, name='rl2')
sns.lineplot(data=s)
title = state + 'RL2'
plt.title(title)
plt.show()

# calculate mean and std
if state == '[TEST]':
    adjacent_five = loss_list[best_ep - 2:best_ep + 3]
    print(adjacent_five)
    print('mean:', np.mean(adjacent_five))
    print('std:', np.std(adjacent_five))

    adjacent_five = rl2_list[best_ep - 2:best_ep + 3]
    print(adjacent_five)
    print('mean:', np.mean(adjacent_five))
    print('std:', np.std(adjacent_five))
