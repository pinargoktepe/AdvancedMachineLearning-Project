split_factor = 0.7

import numpy as np
evalpath = 'Eval/list_eval_partition.txt'
labelpath = 'Anno/list_category_img.txt'
evaltxt = np.loadtxt(evalpath, dtype='object')
labels = np.loadtxt(labelpath, dtype='object')
print(labels.shape)

import os
from shutil import copyfile

n = len(labels)
filepaths = np.empty(n, dtype='object')

n_label = np.zeros(50, dtype=np.int)
for i in range(n):
    n_label[int(labels[i,1])] = n_label[int(labels[i,1])] + 1

print(n_label)

m_label = np.zeros(50, dtype=np.int)
for i in range(n):
    if (evaltxt[i,0] != labels[i,0]):
        print('Error! different paths')
    
    m_label[int(labels[i,1])] = m_label[int(labels[i,1])] + 1
    if (m_label[int(labels[i,1])] <= n_label[int(labels[i,1])] * split_factor):
        target = 'dataset1/' + evaltxt[i,1] + '/' + str(i+1)+'.jpg'
    else:
        target = 'dataset2/' + evaltxt[i,1] + '/' + labels[i,1] + '/' + str(i+1)+'.jpg'
    os.makedirs(os.path.dirname(target), exist_ok=True)
    copyfile(evaltxt[i,0], target)
    filepaths[i] = target
    if (i%1000==0):
        print(str(i))
np.savetxt('filepaths.txt', filepaths, fmt='%s')
print('done!')