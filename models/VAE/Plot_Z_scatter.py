#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import pandas as pd
from sklearn.decomposition import PCA 
pca = PCA(n_components=2) 

z = np.load('Z_5m.npz')['Z']
y = np.load('Z_5m.npz')['y']

pca.fit(z)
z_2d = pca.transform(z) 

# df = pd.DataFrame()
# df['z_0'] = z_2d[:,0]
# df['z_1'] = z_2d[:,1]
# df['y']   = np.argmax(y, axis= -1)

df = pd.DataFrame()
df['z_0'] = z[:, 0]
df['z_1'] = z[:, 1]
df['y']   = np.argmax(y, axis= -1)

ax = sns.scatterplot(x="z_0", y="z_1", hue="y", data=df)
plt.savefig("z_2d_scatter.png")
