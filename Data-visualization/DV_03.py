#3a

import numpy as np
import matplotlib.pyplot as plt
data = [np.random.randint(50,141) for _ in range(100)]
plt.figure(figsize=(6, 4),dpi=150)
plt.hist(data, bins=10)
plt.axvline(x=100,color='r')
plt.axvline(x=115,color='r',linestyle='--')
plt.axvline(x=85,color='r',linestyle='--')
plt.xlabel('IQ')
plt.ylabel('Freq')
plt.title('IQ for a test')
plt.show()
plt.figure(figsize=(6,4),dpi=150)
plt.boxplot(data)
ax = plt.gca()
ax.set_xticklabels(['Test group'])
plt.ylabel('IQ score')
plt.title('IQ for a test')
plt.show()
ga=[np.random.randint(50,141) for _ in range(100)]
gb=[np.random.randint(50,141) for _ in range(100)]
gc=[np.random.randint(50,141) for _ in range(100)]
gd=[np.random.randint(50,141) for _ in range(100)]
plt.figure(figsize=(6,4),dpi=150)
plt.boxplot([ga,gb,gc,gd])
ax = plt.gca()
ax.set_xticklabels(['GA','GB','GC','GD'])
plt.ylabel('IQ score')
plt.title('IQ for dif test')
plt.show()


#3b
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
data = pd.read_csv('D://DV-Dataset//anage_data.csv')
longe = 'Maximum longevity (yrs)'
mass = 'Body mass (g)'
data = data[np.isfinite(data[longe]) & np.isfinite(data[mass])]
aves = data[data['Class'] == 'Aves']
aves = aves[aves[mass] < 20000]
fig = plt.figure(figsize=(8, 8), dpi=150, constrained_layout=True)
gs = GridSpec(4, 4, figure=fig)
histx_ax = fig.add_subplot(gs[0, :-1])
histy_ax = fig.add_subplot(gs[1:, -1])
scatter_ax = fig.add_subplot(gs[1:, :-1])
scatter_ax.scatter(aves[mass], aves[longe])
histx_ax.hist(aves[mass], bins=20, density=True)
histx_ax.set_xticks([])
histy_ax.hist(aves[longe], bins=20, density=True, orientation='horizontal')
histy_ax.set_yticks([])
plt.xlabel('Body mass in grams')
plt.ylabel('Maximum longevity in years')
fig.suptitle('Scatter plot with marginal histograms')
plt.show()
