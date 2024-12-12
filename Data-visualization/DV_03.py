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
data = pd.read_csv('anage_data.csv')
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

#3c
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
 
%matplotlib inline
image_folder = 'Datasets'#path
img_filenames = sorted(os.listdir(image_folder))
img_filenames = [f for f in img_filenames if f.endswith(('.png', '.jpg', '.jpeg'))]
imgs = [mpimg.imread(os.path.join(image_folder, img_filename)) for img_filename in img_filenames]
fig, axes = plt.subplots(2, 2)
fig.set_size_inches(6, 6)
fig.dpi = 150
axes = axes.ravel()
labels = ['coast', 'beach', 'building', 'city at night']
for i in range(len(imgs)):
   axes[i].imshow(imgs[i])
   axes[i].set_xticks([]) 
   axes[i].set_yticks([])  
   axes[i].set_xlabel(labels[i]) 
plt.tight_layout()
plt.show()


#3d
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Correcting variable names and generating random numbers
ga = np.random.randint(85, 100, 60)
gb = np.random.randint(85, 100, 60)
gc = np.random.randint(85, 100, 60)
gd = np.random.randint(85, 100, 60)

# Creating the plot
plt.figure(figsize=(10,5), dpi=300)

# Plotting the scatter plot for ga vs gb
plt.subplot(2, 2, 1)
plt.scatter(ga, gb, color="red")
plt.xlabel("group a")
plt.ylabel("group b")
plt.title("ga vs gb")

# Display the plot
plt.tight_layout()
plt.show()



