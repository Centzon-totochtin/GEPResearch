import matplotlib.pyplot as plt
import numpy as np
import torch


f1 =torch.load('low_dimension_feature.pt')
color=['green','green','green',
         'red','red','red',
         'blue','blue','blue',
         'orange','orange','orange',
         'gray','gray','gray']
plt.figure(figsize=(5,5))
plt.axes().get_yaxis().set_ticks_position('right')
plt.scatter(f1[:,0],f1[:,1],color = color,linewidths=0.01)
plt.xlim([-1,-0.4])
plt.ylim([-1,-0.5])

plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.show()
