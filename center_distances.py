'''
    center distance of shuffled matrix
'''
import torch
import numpy as np
import matplotlib.pyplot as plt

from data import graphs
from utils import shuffling,distance,F

shuffle_num=3 # the number of adjacent matrix shuffled form initial adjacent matrix
graph_num=5 # the number of graphs
T=13 #iteration steps
a=0.5
b=3.4
new_matrix=[]
for i in range(graph_num):
    group=[]
    for j in range(shuffle_num):
        shuffled_matrix=shuffling(graphs[i])
        group.append(shuffled_matrix)
    new_matrix.append(group)

center_d=[]
for t in range(T):
    t_step_d=[]
    for i in range(graph_num):
        group=[]
        for j in range(shuffle_num):
            Ft = F(new_matrix[i][j],T=t,a=a,b=b)
            group.append(Ft)
        m = torch.mean(torch.stack(group, dim=0), dim=0)
        for j in range(shuffle_num):
            d = distance(group[j],m)
            t_step_d.append(d)
    center_d.append(t_step_d)

color= ['green','green','green',
         'red','red','red',
         'blue','blue','blue',
         'orange','orange','orange',
         'gray','gray','gray']
center_d = np.log(np.array(center_d).T)

linestyle=['-',':','-.']*5
for i in range(graph_num*shuffle_num):
    plt.plot(np.arange(T),center_d[i],color=color[i],linestyle=linestyle[i%3])
plt.show()







