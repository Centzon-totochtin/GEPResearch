'''
    center distance of factor alpha
'''

import torch
import numpy as np
import matplotlib.pyplot as plt

from data import graphs
from utils import F,FA,shuffling,distance

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

factor_alpha_center_d=[]

for t in range(T):
    t_step_d=[]
    for i in range(graph_num):
        group=[]
        for j in range(shuffle_num):
            Ft = F(new_matrix[i][j],T=t,a=a,b=b)
            Fa = FA(Ft)
            group.append(Fa)
        m = torch.mean(torch.stack(group, dim=0), dim=0)
        for j in range(shuffle_num):
            d = distance(group[j],m)
            t_step_d.append(d)
    factor_alpha_center_d.append(t_step_d)


color= ['green','green','green',
         'red','red','red',
         'blue','blue','blue',
         'orange','orange','orange',
         'gray','gray','gray']
center_d = np.array(factor_alpha_center_d).T

linestyle=['-',':','-.']*5
for i in range(graph_num*shuffle_num):
    plt.plot(np.arange(T),center_d[i],color=color[i],linestyle=linestyle[i%3])

plt.xlabel('Iteration steps')
plt.ylabel('Center distance of factor alpha')
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.xlim([0,12])
plt.xticks(np.arange(13))
plt.legend(['Graph A(1)','Graph A(2)','Graph A(3)',
            'Graph B(1)','Graph B(2)','Graph B(3)',
            'Graph C(1)','Graph C(2)','Graph C(3)',
            'Graph D(1)','Graph D(2)','Graph D(3)',
            'Graph E(1)','Graph E(2)','Graph E(3)'],
           bbox_to_anchor=(1.0,1.0),prop= {'size':8})
plt.show()


