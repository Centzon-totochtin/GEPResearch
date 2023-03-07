'''
    adjacent matrix of graphs
'''
import numpy as np
import torch

graphs=[np.array([[0,1,0,1,0],
                  [1,0,1,0,0],
                  [0,1,0,1,0],
                  [1,0,1,0,1],
                  [0,0,0,1,0]]),
        np.array([[0,1,1,0,0],
                  [1,0,1,0,0],
                  [1,1,0,1,1],
                  [0,0,1,0,0],
                  [0,0,1,0,0]]),
        np.array([[0,1,0,0,0],
                  [1,0,1,0,0],
                  [0,1,0,1,0],
                  [0,0,1,0,1],
                  [0,0,0,1,0]]),
        np.array([[0,1,1,1,1],
                  [1,0,0,0,0],
                  [1,0,0,0,0],
                  [1,0,0,0,0],
                  [1,0,0,0,0]]),
        np.array([[0,1,1,0,0],
                  [1,0,0,0,0],
                  [1,0,0,1,1],
                  [0,0,1,0,0],
                  [0,0,1,0,0]])]
graphs =[torch.tensor(x,dtype=torch.float32) for x in graphs]