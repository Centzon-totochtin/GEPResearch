import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from data import graphs
from utils import FA,F,shuffling
import matplotlib.pyplot as plt

class autuencoder(nn.Module):
    def __init__(self):
        super(autuencoder, self).__init__()
        self.fc1 = nn.Linear(25,2)
        self.fc2 = nn.Linear(2,25)
        self.act = nn.LeakyReLU()


    def encode(self,x):
        b,_,_ =x.shape
        x = x.reshape(b,25)
        x = self.act(self.fc1(x))
        return x

    def decode(self,x):
        b,_ =x.shape
        x = self.act(self.fc2(x))
        x = x.reshape(b,5,5)
        return x
    def forward(self,x):
        z = self.encode(x)
        x = self.decode(z)
        return x

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
data = []
for i in range(graph_num):
    group=[]
    for j in range(shuffle_num):
        Ft = F(new_matrix[i][j],T=T,a=a,b=b)
        Fa = FA(Ft)
        data.append(Fa)
print(data)
data = torch.stack(data,dim=0)

if __name__=='__main__':
    autoencoder = autuencoder()
    optimizer = torch.optim.Adam(autoencoder.parameters(),lr=2e-1)
    criterion = nn.MSELoss()
    for i in range(3000):
        yp = autoencoder(data)
        loss = criterion(yp,data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss)

torch.save(autoencoder,'autoencoder.pt')
f = autoencoder.encode(data)
#torch.save(f,'low_dimension_feature.pt')

