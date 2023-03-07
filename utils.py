import torch

def FB(x):
    ''' calculate factor alpha'''
    return torch.floor(torch.log10(x))

def FA(x):
    '''calculate factor beta'''
    return x/torch.pow(10,FB(x))

def F(x,T,a=0.5,b=2):
    '''forward iteration function'''
    x0=x
    for i in range(T):
        x= (torch.mm(x0,x0)+a)/b
        x0=x
    return x
'''
def distance(x,y):
    used in calculating center distance
    return torch.dist(x,y,p=2).detach().numpy()
'''

def distance(x,y):
    return torch.sum(torch.sqrt((x-y)**2)).detach().numpy()
def shuffling(x):
    n = len(x)
    id = torch.randperm(n)
    x = x[id][:,id]
    return x