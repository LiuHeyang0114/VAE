import torch
x = torch.arange(-5,6,1)
y = torch.arange(-5,6,1)
resultx,resulty = torch.meshgrid(x,y)
print(resultx.reshape(1,121))
print(resulty)
z = torch.cat((resultx.reshape(121,1),resulty.reshape(121,1)),dim = 1).float()
print(z.type())
print(z.size())

k = (10*torch.rand(64, 2)-5)
print(k.type())