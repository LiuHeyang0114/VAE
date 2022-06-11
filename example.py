import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from models import VAE,loss_function
import argparse


parser = argparse.ArgumentParser(description='PyTorch VAE')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')
parser.add_argument('--dim', type=int, default=2,
                    help='latent vector z dimension size')
args = parser.parse_args()

dim_size = args.dim
# Download MNIST Dataset
train_dataset = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor(), download=False)

# MNist Data Loader
batch_size=50
epoch_num=50
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

vae = VAE(input_size =784, hidden_size = 512, hidden_size2=256, z_dim=dim_size)
if torch.cuda.is_available():
    vae.cuda()

optimizer = optim.Adam(vae.parameters(),lr=args.lr)

print('====> lr: {:.4f} '.format(args.lr))
# iterator over train set
for epoch in range(1, epoch_num + 1):
    vae.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        #print("In train stage: data size: {}".format(data.size()))
        data = data.cuda()
        optimizer.zero_grad()

        recon_batch, mu, var = vae(data)        #latent vector generation
        loss = loss_function(recon_batch, data, mu, var)  #compute train_loss

        loss.backward()                   #BP
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),100. * batch_idx / len(train_loader), loss.item() / len(data)))
        if batch_idx == 0:
            nelem = data.size(0)
            nrow = 10
            save_image(data.view(nelem, 1, 28, 28), './images/image_' +str(args.lr)+'_'+ str(args.dim)+'.png', nrow=nrow)
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

# iterator over test set
vae.eval()
test_loss = 0
with torch.no_grad():
    for data, _ in test_loader:
        #print("In test stage: data size: {}".format(data.size()))
        data = data.cuda()
        recon, mu, var = vae(data)            #latent vector generation

        # sum up batch loss
        test_loss += loss_function(recon, data, mu, var).item()  #compute test_loss

test_loss /= len(test_loader.dataset)
print('====> Test set loss: {:.4f}'.format(test_loss))

# to be finished by you ...
with torch.no_grad():
    z = torch.randn(64, dim_size).cuda()   #sample latent vector
    sample = vae.decoder(z).cuda()         #push latent vector into decoder for image generation

    save_image(sample.view(64, 1, 28, 28), './samples/sample_'+str(args.lr)+'_'+ str(args.dim)+ '.png')