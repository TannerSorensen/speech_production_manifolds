
#!/usr/bin/env python3

import os
import sys
from glob import glob
import json
from itertools import compress
import argparse

import numpy as np
from skimage import io
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

class VocalTractDataset(Dataset):
    """Vocal Tract dataset."""

    def __init__(self, align_dir, root_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
        """
        self.root_dir = root_dir
        self.align_dir = align_dir

        # get list of filenames
        self.fname = sorted(glob(os.path.join(self.root_dir, '*', 'png', '*', '*.png')))

        # get forced alignments
        self.alignments = self.label_phones(self.fname, self.align_dir)

        # only keep aligned data
        self.fname = list(compress(self.fname, [x is not None for x in self.alignments]))
        self.alignments = list(compress(self.alignments, [x is not None for x in self.alignments]))
        
    def __len__(self):
        return len(self.fname)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get image
        image = torch.from_numpy(io.imread(self.fname[idx]) / np.iinfo(np.uint16).max).float()

        # get target
        target = self.alignments[idx]

        return image, target

    def label_phones(self, filenames, align_dir):
        tr_per_img = 2.0
        tr = 0.006004
        fs = 1.0 / (tr_per_img * tr)

        def sec2frame(t):
            s = int(np.round(t * fs))
            return s

        phone_list = [None] * len(filenames)
        avi_list = [ f.split(os.path.sep)[-2] for f in filenames ]
        img_list = [ int(f.split(os.path.sep)[-1].replace('.png','')) for f in filenames ]
    
        # list json files
        json_files = glob(os.path.join(align_dir,'*','align','*.json'))
    
        for json_idx, json_filename in enumerate(json_files[:1]):
            sys.stdout.write('\r'+str(json_idx+1)+'/'+str(len(json_files))+' '+json_filename)
    
            # load the json file
            d = json.loads(open(json_filename).read())
    
            # determine which avi file the json file corresponds to
            avi = json_filename.split(os.path.sep)[-1].replace('.json','')
    
            for w in d['words']:
                if w['case']=='success':
                    w_name = w['alignedWord']
    
                    # get start time for word
                    word_start_time = float(w['start'])
    
                    # offset tracks time since start of word
                    offset = 0.0
                    for ph in w['phones']:
                        ph_name = ph['phone']
    
                        # get onset and end of phone in seconds
                        ons = word_start_time + offset
                        offset += float(ph['duration'])
                        end =  ons + offset
    
                        # convert onset and end of phone to frames
                        ons = sec2frame(ons)
                        end = sec2frame(end)
    
                        # set phone for all images in range
                        done = object()
                        it = (idx for (idx, (a, img)) in enumerate(zip(avi_list, img_list)) if (a == avi) and (img <= end))
                        idx = next(it, done)
                        while (idx is not done):
                            phone_list[idx] = ph_name.split('_')[0]
                            idx = next(it, done)
    
        return phone_list

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        #self.fc1 = nn.Linear(784, 400)
        self.fc1 = nn.Linear(84*84, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        #self.fc4 = nn.Linear(400, 784)
        self.fc4 = nn.Linear(400, 84*84)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        #mu, logvar = self.encode(x.view(-1, 784))
        mu, logvar = self.encode(x.view(-1, 84*84))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

dataset = VocalTractDataset(
            align_dir="/home/tsorense/speech_production_manifolds_alignments/",
            root_dir="/home/tsorense/speech_production_manifolds/speech_production_manifolds_data/")
train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    #BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 84*84), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (image, _) in enumerate(train_loader):
        image = image.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(image)
        loss = loss_function(recon_batch, image, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(image), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(image)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 84, 84),
                       'results/sample_' + str(epoch).zfill(3) + '.png')
