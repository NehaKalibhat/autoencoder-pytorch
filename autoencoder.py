import os

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from torchvision.utils import save_image
import numpy as np
import pdb
import matplotlib.pyplot as plt
import argparse

num_epochs = 100
batch_size = 128
learning_rate = 1e-3

img_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

#dataset = CIFAR10('../datasets/cifar', download=True, transform=img_transform)
dataset = MNIST('../datasets/mnist', download=True, transform=img_transform)
dataloader1 = DataLoader(dataset, batch_size=batch_size, shuffle=False)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class autoencoder(nn.Module):
    '''
    def __init__(self):
        super(autoencoder, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            nn.ReLU(),
# 			nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
#             nn.ReLU(),
        )
        self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
#             nn.ReLU(),
			nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)
    '''
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 128, 3, stride=1, padding=1),  # b, 256, 32, 32
            nn.ReLU(True),
             nn.Conv2d(128, 64, 4, stride=2, padding=1),  # b, 96, 16, 16
            nn.ReLU(True),
            nn.Conv2d(64, 32, 2, stride=2, padding=2),  # b, 48, 10, 10
            nn.ReLU(True),
            nn.Conv2d(32, 16, 3, stride=1, padding=1),  # b, 24, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 12, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 12, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 12, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 128, 3, stride=2),  # b, 24, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 2, stride=2),  # b, 48, 10, 10
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 2, stride=2, padding=1),  # b, 12, 18, 18
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 5, stride=1, padding=1),  # b, 8, 20, 20
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 1, 2, stride=2, padding=4),  # b, 1, 32, 32
            nn.Tanh()
        )
        self.encoder = self.encoder.cuda()
        self.decoder = self.decoder.cuda()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.criterion = nn.MSELoss()
        fh = open(path + "/train.log", 'w')
        fh.write('Logging')
        fh.close()
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def log(self, message):
        print(message)
        fh = open(path + "/train.log", 'a+')
        fh.write(message + '\n')
        fh.close()
        
        
    def save_autoencoder(self, model_path):
        self.log(f'Saving autoencoder as {model_path}')
        model_state = {}
        model_state['autoencoder'] = self.state_dict()
        model_state['optimizer'] = self.optimizer.state_dict()
        torch.save(model_state, model_path)
        
    def load_autoencoder(self, model_path):
        self.log(f'Loading saved autoencoder named {model_path}') 
        model_state = torch.load(model_path)
        self.load_state_dict(model_state['autoencoder'])
        self.optimizer.load_state_dict(model_state['optimizer'])
    
    def one_shot_prune(self, pruning_perc, layer_wise = False, trained_original_model_state = None):
        self.log(f"************pruning {pruning_perc} of the network****************")
        self.log(f"Layer-wise pruning = {layer_wise}")
        model = None
        if trained_original_model_state:
            model = torch.load(trained_original_model_state)
            
        if pruning_perc > 0:
            masks = {}
            flat_model_weights = np.array([])
            for name in model:
                #if "weight" in name:
                layer_weights = model[name].data.cpu().numpy()
                flat_model_weights = np.concatenate((flat_model_weights, layer_weights.flatten()))
            global_threshold = np.percentile(abs(flat_model_weights), pruning_perc)

            zeros = 0
            total = 0
            self.log("Autoencoder layer-wise pruning percentages")
            for name in model:
                #if "weight" in name:
                weight_copy = model[name].data.abs().clone()
                threshold = global_threshold
                if layer_wise:
                    layer_weights = model[name].data.cpu().numpy()
                    threshold = np.percentile(abs(layer_weights), pruning_perc)
                mask = weight_copy.gt(threshold).int()
                self.log(f'{name} : {mask.numel() - mask.nonzero().size(0)} / {mask.numel()}  {(mask.numel() - mask.nonzero().size(0))/mask.numel()}')
                zeros += mask.numel() - mask.nonzero().size(0)
                total += mask.numel()
                masks[name] = mask
            self.log(f"Fraction of weights pruned = {zeros}/{total} = {zeros/total}")
            self.masks = masks
    
    def get_percent(self, total, percent):
        return (percent/100)*total


    def get_weight_fractions(self, number_of_iterations, percent):
        percents = []
        for i in range(number_of_iterations+1):
            percents.append(self.get_percent(100 - sum(percents), percent))
        self.log(f"{percents}")
        weight_fractions = []
        for i in range(1, number_of_iterations+1):
            weight_fractions.append(sum(percents[:i]))

        self.log(f"Weight fractions: {weight_fractions}")

        return weight_fractions
    
    def iterative_prune(self, init_state, 
                        trained_original_model_state, 
                        number_of_iterations, 
                        percent = 20, 
                        init_with_old = True):
        
        
        weight_fractions = self.get_weight_fractions(number_of_iterations, percent)       
        self.log("***************Iterative Pruning started. Number of iterations: {} *****************".format(number_of_iterations))
        for pruning_iter in range(0, number_of_iterations):
            self.log("Running pruning iteration {}".format(pruning_iter))
            self.__init__()
            
            trained_model = trained_original_model_state
            if pruning_iter != 0:
                trained_model = path + "/"+ "end_of_" + str(pruning_iter - 1) + '.pth'
            
            self.one_shot_prune(weight_fractions[pruning_iter], trained_original_model_state = trained_model)
            model.train(prune = True, init_state = init_state, init_with_old = init_with_old)
            torch.save(self.state_dict(), path + "/"+ "end_of_" + str(pruning_iter) + '.pth')
            
            sample = self.forward(test_input.cuda())
            save_image(sample * 0.5 + 0.5, path + '/image_' + str(pruning_iter) + '.png')

        self.log("Finished Iterative Pruning")
        
        
    def mask_gan(self):
        model = self.state_dict()
        for name in model:
            #if "weight" in name:
            model[name].data.mul_(self.masks[name])
        self.load_state_dict(model)
    
    def train(self, prune, init_state, init_with_old):
        self.log(f"Number of parameters in model {sum(p.numel() for p in model.parameters())}")
        if not prune:
            model.save_autoencoder('./mnist/before_train.pth')
        
        if prune and init_with_old:
            model.load_autoencoder(init_state)

        for epoch in range(num_epochs):
            for data in dataloader:
                img, _ = data
                img = img.cuda()
                # ===================forward=====================
                output = self.forward(img)
                loss = self.criterion(output, img)
                # ===================backward====================
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if prune:
                    self.mask_gan()

            # ===================log========================
            self.log('epoch [{}/{}], loss:{:.4f}'
                  .format(epoch + 1, num_epochs, loss.data.item()))
            '''
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                #save_image(torch.cat((output, img), 0) * 0.5 + 0.5, './cifar/image_{}.png'.format(epoch))
                sample = self.forward(test_input.cuda())
                save_image(sample * 0.5 + 0.5, path + '/image_{}.png'.format(epoch))
            '''
        torch.save(self.state_dict(), path + '/autoencoder.pth')
        
        
test_input, classes = next(iter(dataloader1)) 
print(classes)
    
path = './mnist_random_iter'
init_state = './mnist/before_train.pth'
trained_original_model_state = './mnist/autoencoder.pth'
init_with_old = False
model = autoencoder()
#model.one_shot_prune(98, trained_original_model_state = trained_original_model_state)
model.iterative_prune(init_state = init_state, 
                    trained_original_model_state = trained_original_model_state, 
                    number_of_iterations = 20, 
                    percent = 20, 
                    init_with_old = init_with_old)
#model.train(prune = True, init_state = init_state, init_with_old = init_with_old)

