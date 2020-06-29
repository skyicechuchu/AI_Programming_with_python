import os
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import argparse
import logging
from collections import OrderedDict
import matplotlib.pyplot as plt


class Train:
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.train_dir = data_dir + '/train'
        self.valid_dir = data_dir + '/valid'
        self.test_dir = data_dir + '/test'
        
       
    def transform_data(self):
        train_data_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                            transforms.RandomRotation(30),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                 [0.229, 0.224, 0.225])])

        valid_data_transforms = transforms.Compose ([transforms.Resize (255),
                                             transforms.CenterCrop (224),
                                             transforms.ToTensor (),
                                             transforms.Normalize ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                            ])

        test_data_transforms = transforms.Compose ([transforms.Resize (255),
                                             transforms.CenterCrop (224),
                                             transforms.ToTensor (),
                                             transforms.Normalize ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                            ])


        self.train_image_datasets = datasets.ImageFolder(self.train_dir, transform=train_data_transforms)
        self.valid_image_datasets = datasets.ImageFolder(self.valid_dir, transform=valid_data_transforms)
        self.test_image_datasets = datasets.ImageFolder(self.test_dir, transform=test_data_transforms)


        self.train_dataloaders = torch.utils.data.DataLoader(self.train_image_datasets, batch_size=64, shuffle=True)
        self.valid_dataloaders = torch.utils.data.DataLoader(self.valid_image_datasets, batch_size=64, shuffle=True)
        self.test_dataloaders = torch.utils.data.DataLoader(self.test_image_datasets, batch_size=64, shuffle=True)
    
    def build_models(self, arch, learning_rate, hidden_units, device):
        self.model = eval("models.{}(pretrained=True)".format(arch))

        for param in self.model.parameters():
            param.requires_grad = False
            
        classifier = nn.Sequential  (OrderedDict ([
                            ('fc1', nn.Linear (25088, int(hidden_units[0]))),
                            ('relu1', nn.ReLU ()),
                            ('dropout1', nn.Dropout (0.05)),
                            ('fc2', nn.Linear (int(hidden_units[0]), int(hidden_units[1]))),
                            ('relu2', nn.ReLU ()),
                            ('dropout', nn.Dropout (0.05)),
                            ('fc3', nn.Linear (int(hidden_units[1]), 102)),
                            ('output', nn.LogSoftmax (dim =1))
                            ]))
        self.model.classifier = classifier
        
        
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.classifier.parameters(), lr = learning_rate)
        self.model.to(device)
        
    def training_validation(self,device, epochs):
        running_loss = 0
        steps = 0
        print_every = 40
        train_losses, valid_losses = [], []
        
        for epoch in range(epochs):
            for inputs, labels in self.train_dataloaders:
                steps += 1
                inputs, labels = inputs.to(device), labels.to(device)
                self.optimizer.zero_grad()
        
                logps = self.model.forward(inputs)
                loss = self.criterion(logps, labels)
                loss.backward()
                self.optimizer.step()
        
                running_loss += loss.item()
        
                if steps % print_every == 0:
                    valid_loss = 0
                    accuracy = 0
                    self.model.eval()
                    with torch.no_grad():
                        for inputs, labels in self.valid_dataloaders:
                            inputs, labels = inputs.to(device), labels.to(device)
                            logps = self.model.forward(inputs)
                            batchloss = self.criterion(logps, labels)
                    
                            valid_loss += batchloss.item()
                    
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
                    train_losses.append(running_loss/len(self.train_dataloaders))
                    valid_losses.append(valid_loss/len(self.valid_dataloaders))
                    
                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Valid loss: {valid_loss/len(self.valid_dataloaders):.3f}.. "
                          f"valid accuracy: {accuracy/len(self.valid_dataloaders):.3f}")
                    running_loss = 0
                    self.model.train()
        return train_losses, valid_losses
         
        
    def test_model(self, device):
        total = 0
        correct = 0
        with torch.no_grad():
            for inputs, labels in self.test_dataloaders:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    
    def save_checkpoint(self, save_dir):
        checkpoint = {'state_dict': self.model.state_dict(),
                      'class_to_idx': self.train_image_datasets.class_to_idx,
                      'opt_state': self.optimizer.state_dict}
        file_path = os.path.join(save_dir, 'checkpoint.pth')
        torch.save(checkpoint, file_path)
        
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('data_dir', type=str, help='data directory')
    parser.add_argument('--save_dir', type=str, default='./', help='save model to check points')
    parser.add_argument('--arch', type=str, default='vgg16', help="pre-train model")
    parser.add_argument('--learning_rate', type=float, default='0.001', help='set up the learning rate for training')
    parser.add_argument('--hidden_units', nargs='+', help='set up the hidden units for model')
    parser.add_argument('--epochs', type =int, default='7', help='set up epochs for training cycle')
    parser.add_argument('--gpu', action='store_true', help = 'enable the GPU')
    
    args = parser.parse_args()
    
    data_dir = args.data_dir
    save_dir = args.save_dir
    arch = args.arch
    learning_rate = args.learning_rate
    hidden_units = args.hidden_units
    epochs = args.epochs
    
    cuda = False
    if args.gpu:
        if torch.cuda.is_available():
            cuda = True
        else:
            logging.warning("GPU is not exist, use CPU instead")        
    device = "cuda" if cuda else "cpu"  
    
    train_obj = Train(data_dir)
    train_obj.transform_data()
    train_obj.build_models(arch, learning_rate, hidden_units, device)
    train_losses, valid_losses = train_obj.training_validation(device, epochs)
    train_obj.test_model(device)
    train_obj.save_checkpoint(save_dir)

