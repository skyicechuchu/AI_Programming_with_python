import os
import json
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


class Predict:
    def __init__(self, image_path):
        self.image_path = image_path
       

    def load_checkpoint(self, checkpoint):
        checkpoint = torch.load(checkpoint)
        self.model = models.vgg16(pretrained=True)
        for param in self.model.parameters(): 
            param.requires_grad = False
            
        classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear (25088, 4096)),
                            ('relu1', nn.ReLU ()),
                            ('dropout1', nn.Dropout (0.05)),
                            ('fc2', nn.Linear (4096, 2048)),
                            ('relu2', nn.ReLU ()),
                            ('dropout', nn.Dropout (0.05)),
                            ('fc3', nn.Linear (2048, 102)),
                            ('output', nn.LogSoftmax (dim =1))
                            ]))
        
        self.model.classifier = classifier
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.class_to_idx = checkpoint['class_to_idx']
        
    def process_image(self):
        img = Image.open(self.image_path)

        original_width, original_height = img.size

        if original_width < original_height:
            size=[256, 256**600]
        else: 
            size=[256**600, 256]
        
        img.thumbnail(size)
        center = original_width/4, original_height/4
        left, top, right, bottom = center[0]-(244/2), center[1]-(244/2), center[0]+(244/2), center[1]+(244/2)
        img = img.crop((left, top, right, bottom))

        numpy_img = np.array(img)/255 

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        numpy_img = (numpy_img-mean)/std

        numpy_img = numpy_img.transpose(2, 0, 1)
    
        return numpy_img
    
    def imshow(self, ax=None, title=None):
        if ax is None:
            fig, ax = plt.subplots()

        image = self.image_path.transpose((1, 2, 0))
    
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)
        ax.imshow(image)
    
        return ax
    
    def predict(self, top_k, device, category_names=None):
        self.model.to(device)
        self.model.eval()
        torch_image = torch.from_numpy(np.expand_dims(self.process_image(), 
                                                      axis=0)).type(torch.FloatTensor).to(device)

        log_probs = self.model.forward(torch_image)
        linear_probs = torch.exp(log_probs)
        top_probs, top_labels = linear_probs.topk(top_k)
    
        top_probs = np.array(top_probs.detach())[0] 
        top_labels = np.array(top_labels.detach())[0]
    
        idx_to_class = {val: key for key, val in self.model.class_to_idx.items()}
        top_labels = [idx_to_class[lab] for lab in top_labels]
        
        if category_names:
            with open(category_names, 'r') as f:
                cat_to_name = json.load(f)
            print(cat_to_name)
            class_name = [cat_to_name[i] for i in top_labels]
            
        
        return top_probs, top_labels, class_name
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict based on the model')
    parser.add_argument('image_path', type=str, help='provide an image path')
    parser.add_argument('checkpoint', type=str, help='models checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help="return top k most likely calsses")
    parser.add_argument('--category_names', type=str,  help='a mapping of categories to real names from a json file')
    parser.add_argument('--gpu', action='store_true', help = 'enable the GPU')

    args = parser.parse_args()
    image_path = args.image_path
    checkpoint = args.checkpoint
    top_k = args.top_k
    category_names = args.category_names
    
    cuda = False
    if args.gpu:
        if torch.cuda.is_available():
            cuda = True
        else:
            logging.warning("GPU is not exist, use CPU instead")        
    device = "cuda" if cuda else "cpu" 
    
    predict = Predict(image_path)
    predict.load_checkpoint(checkpoint)
    numpy_img = predict.process_image()
    top_probs, top_labels, class_name = predict.predict(top_k, device, category_names)
    print("="*80)
    print(" "*35 + 'FLOWER PREDICTOR')
    print("="*80)
    print("Input label (or labels) = {}".format(top_labels))
    print("Probability confidence(s) = {}".format(top_probs))
    print("Class(es) name(s) = {}".format(class_name))
    print("="*80)
    
    
   