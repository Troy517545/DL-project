import os
import torch

device = torch.device("cuda:0")

from data_loader import TreeDataset, ImgDataset

tree_train = TreeDataset(tree_dir=os.path.join('bin', 'tree_train'), device=device)
img_train = ImgDataset(img_dir=os.path.join('bin', 'img_train'), device=device)

import time
import numpy as np
from models import ShowAndTellTree
import torch.optim as optim

def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)

image_caption_model = ShowAndTellTree(len(tree_train.word_dict)).to(device)
image_caption_model.apply(weights_init_uniform_rule)

checkpoint = torch.load("weights/weight.pth")
image_caption_model.load_state_dict(checkpoint['model_state_dict'])

for i , (tree, img) in enumerate(zip(tree_train, img_train)):
    split_tree = tree.copy()
    next_node = image_caption_model(img, split_tree)
    print(next_node)
    break