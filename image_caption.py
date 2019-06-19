#%%
import os
import torch

#%% load data
device = torch.device("cuda:0")

from data_loader import TreeDataset, ImgDataset

tree_train = TreeDataset(tree_dir=os.path.join('bin', 'tree_train'), device=device)
img_train = ImgDataset(img_dir=os.path.join('bin', 'img_train'), device=device)

#%% init model
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

#%% training
from tensorboardX import SummaryWriter

epoch = 1
lr = 1e-5
print_frequency = 1
save_frequency = 50
optimizer = optim.Adam(image_caption_model.parameters(), lr=lr)
criterion = torch.nn.BCELoss()

print("Model's state_dict:")
for param_tensor in image_caption_model.state_dict():
    print(param_tensor, "\t", image_caption_model.state_dict()[param_tensor].size())

print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])


# checkpoint = torch.load("weights/weight.pth")
# image_caption_model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']


image_caption_model.train()
for e in range(epoch):
    start = time.time()
    for i , (tree, img) in enumerate(zip(tree_train, img_train)):
        # genarate new dataset
        split_tree = tree.copy()
        queue = [split_tree]
        bfs_seq = []
        while len(queue) != 0:
            # add new childern to queue
            node = queue.pop(0)
            bfs_seq.append(node)
            queue += node.children
        losses = []
        while len(bfs_seq) > 1:
            node = bfs_seq.pop()
            node.parent.num_children -= 1
            node.parent.children.remove(node)
            next_node = image_caption_model(img, split_tree)
            
            # training
            optimizer.zero_grad()
            loss = criterion(next_node, node.value.view(1,-1))
            losses.append(loss)
            loss.backward()
            optimizer.step()
                        
        if i%print_frequency == 0:
            print('tree:{} loss:{}'.format(i, sum(losses)/len(losses)))

        if i%save_frequency ==0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': image_caption_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss':  sum(losses)/len(losses)
                }, os.path.join("weights/weight.pth"))
            print("Save model!")

    end = time.time()
    print('epoch: {} time: {:2f}'.format(e, end-start))