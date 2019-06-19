import os
import torch

device = torch.device("cuda:0")

from data_loader import TreeDataset, ImgDataset

tree_train = TreeDataset(tree_dir=os.path.join('bin', 'tree_train'), device=device)
img_train = ImgDataset(img_dir=os.path.join('bin', 'img_train'), device=device)

tree_eval = TreeDataset(tree_dir=os.path.join('bin', 'tree_eval'), device=device)
img_eval = ImgDataset(img_dir=os.path.join('bin', 'img_eval'), device=device)

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




from tree import Tree

def word_embedding(tree):
        tree.value = tree_train.word_dict[tree.value]
        for child in tree.children:
            self._word_embedding(child)
        return tree
    
def predictTree(image_caption_model, img):
    
    device = torch.device("cuda:0")

    root = Tree("root")
    root = word_embedding(root)
    root.value = torch.tensor(root.value).to(device).float()
    while(1):
        sub_tree = Tree(image_caption_model(img, root).flatten())
        max = 0
        max_index = 0

        for i in range(14):
            if sub_tree.value[i]>max:
                max = sub_tree.value[i]
                max_index = i
        for i in range(14):
            if i != max_index:
                sub_tree.value[i] = 0;
            else:
                sub_tree.value[i] = 1;
        root.add_child(sub_tree)
        if sub_tree.value[2] == 1:
            break
    for child in root.children:
        if child.value[3] == 1 or child.value[7] == 1 or child.value[8] == 1 or child.value[9] == 1 or child.value[12] == 1:
            predictSubTree(image_caption_model, img, child, root, 1)
    return root

def predictSubTree(image_caption_model, img, now_node, root, depth):
    if depth>3:
        return
    device = torch.device("cuda:0")

    count = 0
    while(1):
        count = count +1
        if count == 10:
            end_node = Tree("None")
            end_node = word_embedding(end_node)
            end_node.value = torch.tensor(end_node.value).to(device).float()
            now_node.add_child(end_node)
            return
        
        sub_tree = Tree(image_caption_model(img, root).flatten())
        max = 0
        max_index = 0

        for i in range(14):
            if sub_tree.value[i]>max:
                max = sub_tree.value[i]
                max_index = i
        for i in range(14):
            if i != max_index:
                sub_tree.value[i] = 0
            else:
                sub_tree.value[i] = 1
        now_node.add_child(sub_tree)
        if sub_tree.value[2] == 1:
            break
    for child in now_node.children:
        if child.value[3] == 1 or child.value[7] == 1 or child.value[8] == 1 or child.value[9] == 1 or child.value[12] == 1:
            predictSubTree(image_caption_model, img, child, root, depth+1)
        

def embedding_to_word(tree):
    tree.value = torch.tensor(tree.value).to("cpu").float().detach().numpy().flatten()
    max = 0
    max_index = 0

    for i in range(14):
        if tree.value[i]>max:
            max = tree.value[i]
            max_index = i

    for element in tree_train.word_dict:
        if tree_train.word_dict[element][max_index] == 1:
            tree.value = element
            break   
    for child in tree.children:
        embedding_to_word(child)
    return tree

def embedding_to_word_eval(tree):
    tree.value = torch.tensor(tree.value).to("cpu").float().detach().numpy().flatten()
    max = 0
    max_index = 0

    for i in range(14):
        if tree.value[i]>max:
            max = tree.value[i]
            max_index = i

    for element in tree_train.word_dict:
        if tree_eval.word_dict[element][max_index] == 1:
            tree.value = element
            break   
    for child in tree.children:
        embedding_to_word_eval(child)
    return tree



import json

f= open("./encode.json", "r")

mapping=json.load(f)
print(mapping)


import json
import numpy as np
import pandas as pd
from strkernel.mismatch_kernel import MismatchKernel
from strkernel.mismatch_kernel import preprocess
import networkx as nx
from bs4 import BeautifulSoup
from collections import deque
import matplotlib.pyplot as plt

def get_element_tuple(graph, path):
    arr=[]
    for i in path:
        arr.append(graph.nodes[i]['element'])
    return tuple(arr)

def generate_subpaths(path, l, graph):
    if l >= len(path):
        tuple_path=get_element_tuple(graph, path)
        # tuple_path=tuple(path)
        if tuple_path not in subpath_track:
            subpath_track[tuple_path] = 1
        else:
            subpath_track[tuple_path] += 1
    else:
        index = 0
        while l+index-1 < len(path):
            tuple_path=get_element_tuple(graph, path[index: l+index])
            # tuple_path=tuple(path[index: l+index])
            if tuple_path not in subpath_track:
                subpath_track[tuple_path] = 1
            else:
                subpath_track[tuple_path] += 1
            index += 1

        generate_subpaths(path, l+1, graph)


def get_subpaths(graph, root, track, path):
    track[root] = True # record visited nodes
    if graph.degree(root) == 1:
        
        generate_subpaths(path, 1, graph)
    else:
        for node in graph.neighbors(root):
            if node not in track: # if node not visited
                get_subpaths(graph, node, track, path + [node, ])
                
def get_kernel(subpath_track_1, subpath_track_2):
    decay_rate=0.75
    kernel_v=0
    for p in subpath_track_1:
        for q in subpath_track_2:
            if p==q:    
                kernel_v+=subpath_track_1[p]*subpath_track_2[q]/pow(decay_rate, len(q)-1)
    return kernel_v

def get_normalized_kernel(subpath_track_1, subpath_track_2):
    kernel_12 = get_kernel(subpath_track_1, subpath_track_2)
    kernel_1 = get_kernel(subpath_track_1, subpath_track_1)
    kernel_2 = get_kernel(subpath_track_2, subpath_track_2)
    if kernel_1 < kernel_2:
        kernel_1=kernel_2
    return kernel_12/kernel_1
graph = nx.Graph()

def add_leaf(root_id, content, content_id, node_id):
    while content[content_id]!="0" and content[content_id]!="2":
        if content[content_id]!="1":
            graph.add_node(node_id, element=content[content_id])
            graph.add_edge(root_id, node_id)
            node_id=node_id+1
        else:
            content_id, node_id= add_leaf(node_id-1, content, content_id+1, node_id)
            
        content_id = content_id+1
    return content_id, node_id
    
def dsl_to_dom_tree(root, content):
    node_id = 1
    content_id = 0
    
    graph.add_node(node_id, element="root")
    
    node_id=node_id+1
    content_id=0
    content_id=add_leaf(1, content, content_id, node_id)

def tree_encode(tree, mapping, content):
    if tree.value!='None' and tree.value!='root':
        content.append(mapping[tree.value])
    if len(tree.children)!=0:
        content.append(mapping["{"])
        for child in tree.children:
            tree_encode(child, mapping, content)
        content.append(mapping["}"])






num = 10
img = img_eval[num]
image_caption_model.eval()
t1 = predictTree(image_caption_model, img)

t1 = embedding_to_word(t1)
content = []
tree_encode(t1, mapping, content)
del t1
for i in range(88 - len(content)):
    content.append("0")
content = np.asarray(content)

graph = nx.Graph()
dsl_to_dom_tree(1, content)
G_1=graph
nx.draw(G_1)
# plt.show()


origin_tree = embedding_to_word_eval(tree_eval[num].copy())
content = []
tree_encode(origin_tree, mapping, content)
for i in range(88 - len(content)):
    content.append("0")
content = np.asarray(content)

graph = nx.Graph()
dsl_to_dom_tree(1, content)
G_2=graph
nx.draw(G_2)


subpath_track = {}
track={}
path=[]
get_subpaths(G_1, 1, track ,path)
subpath_track_1=subpath_track

subpath_track = {}
track={}
path=[]
get_subpaths(G_2, 1, track ,path)
subpath_track_2=subpath_track

kernel_v = get_normalized_kernel(subpath_track_1, subpath_track_2)
print(kernel_v)