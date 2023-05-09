# knowledge-graph
# deepwalk
# node2vec
In these repository , i have created knowledge graph using dataset provided inn BIKE challenge , where u can find linkage of nodes just by entering node value and other parameters just by own.
# initially , import these libraries 
'''import networkx as nx
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
%matplotlib inline
from gensim.models import Word2Vec
import word2vec
import node2vec
import warnings
warnings.filter
#  importing dataset
dataset=pd.read_csv("example.csv")
# undirected graph using deepwalk
G is the variable used where source="node_1"and target="neighbor_1"
# create a function for random walk
def get_random_walk(node, walk_length):
# creating undirected graph
all_nodes = list(G.nodes())
number_of_random_walks = 5
random_walks = []

for node in tqdm(all_nodes):

    for i in range(number_of_random_walks):
        random_walks.append(get_random_walk(node, 10))
# check random walk for node  particular node like-'2'
get_random_walk('10.1590/S0103-50532003000300007', 10)
# printing similar nodes using nodes i'd
 like node i'd ='10.1590/S0103-50532003000300007' then the result will be ('Urucuca/BA', 0.8874707221984863)("3',4',5',3,5,7,8-heptamethoxyflavone", 0.8841790556907654)("3,5,6,7,3',4',5'-heptamethoxyflavonol", 0.8646891713142395)('Murraya paniculata', 0.8487561941146851)("8-hydroxy-3,5,7,3',4',5'-hexamethoxyflavonol", 0.7443819642066956)('10.1016/S0031-9422(97)00598-0', 0.7366040945053101)('Rapanea lancifolia (Myrsinaceae)', 0.7098177671432495)
('Oleanonic acid', 0.6936820149421692)('taxifolin', 0.6897762417793274)("3',4',5',5,7,8-hexamethoxyflavone", 0.6854137778282166). similarly change the nodes values 
# printing undirected graph
nx.draw_networkx(G). 
the result will be very complex not in readiable format we can use pca(principal component analysis) for that

# using prinicpal component analysis
def plot_nodes(word_list):
  
# checking result
numbers = list(G.nodes)
plot_nodes(numbers)
# using hist@k for evaluation for deepwalkk(undirected graph)
node2idx = {node: idx for idx, node in enumerate(model.wv.index_to_key)}
let's k value be k_values = [1, 2] then print(f"Hits@{k}: {hits_at_k:.4f}"), result will be 
Hits@1: 0.1827 ,Hits@2: 0.2049
# for directed graph
install  weighted-metapath2vec ,from node2vec import Node2Vec
use coulmn '0' and '1' as a source and target
# initalize the model
after taking parameters like dimensions ,walklength,num_walks,workers etc.
train model 
# results 
print the the graph without label to make it understable other wise use "with_labels=True"
#  to see individual connected node use their node id to print connnected node 
here i have created an function which only print required coonected node that are required to see using node_1 i'd
# for evaluation 
use hits@k 
