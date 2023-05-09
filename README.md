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
edge_list=pd.read_csv("my_data.csv")
# printing head of dataset
edge_list.head()
# string to integers
 edge_list = edge_list.applymap(str)
# create undirected graph from the edgelist
G=nx.from_pandas_edgelist(edge_list, source='node_1', target='neighbor_1', create_using=nx.Graph())
# printing nodes
G.nodes()
# function to generate random walk sequences of nodes for a particular node
def get_random_walk(node, walk_length):
    
    random_walk_length = [node]
    
    for i in range(walk_length-1):
        neighbors = list(G.neighbors(node))
        neighbors = list(set(neighbors) - set(random_walk_length))    
        if len(neighbors) == 0:
            break
        random_neighbor = random.choice(neighbors)
        random_walk_length.append(random_neighbor)
        node = random_neighbor
        
    return random_walk_length
# create undirected graph from the edgelist
G = nx.from_pandas_edgelist(edge_list, source='node_1', target='neighbor_1', create_using=nx.Graph()) 

def get_random_walk(node, walk_length):
  
    random_walk_length = [node]
    
    for i in range(walk_length-1):
        # list of neighbors
        neighbors = list(G.neighbors(node))
        neighbors = list(set(neighbors) - set(random_walk_length))    
        if len(neighbors) == 0:
            break
        random_neighbor = random.choice(neighbors)
        random_walk_length.append(random_neighbor)
        node = random_neighbor
        
    return random_walk_length

from gensim.models import Word2Vec
sentences = []
for node in G.nodes():

    walks = [get_random_walk(node, walk_length=10) for _ in range(5)]
    sentences += walks
    
model = Word2Vec(sentences, window=4, sg=1, hs=0,
                 negative=10, # for negative sampling
                 alpha=0.03, min_alpha=0.0007, #reduce alpha value as per requirement
                 seed=14)

model.build_vocab(sentences, progress_per=2)

model.train(sentences, total_examples=model.corpus_count, epochs=20, report_delay=1)
# random_walks

all_nodes = list(G.nodes())
number_of_random_walks = 5
random_walks = []

for node in tqdm(all_nodes):

    for i in range(number_of_random_walks):
        random_walks.append(get_random_walk(node, 10))
# check random walk for node  particular node like-'2'
get_random_walk('10.1590/S0103-50532003000300007', 10)
# printing similar nodes using nodes i'd
"" like node i'd ='10.1590/S0103-50532003000300007' then the result will be" 
 for node, _ in model.wv.most_similar('10.1590/S0103-50532003000300007'):
    print((node, _))
 output = ('Urucuca/BA', 0.8874707221984863)("3',4',5',3,5,7,8-heptamethoxyflavone", 0.8841790556907654)("3,5,6,7,3',4',5'-heptamethoxyflavonol", 0.8646891713142395)('Murraya paniculata', 0.8487561941146851)("8-hydroxy-3,5,7,3',4',5'-hexamethoxyflavonol", 0.7443819642066956)('10.1016/S0031-9422(97)00598-0', 0.7366040945053101)('Rapanea lancifolia (Myrsinaceae)', 0.7098177671432495)
('Oleanonic acid', 0.6936820149421692)('taxifolin', 0.6897762417793274)("3',4',5',5,7,8-hexamethoxyflavone", 0.6854137778282166). similarly change the nodes values 
# printing undirected graph
nx.draw_networkx(G). 
the result will be very complex not in readiable format we can use pca(principal component analysis) for that
# using prinicpal component analysis
def plot_nodes(word_list):
    X = model.wv[word_list]
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    
    
    plt.figure(figsize=(12,9))
    plt.scatter(result[:, 0], result[:, 1])
    for i, word in enumerate(word_list):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))
        
    plt.show()
  
# checking result
numbers = list(G.nodes)
plot_nodes(numbers)
# using hist@k for evaluation for deepwalkk(undirected graph)
code for hits@k
node2idx = {node: idx for idx, node in enumerate(model.wv.index_to_key)}
def calculate_hits_k(k):
    hits = 0
    total_nodes = len(G.nodes())
    for node in G.nodes():
        node_idx = node2idx[node]
        node_vec = model.wv[node]
        cos_sims = model.wv.cosine_similarities(node_vec, model.wv.vectors)
        top_k_idxs = cos_sims.argsort()[::-1][1:k+1]
        top_k_nodes = [model.wv.index_to_key[idx] for idx in top_k_idxs]
        for neighbor in G.neighbors(node):
            if neighbor in top_k_nodes:
                hits += 1
    hits_at_k = hits / (total_nodes * k)
    return hits_at_k
k_values = [1, 2, 3,4,5]
for k in k_values:
    hits_at_k = calculate_hits_k(k)
print(f"Hits@{k}: {hits_at_k:.4f}")
  # output of hits@k
Hits@1: 0.1827
Hits@2: 0.2049
Hits@3: 0.2115
Hits@4: 0.2088
Hits@5: 0.1952
# for directed graph
install  weighted-metapath2vec ,from node2vec import Node2Vec
# use coulmn '0' and '1' as a source and target
edge_list.rename(columns = {0:'source', 1: 'target'}, inplace = True)
# initalize the model
G=nx.from_pandas_edgelist(edge_list, source='node_1', target='neighbor_1', create_using=nx.Graph())
node2vec = Node2Vec(G, dimensions=128, walk_length=40, num_walks=100, workers=2)
model = node2vec.fit(window=10, min_count=1) // parameters taken here are normal not specific to purpose so change as per requirement 
# verify the most similar nodes generated by node2vec from the graph
nx.draw(G, with_labels=True)
plt.show()
largest_cc = sorted(nx.connected_components(G), key=len, reverse=True)[:] // change values under [:]
subgraph = G.subgraph(set().union(*largest_cc))

# verify the most similar nodes generated by node2vec from the graph without labels
nx.draw(G, with_labels=False)
plt.show()
largest_cc = sorted(nx.connected_components(G), key=len, reverse=True)[:]
subgraph = G.subgraph(set().union(*largest_cc))

#  to see individual connected node use their node id to print connnected node 
\# change the node ID order to get knowldge garph
similar_nodes = model.wv.most_similar('10.1021/np50086a012', topn=10)// change node id in order to get connected graph for any individual node also values of top
node_ids = [node_id for node_id, _ in similar_nodes]
subgraph = G.subgraph(node_ids)
nx.draw_networkx(subgraph)

# for evaluation 
from sklearn.metrics.pairwise import cosine_similarity


# Step 1: Create node2idx dictionary
node2idx = {node: idx for idx, node in enumerate(model.wv.index_to_key)}

# Step 2: Define function to calculate Hits@K
def calculate_hits_k(k):
    hits_k = 0
    for node in subgraph.nodes():
        neighbors = [n for n in subgraph.neighbors(node)]
        if len(neighbors) < k:
            continue
        node_idx = node2idx[node]
        neighbor_indices = [node2idx[n] for n in neighbors]
        similarity_scores = cosine_similarity([model.wv[node_idx]], model.wv[neighbor_indices])
        top_k_indices = np.argsort(similarity_scores, axis=1)[0][-k:]
        top_k_neighbors = [neighbors[i] for i in top_k_indices]
        hits_k += int(node in top_k_neighbors)
    return hits_k / len(subgraph)

# Step 3: Calculate Hits@K for k=1,2,3,4,5
for k in range(1,2):
    hits_k = calculate_hits_k(k)
    print(f"Hits@{k}: {hits_k:.4f}")
output-Hits@1: 0.0000
       Hits@2: 0.0000
      
      
  # Note
  remember all parametrs values taken here is not static so  that it can give any speicific output som change according to requirements
