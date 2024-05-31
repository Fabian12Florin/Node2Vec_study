import networkx as nx
from node2vec import Node2Vec
import matplotlib.pyplot as plt

# Create a graph
G = nx.karate_club_graph()

# Apply the node2vec algorithm
node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, p=0.25, q=0.25)
node2vec.preprocess_transition_probs()
model = node2vec.learn_embeddings()

# Get embeddings
embeddings = model.wv

# Visualize embeddings
def plot_embeddings(embeddings):
    X = []
    Y = []
    for node in embeddings.index_to_key:
        X.append(embeddings[node][0])
        Y.append(embeddings[node][1])
    plt.scatter(X, Y)
    for i, node in enumerate(embeddings.index_to_key):
        plt.annotate(node, (X[i], Y[i]))
    plt.show()

plot_embeddings(embeddings)

