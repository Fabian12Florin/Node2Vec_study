import numpy as np
import networkx as nx
from gensim.models import Word2Vec

class Node2Vec:
    def __init__(self, graph, dimensions=128, walk_length=80, num_walks=10, p=1, q=1):
        self.graph = graph
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.walks = []

    def node2vec_walk(self, start_node):
        walk = [start_node]
        while len(walk) < self.walk_length:
            cur = walk[-1]
            cur_neighbors = list(self.graph.neighbors(cur))
            if len(cur_neighbors) > 0:
                if len(walk) == 1:
                    walk.append(np.random.choice(cur_neighbors))
                else:
                    prev = walk[-2]
                    next_node = self.alias_draw(self.alias_nodes[cur_neighbors])
                    walk.append(next_node)
            else:
                break
        return walk

    def simulate_walks(self):
        nodes = list(self.graph.nodes())
        for _ in range(self.num_walks):
            np.random.shuffle(nodes)
            for node in nodes:
                self.walks.append(self.node2vec_walk(node))

    def learn_embeddings(self):
        self.simulate_walks()
        walks = [list(map(str, walk)) for walk in self.walks]
        model = Word2Vec(walks, vector_size=self.dimensions, window=10, min_count=0, sg=1, workers=4)
        return model

    def get_alias_edge(self, src, dst):
        unnormalized_probs = []
        for dst_nbr in self.graph.neighbors(dst):
            if dst_nbr == src:
                unnormalized_probs.append(self.graph[dst][dst_nbr]['weight'] / self.p)
            elif self.graph.has_edge(dst_nbr, src):
                unnormalized_probs.append(self.graph[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(self.graph[dst][dst_nbr]['weight'] / self.q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
        return self.alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        self.alias_nodes = {}
        self.alias_edges = {}
        for node in self.graph.nodes():
            unnormalized_probs = [self.graph[node][nbr]['weight'] for nbr in self.graph.neighbors(node)]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
            self.alias_nodes[node] = self.alias_setup(normalized_probs)

        for edge in self.graph.edges():
            self.alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])

    def alias_setup(self, probs):
        K = len(probs)
        q = np.zeros(K)
        J = np.zeros(K, dtype=np.int)
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            q[kk] = K * prob
            if q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)
        while smaller and larger:
            small = smaller.pop()
            large = larger.pop()
            J[small] = large
            q[large] = q[large] + q[small] - 1.0
            if q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)
        return J, q

    def alias_draw(self, alias_data):
        J, q = alias_data
        K = len(J)
        kk = int(np.floor(np.random.rand() * K))
        if np.random.rand() < q[kk]:
            return kk
        else:
            return J[kk]
