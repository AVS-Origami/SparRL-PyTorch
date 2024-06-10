import networkx as nx
import community as community_louvain
from networkx.algorithms.community import greedy_modularity_communities
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import random

class Graph:
    def __init__(self, args):
        self.args = args

        if self.args.tree:
            print("Making graph from tree.")
            self._G = self.parse_tree(self.args.edge_list)
        elif self.args.is_dir:
            print("Making directed.")
            self._G = nx.read_edgelist(self.args.edge_list, nodetype=int, create_using=nx.DiGraph)
        else:
            if self.args.obj == "spearman":
                print("Making directed.")
                self._G = nx.read_edgelist(self.args.edge_list, nodetype=int, create_using=nx.DiGraph)
            else:
                print("Making undirected.")
                self._G = nx.read_edgelist(self.args.edge_list, nodetype=int)
        
        self._relabel_nodes()
        print("self.get_num_edges()", self.get_num_edges())

    def _relabel_nodes(self):
        """Relabel nodes to [1, |V|]."""
        mapping = dict(zip(self._G.nodes, range(1,self.num_nodes+1)))
        self._G = nx.relabel_nodes(self._G, mapping)
        
    def add_edge(self, src_id, dst_id):
        if not isinstance(src_id, int):
            src_id = int(src_id)
        if not isinstance(dst_id, int):
            dst_id = int(dst_id)

        assert not self._G.has_edge(src_id, dst_id)
        self._G.add_edge(src_id, dst_id)

    def add_node(self, src_id, x=0.0, y=0.0, c=0.0):
        if not isinstance(src_id, int):
            src_id = int(src_id)

        assert not self._G.has_node(src_id)
        self._G.add_node(src_id, x=x, y=y, c=c)
    
    def del_edge(self, src_id, dst_id):
        if not isinstance(src_id, int):
            src_id = int(src_id)
        if not isinstance(dst_id, int):
            dst_id = int(dst_id)

        assert self._G.has_edge(src_id, dst_id)
        self._G.remove_edge(src_id, dst_id)

    def del_node(self, src_id):
        if not isinstance(src_id, int):
            src_id = int(src_id)

        assert self._G.has_node(src_id)
        self._G.remove_node(src_id)
    
    def get_num_edges(self):
        # Get the number of edges in the graph
        return self._G.number_of_edges()

    def get_num_nodes(self):
        # Get the number of nodes in the graph
        return self._G.number_of_nodes()

    def get_page_ranks(self):
        return nx.pagerank(self._G, tol=1e-4)
    
    def get_shortest_path(self, src_id, dst_id):
        try:
            path = nx.shortest_path(self._G, src_id, dst_id)
        except nx.exception.NetworkXNoPath as e:
            #print(e)
            path = []
        
        return path
        #return snap.GetShortPath(self._G, int(src_id), int(dst_id))
    
    def degree(self, node_ids: list):
        if isinstance(self._G, nx.DiGraph):
            # Get in and out degrees
            out_degrees = [d[1] for d in self._G.out_degree(node_ids)]
            in_degrees = [d[1] for d in self._G.in_degree(node_ids)]
            return out_degrees, in_degrees 
        else:
            degrees = [d[1] for d in self._G.degree(node_ids)]
            
            return degrees        
    
    @property
    def num_nodes(self):
        return self._G.number_of_nodes()
    
    def get_neighbors(self, node):
        return self._G.neighbors(node)
    
    def sample_edges(self, size: int) -> list:
        """Sample edges from the graph."
        
        Args:
            size: number of samples.
        """
        return random.sample(list(self._G.edges), size)

    def sample_nodes(self, size: int) -> list:
        """Sample nodes from the graph."
        
        Args:
            size: number of samples.
        """
        return random.sample(list(self._G.nodes), size)
        #return list(self._G.nodes)

    def copy(self):
        return Graph(self.args)

    def get_node_ids(self) -> list:
        # node_ids = []
        return list(self._G.nodes())
        # for node in self._G:
        #     node_ids.append(node)
        # return node_ids
    
    def partition(self):
        """Partition the graph

        Returns:
            Tuple of edgecuts and partition of nodes.
        """
        return nxmetis.partition(self._G.to_undirected(), self.args.num_parts)
    
    def get_edges(self, node_ids):
        """Get edges."""
        return list(self._G.edges(node_ids))
    
    def has_edge(self, src_id, dst_id):
        return self._G.has_edge(src_id, dst_id)
    
    def draw(self, node_colors=None, with_labels=True):
        nx.draw(self._G, node_color=node_colors, with_labels=with_labels)
        plt.show()

    def louvain(self, should_plot=False):
        partition = community_louvain.best_partition(self._G, randomize=False)
        if should_plot:
            pos = nx.spring_layout(self._G)
            # color the nodes according to their partition
            cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
            nx.draw_networkx_nodes(self._G, pos, partition.keys(), node_size=40,
                           cmap=cmap, node_color=list(partition.values()))
            nx.draw_networkx_edges(self._G, pos, alpha=0.5)
            plt.show()
        return partition
    
    def modularity_communities(self):
        """Find communities in graph using Clauset-Newman-Moore 
        greedy modularity maximization."""
        partition = list(greedy_modularity_communities(self._G))
        parts = [0] * self.get_num_nodes()
        for i, part in enumerate(partition):
            for node in list(part):
                parts[node] = i
        return parts

    def get_G(self):
        """Get the underlying networkx graph."""
        return self._G
    
    def replace_G(self, G):
        """Replace underlying graph with a new graph."""
        self._G = G
        self._relabel_nodes()
    
    def write_edge_list(self, edge_filename):
        with open(edge_filename, "w") as f:
            edges = list(self._G.edges())
            for i, edge in enumerate(edges):
                line = f"{edge[0] - 1} {edge[1] - 1}"
                if i + 1 < len(edges):
                    line += "\n"
                f.write(line)

    def write_tree(self, tree_filename):
        def sort_fn(e):
            return e[1]

        with open(tree_filename, "w") as f:
            edges = list(self._G.edges())
            edges.sort(key=sort_fn)
            for i, edge in enumerate(edges):
                line = f"{edge[0] - 1},{self._G.nodes[edge[0]]['x']},{self._G.nodes[edge[0]]['y']}|{edge[1] - 1},{self._G.nodes[edge[1]]['x']},{self._G.nodes[edge[1]]['y']}"
                if i + 1 < len(edges):
                    line += "\n"
                f.write(line)

    def single_source_shortest_path(self, node_id: int, cutoff=50):
        return nx.single_source_shortest_path_length(self._G, node_id, cutoff=cutoff)

    def nx_viz(self):
        pos = {}
        for n in self._G.edges():
            pos[n[0]] = np.array([self._G.nodes[n[0]]['x'], self._G.nodes[n[0]]['y']])
            pos[n[1]] = np.array([self._G.nodes[n[1]]['x'], self._G.nodes[n[1]]['y']])

        for n in self._G.nodes():
            if n not in pos.keys():
                pos[n] = np.array([0.0, 0.0])
        
        nx.draw(self._G, pos, with_labels=False, node_size=5)
        plt.show()

    def parse_tree(self, tree_file):
        G = nx.Graph()
        G.add_node(0, x=0.0, y=0.0, c=1.0)
        with open(tree_file) as tree:
            for line in tree:
                ends = line.split('|')
                (i, x, y, c) = tuple(map(float, ends[0].split(',')))
                G.add_node(int(i), x=x, y=y, c=c)
                a = int(ends[0].split(',')[0])
                b = int(ends[1].split(',')[0])
                G.add_edge(a, b)

        #print(G.nodes(data=True))
        return G
