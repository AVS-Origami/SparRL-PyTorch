import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from transformer import Encoder
from conf import *
from agents.storage import State

class NoisyLinear(nn.Module):
    def __init__(self, args, in_features, out_features):
        super().__init__()
        self.args = args
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        if not self.args.load:
            self.reset_parameters()

        self.reset_noise()
    
    def forward(self, x):
        if self.training: 
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.args.noise_std / math.sqrt(self.weight_sigma.size(1)))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.args.noise_std / math.sqrt(self.bias_sigma.size(0)))
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x
    
    def sigma_mean_abs(self):
        return self.weight_sigma.abs().mean()

class GAT(nn.Module):
    def __init__(self, args, num_nodes):
        super().__init__()
        self.args = args
        self.node_embs = nn.Embedding(num_nodes + 1, self.args.hidden_size) 
        self.a = nn.Linear(self.args.hidden_size * 2, 1)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, node_ids: torch.Tensor, neighs: torch.Tensor, mask: torch.Tensor):
        # Get initial embeddings
        node_ids.to(device)
        neighs.to(device)
        mask.to(device)
        src_node_embs = self.node_embs(node_ids.unsqueeze(2))
        
        neigh_embs = self.node_embs(neighs)
        
        # Add the source node to perform attention over it as well
        neigh_embs = torch.cat((src_node_embs, neigh_embs), 2)
        
        embs = torch.cat((
            src_node_embs.repeat(1, 1, self.args.max_neighbors+1, 1), neigh_embs), -1)
        
        att_scores = self.act(self.a(embs))
        #print("att_scores", att_scores)
        att_scores[:, :, 1:] += mask * -1e9

        # print("neighs", neighs)
        # print("mask", mask)
        att_weights = F.softmax(att_scores, dim=2)
        # Mask out invalid neighbors
        #print("torch.matmul(att_weights.transpose(3,2), neigh_embs).shape", torch.matmul(att_weights.transpose(3,2), neigh_embs).shape)
        
        node_embs = torch.matmul(att_weights.transpose(3,2), neigh_embs).view(node_ids.shape[0], node_ids.shape[1], self.args.hidden_size)
        
        return node_embs



        

        # Get a nodes adjacent edges
        # if node_id > 0:
        #     neighs.append(node_id)
        #     neighs = torch.tensor(neighs, device=device, dtype=torch.int32)            
        #     if len(neighs) > 1:
        #         # Perform attention over neighbors
        #         neigh_embs = self.node_embs(neighs)
        #         embs = torch.cat((neigh_embs, self.node_embs(node_id).repeat(len(neighs), 1)), -1)
        #         att_scores = self.a(embs)
        #         att_weights = F.softmax(self.act(att_scores), dim=0)
        #         node_emb = torch.matmul(att_weights.t(), neigh_embs)
        #     else:
        #         # Use original node embeddings
        #         node_emb = self.node_embs(node_id)
        # else:
        #     node_emb = self.node_embs(node_id)

        # return node_emb
        
        

class NodeEncoder(nn.Module):
    """Create node embedding using local statistics."""
    def __init__(self, args, num_nodes: int):
        super().__init__()
        self.args = args
        self.num_nodes = num_nodes
        self.gat = GAT(self.args, self.num_nodes)
        # self.node_embs = nn.Embedding(self.num_nodes+1, self.args.hidden_size) 

        # if self.args.node_embs:
        #     self.load_pretrained_embs()
        
        #self.node_fc = nn.Linear(self.args.hidden_size, self.args.hidden_size)
        # self.node_fc = nn.BatchNorm1d(self.args.hidden_size)
        self.fc_1 = nn.Linear(self.args.hidden_size + NUM_LOCAL_STATS + 1, self.args.hidden_size)
        self.fc_2 = nn.Linear(self.args.hidden_size, self.args.hidden_size)
        self.act_1 = nn.LeakyReLU(0.2)
        self.act_2 = nn.LeakyReLU(0.2)

        #self.norm_1 = nn.LayerNorm(self.args.hidden_size, eps=1e-6)
        # self.norm_2 = nn.LayerNorm(self.args.hidden_size, eps=1e-10)
        self.dropout_1 = nn.Dropout(self.args.drop_rate)
        # self.dropout_2 = nn.Dropout(self.args.drop_rate)
        
    def load_pretrained_embs(self):
        weights_dict = torch.load(self.args.node_embs)
        # Add pad embedding
        pretrained_node_embs = torch.cat((self.node_embs(torch.tensor([0])), weights_dict["node_embs"])) 
        self.node_embs = self.node_embs.from_pretrained(pretrained_node_embs, freeze=True)

    def forward(self, state):
        state = State(
            state[0],
            state[1],
            state[2],
            state[3],
            state[4],
            state[5]
        )
            
        # Get initial node embeddings
        subgraph = state.subgraph
        local_stats = torch.cat((state.local_stats, state.global_stats.repeat(1, state.local_stats.shape[1], 1)), -1)
        batch_size = subgraph.shape[0]
        
        
        node_embs = self.gat(subgraph, state.neighs, state.mask)
        
        #node_embs = self.norm_1(node_embs.reshape(-1, self.args.hidden_size))
        #node_embs = node_embs.reshape(batch_size, -1, self.args.hidden_size)
        #node_embs = self.norm_1(node_embs)
        #node_embs = self.dropout_1(node_embs)
        # node_embs = self.norm_1(node_embs)
        #node_embs = (node_embs)
        
        # Combine node embeddings with current local statistics
        node_embs = self.act_1(self.fc_1(
            torch.cat((node_embs, local_stats), -1)))
        #node_embs = self.norm_1(node_embs)
        #node_embs = F.gelu(node_embs)

        # Create final node embeddings
        node_embs = self.act_2(self.fc_2(node_embs))
        # node_embs = self.norm_1(node_embs)
        #node_embs = self.dropout_1(node_embs)
        #node_embs = F.gelu(node_embs)
        
        return node_embs 

class EdgeEncoder(nn.Module):
    """Map node embedding to edge embedding."""
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Used to combine the embeddings
        # self.edge_conv1d = nn.Conv1d(
        #     self.args.hidden_size,
        #     self.args.hidden_size,
        #     kernel_size=2,
        #     stride=2)
        self.edge_fc_1 = nn.Linear(self.args.hidden_size * 2, self.args.hidden_size)
        self.edge_fc_2 = nn.Linear(self.args.hidden_size, self.args.hidden_size)
        self.act_1 = nn.LeakyReLU(0.2)
        self.act_2 = nn.LeakyReLU(0.2)
        #self.norm_1 = nn.LayerNorm(self.args.hidden_size, eps=1e-10)
        #self.dropout_1 = nn.Dropout(self.args.drop_rate)

        # self.norm_1 = nn.LayerNorm(self.args.hidden_size, eps=1e-6)
        # # self.norm_2 = nn.LayerNorm(self.args.hidden_size, eps=1e-10)
        # self.dropout_1 = nn.Dropout(self.args.drop_rate)
        # self.dropout_2 = nn.Dropout(self.args.drop_rate)

        # Used produce to produce the final edge embedding
        #self.edge_fc_3 = nn.Linear(self.args.hidden_size, self.args.hidden_size)

    def forward(self, node_embs: torch.Tensor):
        # Combine the node embeddings to create edge embeddings
        #print("node_embs", node_embs)
        node_embs = node_embs.reshape(node_embs.shape[0], -1, self.args.hidden_size * 2)
        #print("node_embs", node_embs)
        #print("node_embs.count_nonzero()", node_embs.count_nonzero(), node_embs.shape)
        edge_embs = self.act_1(self.edge_fc_1(node_embs))
        edge_embs = self.act_2(self.edge_fc_2(edge_embs))

        #print("edge_embs.count_nonzero()", edge_embs.count_nonzero())
        # edge_embs = self.norm_1(edge_embs.transpose(1,2))
        # edge_embs = self.dropout_1(edge_embs)
        
        # Create the final edge embeddings
        #edge_embs = F.gelu(self.edge_fc_3(edge_embs))
        # edge_embs = self.norm_1(edge_embs)
        #edge_embs = self.dropout_1(edge_embs)
        # edge_embs = self.edge_fc(edge_embs)
        # edge_embs = self.norm_2(edge_embs)
        # edge_embs = self.dropout_2(edge_embs)

        return edge_embs

        
class SparRLNet(nn.Module):
    def __init__(self, args, num_nodes: int):
        super().__init__()
        self.args = args
        self.num_nodes = num_nodes
        
        
        self.node_enc = NodeEncoder(self.args, self.num_nodes)
        self.edge_enc = EdgeEncoder(self.args)

        self.q_fc_3 = nn.Linear(self.args.hidden_size, 1)


    def reset_noise(self):
        pass
        # self.v_fc_1.reset_noise()
        # self.v_fc_2.reset_noise()

        # self.adv_fc_1.reset_noise()
        # self.adv_fc_2.reset_noise()
        #self.q_fc_1.reset_noise()
        #self.q_fc_2.reset_noise()

    def forward(self, subgraph, global_stats, local_stats, mask, neighs, childs) -> torch.Tensor:
        # if isinstance(state, State):
        #     state = [state.subgraph, state.global_stats, state.local_stats, state.mask, state.neighs]

        state = [subgraph.to(device), global_stats.to(device), local_stats.to(device), mask.to(device), neighs.to(device), childs.to(device)]

        batch_size = subgraph.shape[0]

        # Create node embedding
        node_embs = self.node_enc(state)

        # Create edge embeddings
        #embs = self.edge_enc(node_embs)

        q_vals = self.q_fc_3(node_embs)
        #print("q_vals", q_vals)
        if batch_size == 1:
            return q_vals.view(-1)
        else:
            # print(state)
            # print("q_vals", q_vals)
            # print("q_vals.shape", q_vals.shape)
            # print("embs.shape", embs.shape, "\n\n")
            #print("q_vals.view(q_vals.shape[0], q_vals.shape[1])", q_vals.view(q_vals.shape[0], q_vals.shape[1]))
            #print("q_vals", q_vals)
            return q_vals.view(q_vals.shape[0], q_vals.shape[1])