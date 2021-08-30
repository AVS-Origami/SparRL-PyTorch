import torch
from dataclasses import dataclass


@dataclass
class State:
    subgraph: torch.Tensor
    global_stats: torch.Tensor
    local_stats: torch.Tensor


@dataclass
class Experience:
    state: State
    next_state: State
    action: int
    reward: float
    is_expert: bool = False
    gamma: float = 0.99


class Agent:
    def __init__(self, args, is_expert: bool =False):
        self.args = args
        # Temporary buffer for experiences in an episode
        self._exp_buffer = []
        self.is_expert = is_expert

    def _get_valid_edges(self, subgraph: torch.Tensor):
        """Get valid edges that are in subgraph."""
        # Put edges in same rows
        temp_subgraph = subgraph.reshape(subgraph.shape[0]//2, 2)

        # Get nonzero edges
        return temp_subgraph.sum(axis=-1).nonzero().flatten()
    
    def add_ex(self, ex):
        """Add an time step of experience."""
        pass

    def reset(self):
        """Reset after an episode."""
        pass

    def __call__(self, state) -> int:
        """Make a sparsification decision based on the state.

        Returns:
            an edge index.
        """
        pass