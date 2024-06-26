import torch
from dataclasses import dataclass
from conf import *

@dataclass
class State:
    subgraph: torch.Tensor
    global_stats: torch.Tensor
    local_stats: torch.Tensor
    mask: torch.Tensor
    neighs: torch.Tensor
    childs: torch.Tensor

    def unpack(self):
        self.subgraph.to(device)
        self.global_stats.to(device)
        self.local_stats.to(device)
        self.mask.to(device)
        self.neighs.to(device)
        self.childs.to(device)

        return [
            self.subgraph,
            self.global_stats,
            self.local_stats,
            self.mask,
            self.neighs,
            self.childs
        ]


@dataclass
class Experience:
    state: State
    next_state: State
    action: int
    reward: float
    is_expert: bool = False
    gamma: float = 0.99


@dataclass
class StateMessage:
    """Message send between processes containing the state or previous experiences."""
    state: State = None
    mask: torch.Tensor = None
    ex_buffer: list = None


@dataclass
class ActionMessage:
    """Contains action given to child process."""
    action: int