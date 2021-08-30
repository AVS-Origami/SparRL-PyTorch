import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json

from model import SparRLNet
from conf import *
from agents.agent import Agent, State
from agents.expert_agent import ExpertAgent


class RLAgent(Agent):
    def __init__(self, args, memory, num_nodes: int, expert_agent: ExpertAgent = None):
        super().__init__(args)
        self._memory = memory
        self._expert_agent = expert_agent

        # Number of elapsed parameter update steps
        self._train_dict = {
            "update_step" : 0,
            "episodes" : 0,
            "final_rewards" : [],
            "mse_losses" : []
        }

        # self._update_step = 0

        # # Save the final rewards
        # self._final_rewards = []
        
        # Create SparRL networks
        self._sparrl_net = SparRLNet(self.args, num_nodes).to(device)
        self._sparrl_net_tgt = SparRLNet(self.args, num_nodes).to(device)

        # Create optimizer and LR scheduler to decay LR
        self._optim = optim.Adam(self._sparrl_net.parameters(), self.args.lr)
        self._lr_scheduler = optim.lr_scheduler.MultiplicativeLR(
            self._optim,
            lr_lambda=lambda e: self.args.lr_gamma)
        
        self._gam_arr = self._create_gamma_arr()
        self._model_file = os.path.join(self.args.save_dir, "sparrl_net")
        self._train_dict_file = os.path.join(self.args.save_dir, "train_dict.json")

        if self.args.load:
            self.load()


    @property
    def epsilon_threshold(self):
        """Return the current epsilon value used for epsilon-greedy exploration."""
        # Adjust for number of expert episodes that have elapsed
        cur_step = max(self._train_dict["update_step"] - self.args.expert_episodes * self.args.train_iter, 0)
        return self.args.min_epsilon + (self.args.epsilon - self.args.min_epsilon) * \
            math.exp(-1. * cur_step / self.args.epsilon_decay)
    
    @property
    def is_ready_to_train(self) -> bool:
        """Check for if the model is ready to start training."""
        return self._memory.cur_cap() >= self.args.batch_size

    def reset(self):
        # Update number of elapsed episdoes
        self._train_dict["episodes"] += 1

        # Add episode experiences
        self._add_stored_exs()

    def save(self, final_reward):
        """Save the models."""
        self._train_dict["final_rewards"].append(final_reward)

        model_dict = {
            "sparrl_net" : self._sparrl_net.state_dict(),
            "sparrl_net_tgt" : self._sparrl_net_tgt.state_dict(),
            "optimizer" : self._optim.state_dict(),
            "lr_scheduler" : self._lr_scheduler.state_dict()
        }

        torch.save(model_dict, self._model_file)

        with open(self._train_dict_file, "w") as f:
            json.dump(self._train_dict, f)
        

        # Save the experience replay
        self._memory.save()

    def load(self):
        """Load the models."""
        model_dict = torch.load(self._model_file, map_location=device)
        
        self._sparrl_net.load_state_dict(model_dict["sparrl_net"])
        self._sparrl_net_tgt.load_state_dict(model_dict["sparrl_net_tgt"])
        self._optim.load_state_dict(model_dict["optimizer"])
        self._lr_scheduler.load_state_dict(model_dict["lr_scheduler"])

        with open(self._train_dict_file) as f:
            self._train_dict = json.load(f)

    def add_ex(self, ex):
        """Add an time step of experience."""
        if not self.args.eval and self._memory:
            ex.is_expert = self._should_add_expert_ex
            self._exp_buffer.append(ex)

    def _update_target(self):
        """Perform soft update of the target policy."""
        for tgt_sparrl_param, sparrl_param in zip(self._sparrl_net_tgt.parameters(), self._sparrl_net.parameters()):
            tgt_sparrl_param.data.copy_(
                self.args.tgt_tau * sparrl_param.data + (1.0-self.args.tgt_tau) * tgt_sparrl_param.data)

    def _create_gamma_arr(self):
        """Create a gamma tensor for multi-step DQN."""
        gam_arr = torch.ones(self.args.dqn_steps)
        for i in range(1, self.args.dqn_steps):
            gam_arr[i] = self.args.gamma * gam_arr[i-1] 
        return gam_arr

    def _sample_action(self, q_vals: torch.Tensor, argmax=False) -> int:
        """Sample an action from the given Q-values."""
        if not argmax and self.epsilon_threshold >= np.random.rand():
            # Sample a random action
            action = np.random.randint(q_vals.shape[0])
        else:
            with torch.no_grad():
                # Get action with maximum Q-value
                action = q_vals.argmax()

        return int(action)



    def _add_stored_exs(self):
        """Add experiences stored in temporary buffer into replay memory.
        
        This method makes the assumption that self._exp_buffer only contains experiences
        from the same episode.
        """
        rewards = torch.zeros(self.args.dqn_steps)
        for i in reversed(range(len(self._exp_buffer))):
            rewards[0] = self._exp_buffer[i].reward
            cur_gamma = self.args.gamma

            # Update the experience reward to be the n-step return
            if i + self.args.dqn_steps < len(self._exp_buffer):
                self._exp_buffer[i].reward = rewards.dot(self._gam_arr)
                self._exp_buffer[i].next_state = self._exp_buffer[i + self.args.dqn_steps].state
                cur_gamma = cur_gamma ** self.args.dqn_steps

            # Update gamma based on n-step return
            self._exp_buffer[i].gamma = cur_gamma

            with torch.no_grad():
                # Get the Q-value for the state, action pair
                q_val = self._sparrl_net(self._exp_buffer[i].state)[self._exp_buffer[i].action]
                
                if self._exp_buffer[i].next_state is not None:
                    # Get the valid action for next state that maximizes the q-value
                    valid_actions = self._get_valid_edges(self._exp_buffer[i].next_state.subgraph[0])
                    q_next = self._sparrl_net(self._exp_buffer[i].next_state)

                    next_action = self._sample_action(q_next[valid_actions], argmax=True)
                    next_action = valid_actions[next_action] 

                    # Compute TD target based on target function q-value for next state
                    q_next_target = self._sparrl_net_tgt(self._exp_buffer[i].next_state)[next_action]
                    td_target = self._exp_buffer[i].reward + self._exp_buffer[i].gamma *  q_next_target

                else:
                    td_target = self._exp_buffer[i].reward

            td_error = td_target - q_val
            self._memory.add(self._exp_buffer[i], td_error)      

            # Shift the rewards down
            rewards = rewards.roll(1)

        # Clear the experiences from the experince buffer
        self._exp_buffer.clear()


    def _unwrap_exs(self, exs: list):
        """Extract the states, actions and rewards from the experiences."""
        subgraphs = torch.zeros(self.args.batch_size, self.args.subgraph_len*2, device=device, dtype=torch.int32)
        global_stats = torch.zeros(self.args.batch_size, 1, NUM_GLOBAL_STATS, device=device)
        local_stats = torch.zeros(self.args.batch_size, self.args.subgraph_len*2, NUM_LOCAL_STATS, device=device)
        subgraph_mask = torch.zeros(self.args.batch_size, self.args.subgraph_len, device=device)

        actions = []
        rewards = torch.zeros(self.args.batch_size, device=device)
        next_subgraphs = torch.zeros(self.args.batch_size, self.args.subgraph_len*2, device=device, dtype=torch.int32)
        next_global_stats = torch.zeros(self.args.batch_size, 1, NUM_GLOBAL_STATS, device=device)
        next_local_stats = torch.zeros(self.args.batch_size, self.args.subgraph_len*2, NUM_LOCAL_STATS, device=device)
        next_subgraph_mask = torch.zeros(self.args.batch_size, self.args.subgraph_len, device=device)
        
        next_state_mask = torch.zeros(self.args.batch_size, device=device)
        is_experts = torch.zeros(self.args.batch_size, dtype=torch.bool, device=device)
        gammas = torch.zeros(self.args.batch_size, device=device)

        # Unwrap the experiences
        for i, ex in enumerate(exs):
            # Create subgraph mask if edges less than subgraph length
            if ex.state.subgraph.shape[1]//2 < self.args.subgraph_len:
                # Set edges that are null to 1 to mask out
                subgraph_mask[i, ex.state.subgraph.shape[1]:] = 1

            local_stats[i, :ex.state.local_stats.shape[1]] = ex.state.local_stats
            subgraphs[i, :ex.state.subgraph.shape[1]], global_stats[i], local_stats[i, :ex.state.local_stats.shape[1]] = ex.state.subgraph, ex.state.global_stats, ex.state.local_stats
            
            actions.append(ex.action)
            rewards[i] = ex.reward
            is_experts[i] = ex.is_expert
            gammas[i] = ex.gamma
            if ex.next_state is not None:
                next_subgraphs[i, :ex.next_state.subgraph.shape[1]], next_global_stats[i], local_stats[i, :ex.next_state.local_stats.shape[1]] = ex.next_state.subgraph, ex.next_state.global_stats, ex.next_state.local_stats 
                next_state_mask[i] = 1

                # Create subgraph mask if edges less than subgraph length
                if ex.next_state.subgraph.shape[1]//2 < self.args.subgraph_len:
                    # Set edges that are null to 1 to mask out
                    next_subgraph_mask[i, ex.next_state.subgraph.shape[1]:] = 1

        states = State(subgraphs, global_stats, local_stats)
        

        # Get nonempty states
        nonzero_next_states = next_state_mask.nonzero().flatten()
        next_states = State(next_subgraphs[nonzero_next_states], next_global_stats[nonzero_next_states], next_local_stats[nonzero_next_states])
        # next_state_mask = next_state_mask[nonzero_next_states]
        next_subgraph_mask = next_subgraph_mask[nonzero_next_states]
        return states, subgraph_mask, actions, rewards, next_states, next_state_mask, next_subgraph_mask, is_experts, gammas

    def train(self) -> float:
        """Train the model over a sampled batch of experiences.
        
        Returns:
            the loss for the batch
        """

        is_ws, exs, indices = self._memory.sample(self.args.batch_size, self._train_dict["update_step"])
        td_targets = torch.zeros(self.args.batch_size, device=device)
        
        states, subgraph_mask, actions, rewards, next_states, next_state_mask, next_subgraph_mask, is_experts, gammas = self._unwrap_exs(exs)

        # Select the q-value for every state
        actions = torch.tensor(actions, dtype=torch.int64, device=device)

        q_vals_matrix = self._sparrl_net(states, subgraph_mask)

        q_vals = q_vals_matrix.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Run policy on next states
        q_next = self._sparrl_net(
            next_states,
            next_subgraph_mask).detach()


        q_next_target = self._sparrl_net_tgt(
            next_states,
            next_subgraph_mask).detach()

        # index used for getting the next nonempty next state
        q_next_idx = 0
        
        expert_margin_loss = 0

        # Number of terminated states elapsed thus far
        num_term_states = 0

        for i in range(self.args.batch_size):
            # Compute expert margin classification loss (i.e., imitation loss)
            if is_experts[i]:
                margin_mask = torch.ones(self.args.subgraph_len, device=device)
                # Mask out the expert action    
                margin_mask[actions[i]] = 0
                
                # Set to margin value
                margin_mask = margin_mask * self.args.expert_margin
                
                # Compute the expert imitation loss
                expert_margin_loss += torch.max(q_vals_matrix[i] + margin_mask) - q_vals[i]
            
            if not next_state_mask[i]:
                td_targets[i] = rewards[i]
                num_term_states += 1
            else:
                # Get the argmax next action for DQN
                valid_actions = self._get_valid_edges(next_states.subgraph[i - num_term_states])


                action = self._sample_action(
                    q_next[q_next_idx][valid_actions], True)
                action = int(valid_actions[action])

                # Set TD Target using the q-value of the target network
                # This is the Double-DQN target
                td_targets[i] = rewards[i] + gammas[i] * q_next_target[q_next_idx, action]
                q_next_idx += 1
 
        self._optim.zero_grad()

        # Compute L1 loss
        td_errors = td_targets  - q_vals
        loss = torch.mean(td_errors.abs()  *  is_ws)


        self._memory.update_priorities(indices, td_errors.detach().abs(), is_experts)
        loss = loss
        print("loss", loss)
        print("L2 LOSS", torch.mean(td_errors ** 2  *  is_ws))
        print("expert_margin_loss", expert_margin_loss* self.args.expert_lam)
        
        loss += expert_margin_loss * self.args.expert_lam
        loss.backward()
        
        # Clip gradient
        nn.utils.clip_grad.clip_grad_norm_(
            self._sparrl_net.parameters(),
            self.args.max_grad_norm)


        # Train model
        self._optim.step()

        # Check if using decay and min lr not reached
        if not self.args.no_lr_decay and self._optim.param_groups[0]["lr"] > self.args.min_lr:
            # If so, decay learning rate
            self._lr_scheduler.step()
        else:
            self._optim.param_groups[0]["lr"] = self.args.min_lr

        # Update train info
        self._train_dict["update_step"] += 1
        self._train_dict["mse_losses"].append(float(loss.detach()))

        # Update the DQN target parameters
        self._update_target()
        
        # Print out q_values and td_targets for debugging/progress updates
        if (self._train_dict["update_step"] + 1) % 8 == 0:
            print("self.epsilon_threshold", self.epsilon_threshold)
            print("q_values", q_vals)
            print("td_targets", td_targets)
            print("rewards", rewards)

        return float(loss.detach())


    def __call__(self, state, mask=None) -> int:
        """Make a sparsification decision based on the state.

        Returns:
            an edge index.
        """

        # Set for when experience is added
        self._should_add_expert_ex = self._train_dict["episodes"] < self.args.expert_episodes

        if self._should_add_expert_ex:
            # Run through expert policy
            edge_idx = self._expert_agent(state)
        else:
            # Get the q-values for the state
            q_vals = self._sparrl_net(state, mask)

            # Sample an action (i.e., edge to prune)
            edge_idx = self._sample_action(q_vals, self.args.eval)

        return edge_idx
        