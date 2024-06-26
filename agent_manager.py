import torch
# from multiprocessing import Process, Queue, set_start_method
import torch.multiprocessing as mp
from dataclasses import dataclass
import json,os


from graph import Graph
from agents.agent_worker import AgentWorker
from environment import Environment
from replay_memory import PrioritizedExReplay
from agents.rl_agent import RLAgent
from agents.storage import ActionMessage, State
from conf import *
from expert_control import ExpertControl


mp.set_start_method("spawn", force=True)
mp.set_sharing_strategy('file_system')

@dataclass
class ChildProcess:
    process: mp.Process
    env: Environment
    state_queue: mp.Queue
    action_queue: mp.Queue

class AgentManager:
    """Manager multiple agent workers for multiprocess training."""
    def __init__(self, args, rl_agent: RLAgent, results_man = None):
        self.args = args
        self._rl_agent = rl_agent
        if self.args.expert_episodes > 0:
            self._expert_control = ExpertControl(self.args)

        self.org_T_max = self.args.T_max
        self.args.T_max = mp.Value("i", self.args.T_max)
        self._init_workers()

        self._results_man = results_man
        
        self._man_file = os.path.join(self.args.save_dir, "agent_man.json")

        if self.args.load:
            self.load()
        else:
            self._agent_man_dict = {
                "ec_triggered" : False,
                "eval_rewards" : []    
            }        

    def save(self):
        state = self._results_man.env.create_state(self.args.subgraph_len, 250, 27)
        self._rl_agent.save(state)
        with open(self._man_file, "w") as f:
            json.dump(self._agent_man_dict, f)
        
    def load(self):
        with open(self._man_file) as f:
            self._agent_man_dict = json.load(f)
        
        if self._agent_man_dict["ec_triggered"]:
            self.args.expert_lam = 0.0
            self.args.expert_epsilon = -1.0

    def _init_workers(self):
        """Initialize the child processes with duplicate environments."""
        self._child_processes = []
        print("Creating workers.")
        for i in range(self.args.workers):
            # Stores the states from the AgentWorker
            state_queue = mp.Queue(1)

            # Stores the actions to the AgentWorker
            action_queue = mp.Queue(1)

            # Create worker to communcate to global model
            agent_worker = AgentWorker(self.args, state_queue, action_queue)

            # Create duplicate environment
            graph = Graph(self.args)
            env = Environment(self.args, agent_worker, graph)
            
            # Create the child process
            agent_process = mp.Process(target=env.run)
            agent_process.start()
            self._child_processes.append(
                ChildProcess(agent_process, env, state_queue, action_queue))

    def _terminate_episode(self, exs: list):
        """Add experiences from episode to replay buffer."""
        rewards = []
        for ex in exs:
            # Add timestep reward
            rewards.append(ex.reward)

            # Move to CUDA device
            ex.state.subgraph = ex.state.subgraph.to(device)
            ex.state.local_stats = ex.state.local_stats.to(device)
            ex.state.global_stats = ex.state.global_stats.to(device)
            ex.state.mask = ex.state.mask.to(device)
            ex.state.neighs = ex.state.neighs.to(device)
            if ex.next_state:
                ex.next_state.subgraph = ex.next_state.subgraph.to(device)
                ex.next_state.local_stats = ex.next_state.local_stats.to(device)
                ex.next_state.global_stats = ex.next_state.global_stats.to(device)
                ex.next_state.mask = ex.next_state.mask.to(device)
                ex.next_state.neighs = ex.next_state.neighs.to(device)
            
            self._rl_agent.add_ex(ex)

        # Get the average reward
        avg_reward = sum(rewards) / len(rewards)

        # Reset for next episode
        self._rl_agent.reset(avg_reward)

    def _aggregate_states(self, states: list) -> State:
        """Aggregate states from several workers.
        
        Returns:
            the aggregated state.
        """
        batch_size = len(states)
        subgraphs = torch.zeros(batch_size, self.args.subgraph_len * 2, device=device, dtype=torch.int32)
        global_stats = torch.zeros(batch_size, 1, NUM_GLOBAL_STATS, device=device)
        local_stats = torch.zeros(batch_size, self.args.subgraph_len * 2, NUM_LOCAL_STATS, device=device)
        neighs = torch.zeros(batch_size, self.args.subgraph_len*2, self.args.max_neighbors, device=device, dtype=torch.int32)
        mask = torch.ones(batch_size, self.args.subgraph_len*2, self.args.max_neighbors, 1, device=device)
        childs = torch.zeros(batch_size, self.args.subgraph_len*2)
        for i, state in enumerate(states):
            subgraphs[i, :state.subgraph.shape[1]] = state.subgraph
            global_stats[i] = state.global_stats
            local_stats[i, :state.local_stats.shape[1]] = state.local_stats
            mask[i] = state.mask
            neighs[i] = state.neighs
            childs[i] = state.childs
        
        
        return State(subgraphs, global_stats, local_stats, mask, neighs, childs)

    def run(self):
        # Run several batch of episodes
        for e_i in range(self.args.episodes // self.args.workers):
            # Keep up with processes that have episodes still running
            nonterm_p_ids = list(range(len(self._child_processes)))
            
            # Run until all episodes are terminated
            t = 0
            while len(nonterm_p_ids) > 0:
                states = []

                # Get states from all agents
                for p_id in nonterm_p_ids.copy():

                    # Get the current state from this process
                    state_msg = self._child_processes[p_id].state_queue.get()
                    
                    # Check if episode has terminated
                    if state_msg.state is None:
                        # Remove the process ID
                        self._child_processes[p_id].action_queue.put(ActionMessage(-1))
                        nonterm_p_ids.remove(p_id)
                        self._terminate_episode(state_msg.ex_buffer)
                    else:
                        states.append(state_msg.state)

                # Predict on states
                if len(nonterm_p_ids) >= 1:
                    states = self._aggregate_states(states)
                    
                    # Get actions
                    with torch.no_grad():
                        self._rl_agent.reset_noise()
                        action = self._rl_agent(states)
                    
                    if len(nonterm_p_ids) == 1:
                        action = [action]

                    # Pass actions to child processes
                    for i, p_id in enumerate(nonterm_p_ids):
                        self._child_processes[p_id].action_queue.put(
                            ActionMessage(action[i]))
                                # if self._rl_agent.is_ready_to_train:
                # Train the model
                if self._rl_agent.is_ready_to_train:
                    #print("Training")
                    self._rl_agent.reset_noise()
                    self._rl_agent.train()
                    #print("Done Training")
                
                t += 1


            if self.args.T_eval > 0 and (e_i + 1) % self.args.eval_iter == 0:
                self.args.eval = True
                self._results_man.env.reset()
                eval_reward = self._results_man.run_rl_eval(self._rl_agent, self.args.T_eval)
                self.args.eval = False
                print("EVAL REWARD: ", eval_reward)
                self._agent_man_dict["eval_rewards"].append(eval_reward)
                
            # if self._rl_agent.is_ready_to_train:
            #     # Train the model
            #     print("Training")
            #     for _ in range(self.args.train_iter):
            #         self._rl_agent.reset_noise()    
            #         self._rl_agent.train()
                
            #     print("Done Training")

            # Save the models
            
            
            

            # Check if expert control should be triggered
            # if not self._agent_man_dict["ec_triggered"] and \
            #     self.args.expert_episodes > 0 and \
            #     not self.args.no_ec:
            #     self._rl_agent._sparrl_net.eval()    
            #     if self._expert_control._test_mean_reward(self._rl_agent):
            #         self._agent_man_dict["ec_triggered"] = True
            #         self.args.expert_lam = 0.0
            #         self.args.expert_epsilon = -1.0

            #     self._rl_agent._sparrl_net.train()
            
            # print("Should Trigger EC:", self._agent_man_dict["ec_triggered"])
            
            if (e_i + 1) % self.args.save_iter == 0:
                print("SAVING")
                self.save()
                print("DONE SAVING")

        print("SAVING")
        self.save()
        print("DONE SAVING")