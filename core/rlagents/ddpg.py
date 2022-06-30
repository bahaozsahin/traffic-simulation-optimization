import numpy as np

from core.rlagents.rl_agent import RLAgent

class DDPGAgent(RLAgent):
    def __init__(self, networks, epsilon, exp_replay, n_actions, n_steps, n_batch, n_exp_replay, gamma, rl_stats, mode, updates):
        super().__init__(networks, epsilon, exp_replay, n_actions, n_steps, n_batch, n_exp_replay, gamma, rl_stats, mode, updates) 
        
    def get_action(self, state):

        if self.mode == 'train':
            self.retrieve_weights('online')
        a = self.networks['actor'].forward(state[np.newaxis,...], 'online')
        if len(self.exp_replay) < self.n_exp_replay:
            self.epsilon = 1.0
        a += np.random.uniform(-self.epsilon, self.epsilon, size=self.n_actions)

        a[a>1.0] = 1.0
        a[a<-1.0] = -1.0

        return a[0]

    def train_batch(self, update_freq):
        sample_batch = self.sample_replay()

        states, actions, targets = self.process_batch(sample_batch)
        targets = np.expand_dims(targets, -1)

        self.networks['critic'].backward( states, actions, targets )

        actions = self.networks['actor'].forward(states, 'online')
        grads = self.networks['critic'].gradients(states, actions)
        self.networks['actor'].backward(states, grads[0])
        self.rl_stats['updates'] += 1
        self.rl_stats['n_exp'] -= 1
        self.send_weights('online')

        if self.rl_stats['updates'] % update_freq == 0: 
            self.networks['actor'].transfer_weights()    
            self.networks['critic'].transfer_weights()

    def process_batch(self, sample_batch):
        next_states, terminals = [], []
        for trajectory in sample_batch:
            next_states.append(trajectory[-1]['next_s'])                       
            terminals.append(trajectory[-1]['terminal'])                       

        R = self.next_state_bootstrap(np.stack(next_states), np.stack(terminals))
        max_r = self.rl_stats['max_r']
        targets, states, actions = [], [], []

        for trajectory, r in zip(sample_batch, R):
            rewards = []
            for exp in trajectory:
                states.append(exp['s'])
                actions.append(exp['a'])
                rewards.append(exp['r']/max_r)
            targets.extend( self.process_trajectory( rewards, r ) )
                                                                                         
        if self.n_steps > 1:
            idx = np.random.randint(0, len(targets), size = self.n_batch) 
            _states, _actions, _targets = [], [], []
            for i in idx:
                _states.append(states[i])
                _targets.append(targets[i])
                _actions.append(actions[i])
            return np.stack(_states), np.stack(_actions), np.stack(_targets)
        else:
            return np.stack(states), np.stack(actions), np.stack(targets)

    def next_state_bootstrap(self, next_states, terminals):
        bootstrap_actions = self.networks['actor'].forward(next_states, 'target') 
        R = self.networks['critic'].forward(next_states, bootstrap_actions, 'target')

        return [ 0.0 if t is True else r[0] for t, r in zip(terminals, R)] 

    def process_trajectory(self, rewards, R):
        return self.compute_targets(rewards, R)

    def send_weights(self, nettype):
        self.rl_stats[nettype] = self.networks['actor'].get_weights(nettype)

    def retrieve_weights(self, nettype):
        self.networks['actor'].set_weights(self.rl_stats[nettype], nettype)
