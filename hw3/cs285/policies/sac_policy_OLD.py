from cs285.policies.MLP_policy import MLPPolicy
import torch
import numpy as np
from cs285.infrastructure import sac_utils
from cs285.infrastructure import pytorch_util as ptu
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.distributions import Normal
import itertools

class MLPPolicySAC(MLPPolicy):
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=3e-4,
                 training=True,
                 log_std_bounds=[-20,2],
                 action_range=[-1,1],
                 init_temperature=1.0,
                 **kwargs
                 ):
        super(MLPPolicySAC, self).__init__(ac_dim, ob_dim, n_layers, size, discrete, learning_rate, training, **kwargs)
        # Define MLP for forward pass
        self.linear1 = nn.Linear(ob_dim, size).to(ptu.device)
        self.linear2 = nn.Linear(size, size).to(ptu.device)

        self.mean_linear = nn.Linear(size, ac_dim).to(ptu.device)
        self.log_std_linear = nn.Linear(size, ac_dim).to(ptu.device)

        # Action scaling 
        self.action_scale = ptu.from_numpy( np.array([((action_range[1] - action_range[0]) / 2.)]) )
        self.action_bias = ptu.from_numpy( np.array([((action_range[1] + action_range[0]) / 2.)]) ) 

        # This was here before
        self.log_std_bounds = log_std_bounds
        self.action_range = action_range
        self.init_temperature = init_temperature
        self.learning_rate = learning_rate

        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(ptu.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)

        self.target_entropy = -ac_dim

    @property
    def alpha(self):
        # TODO: Formulate entropy term
        entropy = torch.exp(self.log_alpha)
        return entropy

    def get_action(self, obs: np.ndarray, sample=True) -> np.ndarray:
        # TODO: return sample from distribution if sampling
        # if not sampling return the mean of the distribution 
        observations = ptu.from_numpy(obs)
        action_distribution = self.forward(observations) 

        if sample:
            action = action_distribution.rsample()
        else:
            action = action_distribution.mean # TODO NEED TO FIX

        if action.shape == torch.Size([action.shape[0]]):
            action = action.unsqueeze(0)
        else:
            pass 

        action = torch.clamp(action, min=self.action_range[0], max=self.action_range[1])



        # x_t = action_distribution.rsample()  # for reparameterization trick (mean + std * N(0,1))
        # y_t = torch.tanh(x_t)
        # action = y_t * self.action_scale + self.action_bias

        return ptu.to_numpy(action)
    
    def get_log_probs(self, obs: np.ndarray, sample=True) -> np.ndarray:
        # TODO: return log_probs
        # if not sampling return the mean of the distribution 
        observations = ptu.from_numpy(obs)
        action_distribution = self.forward(observations) 

        # if sample:
        #     action = action_distribution.sample()
        # else:
        #     action = action_distribution.sample() # TODO NEED TO FIX

        # log_probs = action_distribution.log_prob(action)


        x_t = action_distribution.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = action_distribution.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_probs = log_prob.sum(1, keepdim=True)

        return ptu.to_numpy(log_probs)


    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        # TODO: Implement pass through network, computing logprobs and apply correction for Tanh squashing

        # HINT: 
        # You will need to clip log values
        # You will need SquashedNormal from sac_utils file 
        # batch_mean = self.mean_net(observation)
        # scale_tril = torch.diag(torch.exp(self.logstd))
        # batch_dim = batch_mean.shape[0]
        # batch_scale_tril = scale_tril.repeat(batch_dim, 1)
        # action_distribution = sac_utils.SquashedNormal( batch_mean, batch_scale_tril )

        if observation.shape == torch.Size([observation.shape[0]]):
            observation = observation.unsqueeze(0)
        else:
            pass 

        batch_mean = self.mean_net(observation)
        log_std = torch.clamp(self.logstd, min=self.log_std_bounds[0], max=self.log_std_bounds[1])
        std = log_std.exp()
        action_distribution = sac_utils.SquashedNormal( batch_mean, std )




        # scale_tril = torch.diag(torch.exp(self.logstd))
        # batch_dim = batch_mean.shape[0]
        # batch_scale_tril = scale_tril.repeat(batch_dim, 1)
        # action_distribution = sac_utils.SquashedNormal( batch_mean, batch_scale_tril )


        # x = F.relu(self.linear1(observation))
        # x = F.relu(self.linear2(x))
        # mean = self.mean_linear(x)
        # log_std = self.log_std_linear(x)
        # log_std = torch.clamp(log_std, min=self.log_std_bounds[0], max=self.log_std_bounds[1])
        # std = log_std.exp()
        # action_distribution = Normal(mean, std)

        # batch_mean = self.mean_net(observation)
        # log_std = torch.clamp(self.logstd, min=self.log_std_bounds[0], max=self.log_std_bounds[1])
        # std = log_std.exp()
        # action_distribution = Normal(batch_mean, std)

        return action_distribution

    def update(self, obs, critic):
        # TODO Update actor network and entropy regularizer
        # return losses and alpha value

        obs = ptu.from_numpy(obs)
        action_distribution = self.forward(obs) # get action wants numpy 
        action = action_distribution.rsample()
        action = torch.clamp(action, min=self.action_range[0], max=self.action_range[1])

        log_probs = action_distribution.log_prob(action).sum(-1, keepdim=True)

        q1_target, q2_target = critic.forward(obs,action) # Critic network wants PyTorch
        target_q = torch.minimum(q1_target, q2_target) # - self.actor.alpha.detach() * next_log_probs
        actor_loss = ((self.alpha.detach() * log_probs) - target_q).mean()

        # target_q = re_n + self.gamma * target_q * (1-terminal_n)
        # target_q = torch.unsqueeze(target_q, dim=1).detach()



        # log_probs = self.get_log_probs(obs) # get_log_probs wants numpy
        # log_probs = ptu.from_numpy(log_probs)
        # obs = ptu.from_numpy(obs)
        # q1, q2 = critic(obs, ptu.from_numpy(action)) # The critic wants torch Tensors 
        # q = torch.min(q1, q2)
        # actor_loss = ((self.alpha * log_probs) - q).mean()

        # print(actor_loss)

        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

        # Dealing with entropy 
        alpha_loss = -1*(self.alpha * (log_probs + self.target_entropy).detach() ).mean()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        # Printing alpha
        # alpha = self.log_alpha.exp()
        # print(alpha)

        return actor_loss, alpha_loss, self.alpha