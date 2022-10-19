from cs285.policies.MLP_policy import MLPPolicy
import torch
import numpy as np
from cs285.infrastructure import sac_utils
from cs285.infrastructure import pytorch_util as ptu
from torch import nn
from torch import optim
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
        self.log_std_bounds = log_std_bounds
        self.action_range = action_range
        self.init_temperature = init_temperature
        self.learning_rate = learning_rate

        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(ptu.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)

        self.target_entropy = -ac_dim
        self.mean_net = ptu.build_mlp(input_size=self.ob_dim,
                                  output_size=self.ac_dim,
                                  n_layers=self.n_layers, size=self.size)
        self.logstd = nn.Parameter(
            torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
        )
        self.mean_net.to(ptu.device)
        self.logstd.to(ptu.device)
        self.actor_optimizer = optim.Adam(
            itertools.chain([self.logstd], self.mean_net.parameters()), lr= 1e-4)

    @property
    def alpha(self):
        # TODO: Formulate entropy term
        
        #pass
        #return entropy
        return torch.exp(self.log_alpha) 

    def get_action(self, obs: np.ndarray, sample=True) -> np.ndarray:
        # TODO: return sample from distribution if sampling
        # if not sampling return the mean of the distribution
        """
        if len(obs.shape) > 1:
          observation = obs
        else:
          observation = obs[None]
        """

        observation = ptu.from_numpy(obs)
        action_distribution = self.forward(observation)
        if sample:
          action = action_distribution.rsample()  # don't bother with rsample
        else:
          action=action_distribution.mean
        if action.shape==torch.Size([action.shape[0]]):
          action=action.unsqueeze(0)
        else:
          pass
        action=torch.clamp(action, self.action_range[0],self.action_range[1])

        return ptu.to_numpy(action)
        #pass 
        #return action

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
        #class SquashedNormal(dist.transformed_distribution.TransformedDistribution):
        #mu, log_std = self.trunk(observation).chunk(2, dim=-1)
        if observation.shape==torch.Size([observation.shape[0]]):
          observation=observation.unsqueeze(0)
        else:
          pass

        mu = self.mean_net(observation)
        #log_std=self.logstd
        #log_std = torch.clamp(log_std,self.log_std_bounds[0],self.log_std_bounds[1])
        #log_std_min, log_std_max = self.log_std_bounds
        #log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std +
        #
        log_std = torch.exp(torch.clamp(self.logstd, self.log_std_bounds[0], self.log_std_bounds[1]))                                                          
        #std = log_std.exp()

        #self.outputs['mu'] = mu
        #self.outputs['std'] = std

        action_distribution = sac_utils.SquashedNormal(mu, log_std)#.rsample()

        #pass 
        return action_distribution

    def update(self, obs, critic):
        # TODO Update actor network and entropy regularizer
        # return losses and alpha value
        #pass

        #detach alpha and target entropy
        obs=ptu.from_numpy(obs)

        dist = self.forward(obs)
        action = dist.rsample()

        action=torch.clamp(action,self.action_range[0],self.action_range[1])

        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = critic.forward(obs, action)

        actor_Q = torch.minimum(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()


        # optimize the actor
        """
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        """
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()


        
        alpha_loss = (self.alpha *
                      (-log_prob - self.target_entropy).detach()).mean()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()


        return actor_loss, alpha_loss, self.alpha