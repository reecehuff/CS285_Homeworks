from cs285.infrastructure import pytorch_util as ptu
from .base_exploration_model import BaseExplorationModel
import torch.optim as optim
from torch import nn
import torch

def init_method_1(model):
    model.weight.data.uniform_()
    model.bias.data.uniform_()

def init_method_2(model):
    model.weight.data.normal_()
    model.bias.data.normal_()


class RNDModel(nn.Module, BaseExplorationModel):
    def __init__(self, hparams, optimizer_spec, **kwargs):
        super().__init__(**kwargs)
        self.ob_dim = hparams['ob_dim']
        self.output_size = hparams['rnd_output_size']
        self.n_layers = hparams['rnd_n_layers']
        self.size = hparams['rnd_size']
        self.optimizer_spec = optimizer_spec

        # <DONE>: Create two neural networks:
        # 1) f, the random function we are trying to learn
        # 2) f_hat, the target network that we are trying to match
        self.f = ptu.build_mlp(
            input_size=self.ob_dim,
            output_size=self.output_size,
            n_layers=self.n_layers,
            size=self.size,
            init_method=init_method_1,
        )
        self.f_hat = ptu.build_mlp(
            input_size=self.ob_dim,
            output_size=self.output_size,
            n_layers=self.n_layers,
            size=self.size,
            init_method=init_method_2,
        )

        self.optimizer = self.optimizer_spec.constructor(
            self.f_hat.parameters(),
            **self.optimizer_spec.optim_kwargs
        )

        self.learning_rate_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            self.optimizer_spec.learning_rate_schedule,
        )

        self.f.to(ptu.device)
        self.f_hat.to(ptu.device)

        # New algorithm!
        self.min_dist_toggle = hparams['use_min_dist']; # True
        print("min_dist_toggle", self.min_dist_toggle)
        self.prev_ob = None

    def forward(self, ob_no):
        # <DONE>: Get the prediction error for ob_no
        # HINT: Remember to detach the output of self.f!
        f_pred = self.f(ob_no).detach()
        f_hat_pred = self.f_hat(ob_no)
        error = torch.sqrt(torch.mean((f_pred - f_hat_pred) ** 2, dim=1))

        # Implement a new exploration algorithm that defines the error as the distance between the current observation and the previous observation
        if self.min_dist_toggle is True:
            if self.prev_ob == None:
                error = torch.zeros_like(ob_no[:, 0], requires_grad=True)
            elif self.prev_ob.shape[0] == ob_no.shape[0]:
                error = (ob_no - self.prev_ob).norm(dim=1)
                self.prev_ob = ob_no
            else: 
                error = torch.zeros_like(ob_no[:, 0], requires_grad=True)

        return error

    def forward_np(self, ob_no):
        ob_no = ptu.from_numpy(ob_no)
        error = self(ob_no)
        return ptu.to_numpy(error)

    def update(self, ob_no):
        # <DONE>: Update f_hat using ob_no
        # Hint: Take the mean prediction error across the batch
        ob_no = ptu.from_numpy(ob_no)
        error = self(ob_no)
        
        mean_pred_error = torch.mean(torch.square(error))

        self.optimizer.zero_grad()
        mean_pred_error.backward()
        self.optimizer.step()

        return ptu.to_numpy(mean_pred_error)
