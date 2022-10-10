import gym 
from collections import OrderedDict

from torch import nn

num_layers = 5
input_dim = 3
output_dim = 10

nn.Sequential()

test = OrderedDict([
          ('conv1', nn.Conv2d(1,20,5)),
          ('relu1', nn.ReLU()),
          ('conv2', nn.Conv2d(20,64,5)),
          ('relu2', nn.ReLU())
        ])

_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}

input_size = 5
output_size = 10
n_layers = 4
size = 16
activation = _str_to_activation['tanh']
output_activation = _str_to_activation['identity']

# Start with the input layer
model_list = []
append_input = nn.Linear(input_size, size)
append_activation = activation
model_list.append(append_input)
model_list.append(append_activation)

# Hidden layers 
for l in range(n_layers - 1): 
    append_input = nn.Linear(size, size)
    append_activation = activation
    model_list.append(append_input)
    model_list.append(append_activation)
    
# Output layer 
append_input = nn.Linear(size, output_size)
append_activation = output_activation
model_list.append(append_input)
model_list.append(append_activation)

model = nn.Sequential(*model_list)

modules = []
modules.append(nn.Linear(10, 10))
modules.append(nn.Linear(10, 10))


sequential = nn.Sequential(*modules)


MJ_ENV_NAMES = ["Ant-v4", "Walker2d-v4", "HalfCheetah-v4", "Hopper-v4"]
MJ_ENV_KWARGS = {name: {"render_mode": "rgb_array"} for name in MJ_ENV_NAMES}
