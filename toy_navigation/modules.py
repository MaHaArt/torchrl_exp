import torch
import torch.nn.functional as F
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, CompositeDistribution, InteractionType
from torch import nn
from torch.distributions import Categorical, Normal
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from toy_navigation.env import action_spec


def normalize(tensor):
    """
    Normalize the input tensor to a range of [-1, 1].
    """
    min_val = tensor.min(dim=-1, keepdim=True)[0]
    max_val = tensor.max(dim=-1, keepdim=True)[0]
    return 2 * (tensor - min_val) / (max_val - min_val + 1e-8) - 1

# actor

class Policy(nn.Module):
    def __init__(self, n_hidden_layers=2, hidden_features=5, device='cuda'):
        super().__init__()
        self.device = device
        self.hidden_layers = nn.ModuleList()
        for i in range(n_hidden_layers):
            in_channels = 4 if i == 0 else hidden_features
            hidden_layer = nn.Linear(in_features=in_channels, out_features=hidden_features)
            nn.init.xavier_uniform_(hidden_layer.weight)
            self.hidden_layers.append(hidden_layer)
        self.axis_output = nn.Linear(in_features=hidden_features, out_features=2)
        nn.init.xavier_uniform_(self.axis_output.weight)
        self.magnitude_output = nn.Linear(in_features=hidden_features, out_features=2)
        nn.init.xavier_uniform_(self.magnitude_output.weight)
        self.to(device)



    def forward(self, position, target):
        """
        Forward pass for the actor network. This method concatenates the `position` and `target` tensors,
        processes them through a series of hidden layers, and computes the final output, which includes
        axis logits for the direction of the movement and magnitude for the (normalized) step_width.
        """
        # Normalize position and target
        position = normalize(position)
        target = normalize(target)

        batched = position.ndim > 1  # called with batched params?

        # Concatenate position and target
        tensor = torch.cat([position, target], dim=-1).to(self.device)
        # Process through hidden layers
        for layer in self.hidden_layers:
            tensor = F.leaky_relu(layer(tensor))

        # Separate loc and scale from magnitude output
        axis_logits = self.axis_output(tensor)
        magnitude = self.magnitude_output(tensor)
        if batched:
            loc = magnitude[...,0]
            scale = magnitude[...,1]
        else:
            loc = magnitude[0]
            scale = magnitude[1]

        loc = F.tanh(loc)
        scale = torch.clamp(F.softplus(scale) + 1e-5,min=1e-5,max=2.)


        result = TensorDict({
            'params': {'axis': {'logits': axis_logits},
                       'magnitude': {'loc': loc,
                                     'scale': scale}}
        })
        return result


def get_actor(n_layers=3, hidden_features=10, device='cuda'):
    """
    Create an actor module composed of a Policy neural network, TensorDictModule to transform input tensordicts to ouput, and the
    ProbabilisticActor configuration

    """
    actor_net = Policy(n_layers, hidden_features, device)
    td_module = TensorDictModule(
        actor_net,
        in_keys={("observation", "position"): 'position', ("observation", "target"): 'target'},
        out_keys = [('params', 'axis', 'logits'), ('params', 'magnitude','loc'), ('params', 'magnitude','scale')]
    )
    policy_module = ProbabilisticActor(
        module=td_module,
        in_keys=["params"],
        distribution_class=CompositeDistribution,
        spec = action_spec(),
        distribution_kwargs={
            "aggregate_probabilities": True,
            "distribution_map": {
                "axis": Categorical,
                "magnitude": Normal,
            },
            "name_map": {
                "axis": ("action", "axis"),  # see action_spec and nested structure of action
                "magnitude": ("action", "magnitude"),
            },
        },
        return_log_prob=True,
        default_interaction_type=InteractionType.MODE  # siehe hinweis zu Laufzeit bei DETERMINISTIC
    )
    policy_module.td_module = td_module
    policy_module.actor_net = actor_net
    return policy_module


class ValueNetwork(nn.Module):
    def __init__(self, n_hidden_layers=1, hidden_features=2, device='cuda'):
        super().__init__()
        self.device = device
        self.hidden_layers = nn.ModuleList()
        for i in range(n_hidden_layers):
            in_channels = 4 if i == 0 else hidden_features
            hidden_layer = nn.Linear(in_features=in_channels, out_features=hidden_features)
            nn.init.xavier_uniform_(hidden_layer.weight)
            self.hidden_layers.append(hidden_layer)
        self.value_output = nn.Linear(in_features=hidden_features, out_features=1)
        nn.init.xavier_uniform_(self.value_output.weight)
        self.to(device)


    def forward(self, position, target):
        # Normalize position and target
        position = normalize(position)
        target = normalize(target)

        tensor = torch.cat([position, target], dim=-1).to(self.device)
        for layer in self.hidden_layers:
            tensor = F.leaky_relu(layer(tensor))
        value = F.tanh(self.value_output(tensor))
        return value


# value net

def get_critic( n_layers=3, hidden_features=10,device='cuda'):
    """
    Creates a ValueOperator critic by initializing the ValueNetwork with specified
    number of layers, and hidden features.
    The critic is responsible for evaluating the value of given states in the environment using a neural
    network configured with the provided parameters.

    """
    critic_net = ValueNetwork(n_hidden_layers=n_layers, hidden_features=hidden_features, device=device)
    critic = ValueOperator(
        module=critic_net,
        in_keys={("observation", "position"): 'position', ("observation", "target"): 'target'},
    )
    return critic


def get_advantage(critic, gamma=0.9995, lmbda=0.95, average_gae=True):
    """
    Computes the generalized advantage estimation (GAE) for a given critic network.
    This function initializes the GAE module with the specified parameters, which are used to
    calculate the advantage function for the  reinforcement learning scenario. Generalized advantage
    estimation helps to reduce variance while maintaining an admissible level of bias.
    """
    module = GAE(
        gamma=gamma, lmbda=lmbda, value_network=critic, average_gae=average_gae
    )
    return module


def get_loss_module(actor, critic,
                    clip_epsilon = 0.2, entropy_eps = 1e-4,normalize_advantage=False,clip_value=True,separate_losses=False,critic_coef=1.0 ):
    """
    Returns the loss module for the Clip Proximal Policy Optimization (PPO)
    algorithm using the given actor and critic networks.
    """

    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=clip_epsilon,
        entropy_bonus=bool(entropy_eps),
        entropy_coef=entropy_eps,
        critic_coef=critic_coef,
        loss_critic_type="smooth_l1",
        normalize_advantage=normalize_advantage,
        reduction="mean",
        clip_value=clip_value,
        separate_losses=separate_losses,

    )
    return loss_module





