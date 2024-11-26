from collections import defaultdict
import torch
from torchrl.collectors import SyncDataCollector
from torchrl.envs.utils import ExplorationType, set_exploration_type
from tqdm import tqdm

from toy_navigation.env import make_toy_env
from toy_navigation.modules import get_actor, get_critic, get_advantage, get_loss_module

import matplotlib.pyplot as plt
import torch.nn.functional as F

def do_train(silent=True, device=None, **kwargs):
    """
    Trains a reinforcement learning model for the custom "Toy Navigation" Environment using given or default configuration parameters.

    This function initializes the necessary training modules, updates configuration settings, and
    executes the training loop for a specified number of epochs and frames. The function supports
    to either report to Ray Tune or generate visual logs for monitoring
    the training progress.


    :param silent: If True, the function suppresses detailed output and reports metrics to Ray Tune.
                   If False, the function shows progress and logging details using `tqdm`.
    :param device:  device for training
    :param kwargs: Additional keyword arguments to override the default configuration parameters.
    :return: None
    """
    # Default configuration parameters
    default_config = {
        'frames_per_batch': 4500, 'total_frames': 76500,
        'learn_rate': 2e-5,'lr_scheduling': True, 'num_epochs': 500,
        'actor_n_layers': 2, 'actor_hidden_features': 40,
        'critic_n_layers': 2,'critic_hidden_features': 40,
        'split_trajs': False,
        'gamma': 0.85,'lmbda': 0.96,'average_gae': True,
        'clip_epsilon' : 0.2, 'entropy_eps' :3e-4, 'normalize_advantage' : True, 'clip_value' : True,
        'separate_losses' : True, 'critic_coef' : 0.75
    }

    # Update default config with any provided kwargs
    config = {**default_config, **kwargs}

    frames_per_batch = config['frames_per_batch']
    learn_rate = config['learn_rate']
    num_epochs = config['num_epochs']
    total_frames = config['total_frames']
    lr_scheduling = config['lr_scheduling']

    if silent:
        from ray import train

    if not device:
     device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Toy Navigation parameters
    grid_extent = 5
    max_step_width = 1
    max_steps = 30  #

    # instantiate modules
    actor = get_actor(device=device, n_layers=config['actor_n_layers'],
                      hidden_features=config['actor_hidden_features'])
    critic = get_critic(device=device, n_layers=config['critic_n_layers'],
                        hidden_features=config['critic_hidden_features'])
    advantage_module = get_advantage(critic, gamma=config['gamma'], lmbda=config['lmbda'],
                                     average_gae=config['average_gae'])
    loss_module = get_loss_module(actor, critic, clip_epsilon = config['clip_epsilon'],
                                  entropy_eps = config['entropy_eps'],normalize_advantage=config['normalize_advantage'],clip_value=config['clip_value'],separate_losses=config['separate_losses'],critic_coef=config['critic_coef'])
    # instantiate optimiser and scheduler
    optim = torch.optim.Adam(loss_module.parameters(), learn_rate)
    if lr_scheduling:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, total_frames // frames_per_batch, 0.0)

    # define collector
    collector = SyncDataCollector(
        lambda: make_toy_env(max_step_width=max_step_width, grid_extent=grid_extent, max_steps=max_steps),
        device=device,
        policy=actor,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        split_trajs=config['split_trajs'],
        exploration_type=ExplorationType.RANDOM,
        set_truncated=True,
    )

    if not silent:
        env = make_toy_env(max_step_width=max_step_width, grid_extent=grid_extent, max_steps=max_steps)
        logs = defaultdict(list)
        pbar = tqdm(total=total_frames)
        eval_str = ""

    for batch_idx, data in enumerate(collector):

        for _ in range(num_epochs):
            advantage_module(data) # do we need torch_no_grad context?
            data_view = data.reshape(-1)  # Dimension: Frames per bach
            loss_vals = loss_module(data_view)
            loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
            )
            loss_value = loss_value.mean()
            if silent:
                # Report metrics to Tune
                reported_loss = loss_value.detach().cpu().item()
                train.report({'loss': reported_loss})
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_norm=1)
            optim.step()
            optim.zero_grad()

        if lr_scheduling:
            scheduler.step()

        if not silent:
            logs["reward"].append(data["next", "reward"].mean().item())
            pbar.update(data.numel())
            cum_reward_str = f"avg. reward={logs['reward'][-1]: 4.2f} (init={logs['reward'][0]: 4.2f})"
            logs["learn_rate"].append(optim.param_groups[0]["lr"])
            lr_str = f"learn_rate policy: {logs['learn_rate'][-1]: 4.6f}"
            if batch_idx % 10 == 0:
                # We evaluate the policy once every 10 batches of data.
                # Evaluation is rather simple: execute the policy without exploration
                # (take the expected value of the action distribution) for a given
                # number of steps (max_steps, which is our ``env`` horizon).
                # The ``rollout`` method of the ``env`` can take a policy as argument:
                # it will then execute this policy at each step.
                with set_exploration_type(ExplorationType.MODE), torch.no_grad():
                    # execute a rollout with the trained policy
                    eval_rollout = env.rollout(max_steps, actor)
                    logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
                    eval_str = (
                        f"eval avg. reward: {logs['eval reward'][-1]: 4.2f} "
                        f"(init: {logs['eval reward'][0]: 4.2f}) "
                    )
                    del eval_rollout
            pbar.set_description(", ".join([eval_str, cum_reward_str,  lr_str]))

    collector.shutdown()
    if not silent:
        pbar.close()
        plt.figure(figsize=(10, 10))
        plt.subplot(2, 2, 1)
        plt.plot(logs["reward"])
        plt.title("training rewards (average)")
        plt.subplot(2, 2, 2)
        plt.plot(logs["step_count"])
        plt.title("Max step count (training)")
        plt.subplot(2, 2, 3)
        plt.plot(logs["eval reward (sum)"])
        plt.title("Reward (test)")
        plt.subplot(2, 2, 4)
        plt.plot(logs["eval step_count"])
        plt.title("Max step count (test)")
        plt.show()
    return actor


def eval_actor(actor, env, max_steps):
    with set_exploration_type(ExplorationType.MODE), torch.no_grad():
        for batch_idx in range(3):
            print(f"\n Eval Experiment {batch_idx}")
            env.reset()
            print(f"Initial Pos: {env.initial_position}, Target: {env.target}")
            eval_rollout = env.rollout(max_steps, actor)
            for j in range(eval_rollout.shape[0]):
                target = eval_rollout[j]['observation', 'target']
                position = eval_rollout[j]['observation', 'position']
                distance = torch.norm(position - target)
                print(f"Step {j}: Position: {position} in distance {distance.item()} \n ")


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trained_actor = do_train(silent=False,actor_n_layers=2, actor_hidden_features=20,
                             frames_per_batch=2000,total_frames=50000)
    eval_env = make_toy_env(max_step_width=1., grid_extent=5., max_steps=20.)
    eval_actor(trained_actor, eval_env, max_steps=20)
