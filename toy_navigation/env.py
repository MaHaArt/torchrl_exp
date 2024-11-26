import torch
from tensordict import TensorDict
from torchrl.data import UnboundedContinuous, Bounded, Composite, Categorical, OneHot
from torchrl.envs import Compose, ObservationNorm, StepCounter, TransformedEnv, DoubleToFloat
from torchrl.envs import EnvBase
from torchrl.envs.utils import check_env_specs


class ToyNavigation(EnvBase):

    def __init__(self, max_step_width=1., grid_extent=5, device="cuda"):
        self.device = device
        super().__init__(device=device, batch_size=[])
        self.max_step_width = max_step_width
        self.grid_extent = grid_extent
        self.target = torch.zeros(2, device=self.device)
        self.position = torch.zeros(2, device=self.device)
        self.initial_position = torch.zeros(2, device=self.device)
        self.steps = 0
        self._make_spec()

    def _reset(self, tensordict=None, **kwargs):
        """
        Constructs and returns a TensorDict containing the current observation.
        The target and position values are randomly generated within the range of
        [-grid_extent, grid_extent].
        """

        self.target = torch.rand(2,
                                 device=self.device) * 2 * self.grid_extent - self.grid_extent  # Random target in [-grid_extent, grid_extent]
        self.position = torch.rand(2,
                                   device=self.device) * 2 * self.grid_extent - self.grid_extent  # Random position in [-grid_extent, grid_extent]
        self.initial_position = self.position
        self.steps = 0
        position = self.position.clone()  # do we have to clone to get distinct positions in the trajectory?
        target = self.target.clone()
        td = TensorDict({
            "observation": {"position": position, "target": target},
        }, batch_size=self.batch_size)
        return td

    def _step(self, tensordict):
        """
        Perform a single step in the environment.

        This function takes an input `tensordict` containing the action details and
        updates the agent's position accordingly. The reward is calculated based
        on the distance to the target, and whether the task is done is determined
        by whether the agent is close enough to the target.

        :param tensordict: A dictionary containing the action information.
                           It should have "axis" and "magnitude" keys to
                           specify the movement direction and (normalized) step size. The step_size is multiplied with max_step_width
        :type tensordict: TensorDict
        :return: A dictionary with updated observation including current position
                 and target, computed reward, and done flag.
        :rtype: TensorDict
        """
        self.steps += 1
        direction = tensordict["action"]["axis"]  # 0 for x, 1 for y
        magnitude = tensordict["action"]["magnitude"]

        move = torch.zeros(2, device=self.device)
        move[direction] = magnitude * self.max_step_width
        self.position += move

        distance = torch.norm(self.position - self.target)
        reward = -distance * 10 - self.steps
        done = distance < 0.1
        # Remark: gets truncated when steps > max_steps, see step counter transform
        position = self.position.clone()  # do we have to clone to get distinct positions in the trajectory?
        target = self.target.clone()
        next = TensorDict({
            "observation": {"position": position, "target": target},
            "reward": reward,
            "done": done,

        }, tensordict.shape)

        return next

    def _set_seed(self, seed):
        torch.manual_seed(seed)

    def _make_spec(self, **kwargs):
        self.observation_spec = Composite(observation=Composite(
            position=UnboundedContinuous(shape=torch.Size([2]), device=self.device),
            target=UnboundedContinuous(shape=torch.Size([2]), device=self.device),
            shape=torch.Size([])
        ), shape=torch.Size([]))

        self.action_spec = action_spec()

        self.reward_spec = UnboundedContinuous(1)

        self.done_spec = Categorical(n=2, shape=torch.Size([1]), dtype=torch.bool)
        self.terminated_spec = Categorical(n=2, shape=torch.Size([1]), dtype=torch.bool)


def action_spec():
    return Composite(
        action=Composite(
            magnitude=Bounded(low=-1., high=1., shape=torch.Size([]), dtype=torch.float),  # normalized width of step
            axis=OneHot(n=2, dtype=torch.int),  # x or y axis
            shape=torch.Size([])
        ), shape=torch.Size([]))


def make_toy_env(max_step_width=1, grid_extent=5, max_steps=10, device='cuda'):
    """
    This function initializes a ToyNavigation environment with the specified maximum step width,
    grid extent, device, and maximum steps. It then applies the StepCounter transformation to the environment
    which introduces step_count in the tensordict and limits the number of step to `max_steps`.
    """
    env = ToyNavigation(max_step_width=max_step_width, grid_extent=grid_extent, device=device)
    env = TransformedEnv(env, Compose(
        #ObservationNorm(in_keys=[("observation", "position"), ("observation", "target")]),
        StepCounter(max_steps=max_steps), ))
    # env.transform[0].init_stats(key=("observation", "position"), num_iter=2000)
    return env


def custom_tensor_print(tensor):
    # Remove tensor brackets and device info, then wrap in parentheses
    tensor_str = str(tensor.cpu().numpy()).strip('[]')
    return f"({tensor_str})"


if __name__ == '__main__':
    env = make_toy_env()
    check_env_specs(env, check_dtype=True)
    eval_rollout = env.rollout(max_steps=5)
    print(eval_rollout)

