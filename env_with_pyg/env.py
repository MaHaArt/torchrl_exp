import random

import networkx as nx
import torch
from tensordict import TensorDict
from torch_geometric.data import Data, Batch
from torchrl.collectors import SyncDataCollector
from torchrl.data import NonTensor
from torchrl.data import UnboundedContinuous, Bounded, Composite, Categorical
from torchrl.envs import Compose, StepCounter, TransformedEnv, DoubleToFloat
from torchrl.envs import EnvBase
from torchrl.envs.utils import check_env_specs


class SampleGraph():

    def __init__(self, nr_of_nodes, nr_of_edges):
        self.graph = nx.DiGraph()
        for n in range(nr_of_nodes):
            self.graph.add_node(n, node_attr_1=random.random(), node_attr_2=random.random() * 2.)
        edges_added = set()
        for e in range(nr_of_edges):
            u = random.randint(0, nr_of_nodes - 1)
            v = random.randint(0, nr_of_nodes - 1)
            if u != v and (u, v) not in edges_added and (v, u) not in edges_added:
                self.graph.add_edge(u, v, edge_attr=random.random())
                edges_added.add((u, v))

    def get_graph_observation(self):
        """
        Example how to the node and edge data from the networkx graph. Returns a
        PyTorch Geometric Data object along with the number of nodes and edges.

        The method iterates through all nodes and edges in the graph, constructs
        corresponding tensors, and stacks them to create the final Data object
        used for the TorchRL Environment.

        :return: A tuple containing:
            - data (Data): The graph data object containing node features, edge
              indices, and edge attributes.
            - int: The number of nodes in the graph.
            - int: The number of edges in the graph.
        :rtype: tuple
        """
        # nodes
        node_tensors = []
        for node in self.graph.nodes(data=True):
            node_tensors.append(self.get_node_tensor(node))
        nodes = torch.stack(node_tensors, dim=0)  #
        # edges
        edges_list = []
        edge_tensors = []
        for node1, node2 in self.graph.edges:
            edge_data = self.graph.get_edge_data(node1, node2)
            edges_list.append([node1, node2])
            edge_tensor = self.get_edge_tensor(edge_data)
            edge_tensors.append(edge_tensor)
        edge_index = torch.tensor(data=edges_list, dtype=torch.long).T
        edge_attributes = torch.stack(edge_tensors, dim=0)
        data = Data(x=nodes, edge_index=edge_index, edge_attributes=edge_attributes)
        return data, self.graph.number_of_nodes(), self.graph.number_of_edges(),

    def get_node_tensor(self, node):
        """
        Returns a tensor containing the features of the provided graph node.

        This method takes a node and stacks the node's attributes into a single
        tensor. The sample attributes used are 'node_attr_1' and 'node_attr_2', provided
        in the form of PyTorch tensors.

        :param node: The input node. It is expected to be a tuple, where the
            second element is a dictionary containing node attributes.
        :type node: tuple
        :returns: A tensor holding the stacked node attributes 'node_attr_1' and
            'node_attr_2'.
        :rtype: torch.Tensor
        """
        node_features = torch.stack(
            tensors=[torch.tensor(node[1]['node_attr_1']), torch.tensor(node[1]['node_attr_2'])], dim=-1)
        return node_features

    def get_edge_tensor(self, edge_data):
        """
        Extracts edge attributes (here as demo `edge_attr`) from the input edge data dictionary and converts them to a PyTorch tensor.
        """
        edge_attr = edge_data['edge_attr']
        return (torch.tensor(edge_attr))


class EnvWithGraphObs(EnvBase):
    """
    EnvWithGraphObs class.

    This class extends the EnvBase to include graph observations. It sets up the necessary specifications for observations,
    rewards, and actions, and includes methods to reset the environment and step through it.

    """

    def __init__(self, nr_of_nodes, nr_of_edges, device="cpu"):
        self.device = device
        super().__init__(device=device, batch_size=[])
        self.g_obs = None
        self.scalar_obs = None
        self.initial_nr_of_nodes = nr_of_nodes
        self.initial_nr_of_edges = nr_of_edges
        self._make_spec()

    def _make_spec(self):
        """
        Creates and sets the specifications for observation, reward, done, and action spaces.

        The method 'self.observation_spec' combines PyG graph data with two scalars (for demo purpose: the number of nodes and edges in the graph).

        :return: None
        """
        self.observation_spec = Composite(observation=Composite(  # observation combines graph_data with two scalars
            graph_data=NonTensor(shape=torch.Size([])),
            nr_of_nodes=UnboundedContinuous(shape=torch.Size([]), dtype=torch.int64, device=self.device),
            nr_of_edges=UnboundedContinuous(shape=torch.Size([]), dtype=torch.int64, device=self.device),
        ), shape=torch.Size([]))
        self.reward_spec = UnboundedContinuous(1)
        self.done_spec = Categorical(n=2, shape=torch.Size([1]), dtype=torch.bool)
        self.action_spec = Bounded(low=0, high=1, shape=torch.Size([]), dtype=torch.int, device=self.device)

    def _reset(self, tensordict=None, **kwargs):
        self.g_obs = SampleGraph(nr_of_nodes=self.initial_nr_of_nodes, nr_of_edges=self.initial_nr_of_edges)
        graph_data, nr_of_nodes, nr_of_edges = self.g_obs.get_graph_observation()
        td = TensorDict({
            "observation": {
                "graph_data": graph_data,
                "nr_of_nodes": nr_of_nodes,
                "nr_of_edges": nr_of_edges,
            },
        }, batch_size=[])
        return td

    def _step(self, tensordict):
        """
        Executes a single step in the environment, updating the graph with a new node and edge based on the  provided action, then retrieves the updated graph's observation.

         A new node is created with one random attribute and the node_attr_2 filled with the action value for demo purposes. In addition an edge is
        added between this new node and the initial node (node 0) in the graph with a random edge attribute. Thus steps lead to new graph structures.

        After updating the graph, the method retrieves the graph's data and constructs a new TensorDict
        containing information about the graph's observation, a dummy reward, and a dummy done flag.
        """
        action = tensordict["action"]
        next_node = self.g_obs.graph.number_of_nodes()
        self.g_obs.graph.add_node(next_node, node_attr_1=random.random(), node_attr_2=action)
        self.g_obs.graph.add_edge(next_node, 0, edge_attr=random.random())
        graph_data, nr_of_nodes, nr_of_edges = self.g_obs.get_graph_observation()
        next = TensorDict({
            "observation": {
                "graph_data": graph_data,
                "nr_of_nodes": nr_of_nodes,
                "nr_of_edges": nr_of_edges,
            },
            "reward": 0.,  # dummy
            "done": False,  # dummy
        }, tensordict.shape)
        return next

    def _set_seed(self, seed):
        torch.manual_seed(seed)


if __name__ == '__main__':
    env = EnvWithGraphObs(nr_of_nodes=10, nr_of_edges=2)
    check_env_specs(env, check_dtype=True)
    max_steps = 10
    env = TransformedEnv(env,
                         Compose(
                             StepCounter(max_steps=max_steps),
                             DoubleToFloat()
                         ))

    collector = SyncDataCollector(
        env,
        policy=None,  # no policy here, we demo the collector ...
        frames_per_batch=3,  # 3 graph objects will be put together in a PyG Batch
        total_frames=12,  # wie have 12 frames, thus we will receive 4 batches:
    )
    # Process batches
    for batch_idx, batch in enumerate(collector):
        # Extract pyg_data from the observation, we receive a list of `torch_geometric.data.Data` Batch
        pyg_data_list = batch["next"]["observation"][
            "graph_data"]  # Access pyg_data in key 'graph_data' in the observation TensorDict. Implictly the device information is cleared, which currently leads to a warning because clear_device_ of TensorDict is not explicitly implemented.
        # Convert the list of PyG Data objects into a PyG Batch
        batched_pyg_data = Batch.from_data_list(pyg_data_list)  # Result is a `torch_geometric.data.Batch`
        print(
            f'Batch {batch_idx} Batch of PyG Graph Data: {batched_pyg_data} to be processed with e.g. PyG ConvolutionalLayers ')


