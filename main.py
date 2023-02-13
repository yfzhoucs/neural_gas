from sklearn import datasets
from matplotlib import pyplot as plt
import random
import numpy as np
from matplotlib.lines import Line2D

class NeuralGas:
    def __init__(self, num_dim=2, age_step=1, epsilon_b=0.2, epsilon_n=0.006, age_max=50, param_lambda=100, alpha=0.5, d=0.995):
        self.num_dim = num_dim
        self.nodes = {}
        self.edges = {}

        for i in range(2):
            node = np.random.rand(self.num_dim)
            self.nodes[i] = {
                'value': node,
                'accumulated_error': 0,
                'edges': {0},
            }
        self.edges[0] = {
            'nodes': {0, 1},
            'age': 0,
        }
        self.max_node_id = 1
        self.max_edge_id = 0
        self.num_input_signals = 0

        self.age_step = age_step
        self.epsilon_b = epsilon_b
        self.epsilon_n = epsilon_n
        self.age_max = age_max
        self.param_lambda = param_lambda
        self.alpha = alpha
        self.d = d
    
    # data1, data2: numpy array of (num_dim,)
    def _l2_dist_(self, data1, data2):
        dist = np.sqrt(np.sum((data1 - data2) ** 2))
        return dist

    # data: numpy array of (num_dim,)
    def _find_2_nearest_neighbours_(self, data):
        node_keys = list(self.nodes.keys())
        nearest_2_nodes = []
        nearest_2_distances = []
        for i in range(len(self.nodes)):
            current_node = self.nodes[node_keys[i]]['value']
            dist = self._l2_dist_(current_node, data)
            if len(nearest_2_nodes) == 0:
                nearest_2_nodes.append(node_keys[i])
                nearest_2_distances.append(dist)
            elif len(nearest_2_nodes) == 1:
                if dist > nearest_2_distances[0]:
                    nearest_2_distances.append(dist)
                    nearest_2_nodes.append(node_keys[i])
                else:
                    nearest_2_distances = [dist, nearest_2_distances[0]]
                    nearest_2_nodes = [node_keys[i], nearest_2_nodes[0]]
            else:
                if dist < nearest_2_distances[0]:
                    nearest_2_distances = [dist] + nearest_2_distances[1:]
                    nearest_2_nodes = [node_keys[i]] + nearest_2_nodes[1:]
                elif dist < nearest_2_distances[1]:
                    nearest_2_distances = nearest_2_distances[:1] + [dist]
                    nearest_2_nodes = nearest_2_nodes[:1] + [node_keys[i]]
                else:
                    pass
        return nearest_2_nodes, nearest_2_distances


    def _increment_age_(self, node, age_step):
        for edge_key in self.edges:
            if node in self.edges[edge_key]['nodes']:
                self.edges[edge_key]['age'] += age_step


    def _move_nearest_neighbour_and_its_neighbours_(self, data, node_id):
        delta_value =  data - self.nodes[node_id]['value']
        self.nodes[node_id]['value'] += delta_value * self.epsilon_b

        list_neighbour_ids = []
        for edge_key in self.edges:
            if node_id in self.edges[edge_key]['nodes']:
                list_neighbour_ids.append((self.edges[edge_key]['nodes'] - {node_id}).pop())
        for node_id in list_neighbour_ids:
            delta_value =  data - self.nodes[node_id]['value']
            self.nodes[node_id]['value'] += delta_value * self.epsilon_n


    def _connect_2_nodes_(self, node1_id, node2_id):
        for edge_id in self.edges:
            current_edge_nodes = self.edges[edge_id]['nodes']
            if node1_id in current_edge_nodes and node2_id in current_edge_nodes:
                self.edges[edge_id]['age'] = 0
                return
        self.max_edge_id += 1
        self.edges[self.max_edge_id] = {
            'nodes': {node1_id, node2_id},
            'age': 0
        }
        self.nodes[node1_id]['edges'].add(self.max_edge_id)
        self.nodes[node2_id]['edges'].add(self.max_edge_id)


    def _remove_edges_too_old_(self):
        edges_too_old = set()
        for edge_id in self.edges:
            if self.edges[edge_id]['age'] > self.age_max:
                edges_too_old.add(edge_id)
        
        nodes_removed_edges = set()
        for edge_id in edges_too_old:

            for node_id in self.edges[edge_id]['nodes']:
                self.nodes[node_id]['edges'].remove(edge_id)
                nodes_removed_edges.add(node_id)
            self.edges.pop(edge_id)
        
        isolated_nodes = set()
        for node_id in nodes_removed_edges:
            if len(self.nodes[node_id]['edges']) == 0:
                isolated_nodes.add(node_id)
        for node_id in isolated_nodes:
            self.nodes.pop(node_id)


    def _add_new_node_(self):
        # find q
        max_error_node = None
        max_error = -1
        for node_id in self.nodes:
            if self.nodes[node_id]['accumulated_error'] > max_error:
                max_error = self.nodes[node_id]['accumulated_error']
                max_error_node = node_id
        
        # find f
        max_error_neighbour = None
        max_error_neighbour_value = -1
        neighbour_ids = set()
        for edge_id in self.nodes[max_error_node]['edges']:
            neighbour_ids.add((self.edges[edge_id]['nodes'] - {max_error_node}).pop())
        for node_id in neighbour_ids:
            if self.nodes[node_id]['accumulated_error'] > max_error_neighbour_value:
                max_error_neighbour_value = self.nodes[node_id]['accumulated_error']
                max_error_neighbour = node_id
        
        # define r
        self.max_node_id += 1
        self.nodes[self.max_node_id] = {
                'value': 0.5 * (self.nodes[max_error_node]['value'] + self.nodes[max_error_neighbour]['value']),
                'accumulated_error': 0,
                'edges': set(),
        }
        self._connect_2_nodes_(self.max_node_id, max_error_node)
        self._connect_2_nodes_(self.max_node_id, max_error_neighbour)

        # remove the old q-f edge
        edge_to_remove = None
        for edge_id in self.edges:
            current_edge_nodes = self.edges[edge_id]['nodes']
            if max_error_node in current_edge_nodes and max_error_neighbour in current_edge_nodes:
                edge_to_remove = edge_id
                break
        self.edges.pop(edge_to_remove)
        self.nodes[max_error_node]['edges'].remove(edge_to_remove)
        self.nodes[max_error_neighbour]['edges'].remove(edge_to_remove)

        # decrease error for q and f, set error for r
        self.nodes[max_error_node]['accumulated_error'] *= self.alpha
        self.nodes[max_error_neighbour]['accumulated_error'] *= self.alpha
        self.nodes[self.max_node_id]['accumulated_error'] = self.nodes[max_error_node]['accumulated_error']


    def _decrease_all_error_(self):
        for node_id in self.nodes:
            self.nodes[node_id]['accumulated_error'] *= self.d


    # data: numpy array of (num_dim,)
    def input_data_point(self, data, gt=None):
        self.num_input_signals += 1
        nearest_2_nodes, nearest_2_distances = self._find_2_nearest_neighbours_(data)
        self._increment_age_(nearest_2_nodes[0], self.age_step)
        self.nodes[nearest_2_nodes[0]]['accumulated_error'] += nearest_2_distances[0]
        self._move_nearest_neighbour_and_its_neighbours_(data, nearest_2_nodes[0])
        self._connect_2_nodes_(nearest_2_nodes[0], nearest_2_nodes[1])
        self._remove_edges_too_old_()
        if self.num_input_signals % self.param_lambda == 0:
            self._add_new_node_()
            if gt is not None:
                self.plot_net(gt)
        self._decrease_all_error_()
        return
    
    def return_net(self):
        return self.nodes, self.edges
    
    def plot_net(self, gt=None):
        plt.clf()

        xys = np.array([self.nodes[node_id]['value'] for node_id in self.nodes])
        plt.scatter(xys[:,0], xys[:,1], c='blue')

        for edge_id in self.edges:
            nodes = []
            for node_id in self.edges[edge_id]['nodes']:
                node_value = self.nodes[node_id]['value']
                nodes.append(node_value)
                # print(node_value)
            line = Line2D((nodes[0][0], nodes[1][0]), (nodes[0][1], nodes[1][1]), color='blue')
            plt.plot(*line.get_data(), color='blue')

            # plt.text((nodes[0][0] + nodes[1][0]) / 2, (nodes[0][1] + nodes[1][1]) / 2, f"{self.edges[edge_id]['age']}")
        
        if gt is not None:
            plt.scatter(gt[:, 0], gt[:, 1], c='gray')
        plt.show()


if __name__ == '__main__':

    data = datasets.make_moons(n_samples=100, noise=0.05)[0]
    # print(data.shape)
    # plt.scatter(data[:,0], data[:,1])
    # plt.show()

    ng = NeuralGas(num_dim=2, age_max=50)
    for epoch in range(42):
        for i in range(data.shape[0]):
            ng.input_data_point(data[i], gt=None)
    print(len(ng.edges), len(ng.nodes))
    ng.plot_net(data)