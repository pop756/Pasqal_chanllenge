import pennylane as qml  # Importing PennyLane for quantum computing
from pennylane import numpy as np  # Importing PennyLane's NumPy for compatibility
import torch  # Importing PyTorch for potential machine learning applications
import torch.nn as nn  # Importing PyTorch's neural network module
import copy  # Importing copy module for deep copying objects
import itertools  # Importing itertools for combinatorial operations
from tqdm import tqdm  # Importing tqdm for progress tracking
from scipy.optimize import minimize
import random
import pennylane as qml

import copy
import itertools
import torch
import pennylane as qml






class TreeNode:
    def __init__(self, key, value):
        """
        Initializes a tree node.
        :param key: The unique identifier for the node.
        :param value: The data associated with the node.
        """
        self.key = key  # Node's unique key
        self.value = value  # Node's stored value
        self.children = {}  # Dictionary to store child nodes (key -> TreeNode mapping)

    def __repr__(self):
        """
        Returns a string representation of the node.
        """
        return f"TreeNode({self.key}: {self.value})"


class Tree:
    def __init__(self, root_key, root_value):
        """
        Initializes a tree with a root node.
        :param root_key: The unique key for the root node.
        :param root_value: The data associated with the root node.
        """
        self.root = TreeNode(root_key, root_value)  # Create the root node
        self.state = self.root  # Set the current state to the root node
        self.node_num = 0
    def has_child(self, key):
        """
        Checks if the current state (node) has a child with the given key.
        :param key: The key of the child node to check.
        :return: True if the child exists, False otherwise.
        """
        return key in self.state.children  # Check if the key exists in the children dictionary

    def move(self, key):
        """
        Moves the current state to a child node if it exists.
        :param key: The key of the child node to move to.
        :raises ValueError: If the child does not exist.
        """
        if self.has_child(key):  # If the child exists, move to it
            self.state = self.state.children[key]

        else:
            raise ValueError(f"Error: No child with key '{key}' exists.")  # Raise an error if child doesn't exist

    def create(self, key, value):
        """
        Creates a new child node under the current state if the key does not already exist.
        :param key: The key of the new child node.
        :param value: The value to store in the new node.
        :raises ValueError: If the key already exists.
        """
        if not self.has_child(key):  # If the child does not exist, create it
            new_node = TreeNode(key, value)
            self.state.children[key] = new_node  # Add the new node to the children dictionary
            self.node_num +=1
        else:
            raise ValueError(f"Error: Child '{key}' already exists.")  # Raise an error if child already exists

    def reset_state(self):
        """
        Resets the current state back to the root node.
        """
        self.state = self.root  # Set state back to the root node


    def display_tree(self, node=None, level=0):
        """
        Recursively prints the structure of the tree.
        :param node: The node to start printing from (default is the root node).
        :param level: The indentation level for printing the tree hierarchy.
        """
        if node is None:  # If no node is provided, start from the root
            node = self.root
        print("  " * level + f"{node.key}: {node.value}")  # Print the current node with indentation
        for child in node.children.values():  # Iterate through all child nodes
            self.display_tree(child, level + 1)  # Recursively print child nodes with increased indentation



class RL_QAOA:
    """
    A reinforcement learning-based approach to solving QAOA (Quantum Approximate Optimization Algorithm)
    for quadratic unconstrained binary optimization (QUBO) problems.

    Parameters
    ----------
    Q : np.ndarray
        QUBO matrix representing the optimization problem.

    n_c : int
        The threshold number of nodes at which classical brute-force optimization is applied.

    init_paramter : np.ndarray
        Initial parameters for the QAOA circuit.

    b_vector : np.ndarray
        The beta vector used in reinforcement learning to guide edge selection.

    QAOA_depth : int
        Depth of the QAOA circuit, representing the number of layers.

    gamma : float, default=0.99
        Discount factor used in reinforcement learning.

    learning_rate_init : float, default=0.001
        Initial learning rate for the Adam optimizer.

    Attributes
    ----------
    qaoa_layer : QAOA_layer
        Instance of the QAOA layer with specified depth and QUBO matrix.

    optimizer : AdamOptimizer
        Adam optimizer instance to optimize QAOA parameters.

    same_list : list
        List of edges that should have the same value.

    diff_list : list
        List of edges that should have different values.

    node_assignments : dict
        Tracks assigned values to the nodes.

    """

    def __init__(self, Q, n_c, init_paramter, b_vector, QAOA_depth, gamma=0.99, learning_rate_init=[0.01,0.05]):
        self.Q = Q
        self.n_c = n_c
        self.param = init_paramter
        self.b = b_vector
        self.p = QAOA_depth
        self.qaoa_layer = QAOA_layer(QAOA_depth, Q)
        self.gamma = gamma
        self.optimzer = AdamOptimizer([init_paramter, b_vector], learning_rate_init=learning_rate_init)
        self.lr = learning_rate_init
        self.tree = Tree('root',None)
        self.tree_grad = Tree('root',None)

    def RL_QAOA(self, episodes, epochs, correct_ans=None):
        self.avg_values = []
        self.min_values = []
        self.prob_values = []
        self.best_states = []
        self.best_same_lists = []
        self.best_diff_lists = []

        """
        Performs the reinforcement learning optimization process with progress tracking.

        Parameters
        ----------
        episodes : int
            Number of Monte Carlo trials for the optimization.

        epochs : int
            Number of optimization iterations to update parameters.

        correct_ans : float, optional
            The correct optimal solution (if available) to calculate success probability.
        """

        for j in range(epochs):

            if self.lr[0] != 0:
                num = self.tree.node_num

                self.tree = Tree('root',None)
                self.tree.node_num = num
                self.tree_grad = Tree('root',None)
                self.tree_grad.node_num = num

            value_list = []
            state_list = []
            QAOA_diff_list = []
            beta_diff_list = []
            same_lists = []
            diff_lists = []

            if correct_ans is not None:
                prob = 0

            # Progress bar for episodes within the current epoch
            for i in tqdm(range(episodes), desc=f'Epoch {j + 1}/{epochs}', unit=' episode'):
                QAOA_diff, beta_diff, value, final_state, same_list, diff_list = self.rqaoa_execute()
                value_list.append(value)
                state_list.append(final_state)
                same_lists.append(same_list)
                diff_lists.append(diff_list)
                QAOA_diff_list.append(QAOA_diff)
                beta_diff_list.append(beta_diff)

                if correct_ans is not None and correct_ans - 0.01 <= value <= correct_ans + 0.01:
                    prob += 1

            # Compute softmax rewards and normalize

            batch_mean = (np.array(value_list) - np.mean(value_list))
            #batch_plus = np.where(batch_mean < 0, batch_mean, 0)
            #softmaxed_rewards = signed_softmax_rewards(batch_plus, beta=15)*episodes
            for index, val in enumerate(batch_mean):
                QAOA_diff_list[index] *= -batch_mean[index]
                beta_diff_list[index] *= -batch_mean[index]

            # Compute parameter updates
            QAOA_diff_sum = np.mean(QAOA_diff_list, axis=0)
            beta_diff_sum = np.mean(beta_diff_list, axis=0)
            value_sum = np.mean(value_list)
            min_value = np.min(value_list)  # Find the lowest reward value
            min_index = np.argmin(value_list)  # Index of lowest reward value
            # Store values
            self.avg_values.append(value_sum)
            self.min_values.append(min_value)
            if correct_ans is not None:
                prob /= episodes
            self.prob_values.append(prob)
            self.best_states.append(state_list[min_index])
            self.best_same_lists.append(same_lists[min_index][:3])  # Store top 3 same list elements
            self.best_diff_lists.append(diff_lists[min_index][:3])  # Store top 3 diff list elements

            # Print optimization progress
            if j % 5 == 0:
                if correct_ans is not None:
                    print(f'  Probability of finding correct solution: {prob:.4f}')
                print(f'  Average reward: {value_sum}')
                print(f'  Lowest reward obtained: {min_value}')
                print(f'  Best state at lowest value: {self.best_states[-1]}')
                #print(f'  Top 3 same constraints: {self.best_same_lists[-1]}')
                #print(f'  Top 3 different constraints: {self.best_diff_lists[-1]}')


            # Update parameters using the Adam optimizer
            update = self.optimzer.get_updates([QAOA_diff_sum, beta_diff_sum])
            self.param += np.array(update[0])
            self.b += np.array(update[1])

    def rqaoa_execute(self, cal_grad=True):
        """
        Executes the RQAOA algorithm by iteratively reducing the QUBO problem.

        Parameters
        ----------
        cal_grad : bool, default=True
            Whether to calculate the gradient.

        Returns
        -------
        tuple or float
            If cal_grad is True, returns gradients, value, and final state.
            Otherwise, returns only the final value.
        """

        Q_init = copy.deepcopy(self.Q)
        Q_action = copy.deepcopy(self.Q)
        self.same_list = []
        self.diff_list = []
        self.node_assignments = {}
        self.edge_expectations = []
        self.edge_expectations_grad = []
        self.policys = []

        QAOA_diff_list = []
        beta_diff_list = []
        index = 0




        while Q_init.shape[0] > self.n_c:
            if self.b.ndim == 1:
                self.beta = self.b
            else:
                self.beta = self.b[index]


            if self.tree.state.value is None:
                edge_expectations = self._qaoa_edge_expectations(
                    Q_init, [i for i in range(self.p * index * 2, self.p * index * 2 + 2 * self.p)]
                )
                self.tree.state.value = edge_expectations
            else:
                edge_expectations = self.tree.state.value
            selected_edge_idx, policy, edge_res = self._select_edge_to_cut(Q_action, Q_init, edge_expectations)

            if cal_grad:
                """ edge_res_grad = self._qaoa_edge_expectations_gradient(
                    Q_init, [i for i in range(self.p * index * 2, self.p * index * 2 + 2 * self.p)], selected_edge_idx
                ) """

                if self.tree_grad.state.value is None:
                    edge_res_grad = self._qaoa_edge_expectations_gradients(
                        Q_init, [i for i in range(self.p * index * 2, self.p * index * 2 + 2 * self.p)]
                    )
                    self.tree_grad.state.value = edge_res_grad
                    self._tree_action(self.tree_grad, edge_expectations,selected_edge_idx,Q_init)

                else:
                    edge_res_grad = self.tree_grad.state.value
                    self._tree_action(self.tree_grad, edge_expectations,selected_edge_idx,Q_init)



                if self.lr[0] != 0:
                    QAOA_diff = self._compute_log_pol_diff(
                        selected_edge_idx, Q_action, edge_res, edge_res_grad, policy
                    ) * self.gamma ** (Q_init.shape[0] - index)

                else:
                    QAOA_diff = np.zeros_like(self.param)

                beta_diff = self._compute_grad_beta(selected_edge_idx, Q_action, policy, edge_res) * self.gamma ** (Q_init.shape[0] - index)
                QAOA_diff_list.append(QAOA_diff)
                beta_diff_list.append(beta_diff)

            Q_init, Q_action = self._cut_edge(selected_edge_idx, edge_res, Q_action, Q_init)
            index += 1

        self.tree.reset_state()
        self.tree_grad.reset_state()
        # Solve smaller problem using brute force
        self._brute_force_optimal(Q_init)
        Value = self._state_energy(np.array(self.node_assignments), self.Q)

        # Copy lists to preserve their state
        same_list_copy = copy.deepcopy(self.same_list)
        diff_list_copy = copy.deepcopy(self.diff_list)

        if self.n_c != self.Q.shape[0]:
            QAOA_diff = np.sum(QAOA_diff_list, axis=0)
        else:
            QAOA_diff = None
        if self.n_c != self.Q.shape[0]:
            if self.beta.ndim == 1:
                beta_diff = np.sum(beta_diff_list, axis=0)
            else:
                beta_diff = np.stack(beta_diff_list, axis=0)
        else:
            beta_diff = None



        # If gradient calculation is enabled, return additional data
        if cal_grad:
            return QAOA_diff, beta_diff, Value, np.array(self.node_assignments), same_list_copy, diff_list_copy
        else:
            return Value

    def _select_edge_to_cut(self, Q_action, Q_init, edge_expectations):
        """
        Selects an edge to be cut based on a softmax probability distribution over interactions.

        Parameters
        ----------
        Q_action : np.ndarray
            Current QUBO matrix tracking active nodes.

        Q_init : np.ndarray
            Initial QUBO matrix.

        edge_expectations : list
            Expectation values of ZZ interactions for all edges.

        Returns
        -------
        tuple
            Index of selected edge, probability distribution, expectation values.
        """
        action_space = self._action_space(Q_action)

        try:
            value = abs(np.array(edge_expectations))

            value = value - np.amax(value)
            interactions = abs(np.array(edge_expectations)) * self.beta[action_space]
            interactions -= np.amax(interactions)
        except:
            print(abs(np.array(edge_expectations)), self.b[action_space])
            raise ValueError("Invalid input", action_space, abs(np.array(edge_expectations)))
        interactions = np.exp(interactions)
        probabilities = interactions/np.sum(interactions)
        #probabilities = torch.softmax(torch.tensor(interactions), dim=0).numpy()
        selected_edge_idx = np.random.choice(len(probabilities), p=probabilities)

        return selected_edge_idx, probabilities, edge_expectations
    def _compute_log_pol_diff(self, idx, Q_action, edge_expectations, edge_expectations_grad, policy):
        """
        Computes the gradient of the log-policy for the selected edge.

        Parameters
        ----------
        idx : int
            Index of the selected edge.

        Q_action : np.ndarray
            QUBO matrix representing the current optimization problem.

        edge_expectations : list
            Expectation values of ZZ interactions for all edges.

        edge_expectations_grad : list
            Gradient of the expectation values of ZZ interactions for all edges.

        policy : list
            Probability distribution over edges for selection.

        Returns
        -------
        np.array
            The computed gradient of the log-policy.
        """
        action_space = self._action_space(Q_action)
        betas = self.beta[action_space]
        gather = np.zeros_like(policy)

        # Compute the weighted sum of policy and betas
        for i in range(len(edge_expectations_grad)):
            gather[i] += policy[i] * betas[i]

        diff_log_pol = betas[idx] * np.sign(edge_expectations[idx]) * edge_expectations_grad[idx]

        # Adjust the gradient with respect to policy values
        for i in range(len(gather)):
            if gather[i]:
                diff_log_pol -= gather[i] * np.sign(edge_expectations[i]) * edge_expectations_grad[i]

        return np.array(diff_log_pol)

    def _compute_log_pol_diff_idx(self, idx, Q_action, edge_expectations, grad, policy):
        """
        Computes the gradient of the log-policy for a given edge selection.

        Parameters
        ----------
        idx : int
            Index of the selected edge.

        Q_action : np.ndarray
            QUBO matrix representing the current optimization problem.

        edge_expectations : list
            Expectation values of ZZ interactions for all edges.

        grad : float
            Gradient value corresponding to the selected edge.

        policy : list
            Probability distribution over edges for selection.

        Returns
        -------
        np.array
            The computed gradient of the log-policy for the selected edge.
        """
        action_space = self._action_space(Q_action)
        betas = self.beta[action_space]

        diff_log_pol = betas[idx] * np.sign(edge_expectations[idx]) * grad - policy[idx] * betas[idx] * np.sign(edge_expectations[idx]) * grad
        return np.array(diff_log_pol)

    def _compute_grad_beta(self, idx, Q_action, policy, edge_expectations):
        """
        Computes the gradient of the beta parameter.

        Parameters
        ----------
        idx : int
            Index of the selected edge.

        Q_action : np.ndarray
            QUBO matrix representing the current optimization problem.

        policy : list
            Probability distribution over edges for selection.

        edge_expectations : list
            Expectation values of ZZ interactions for all edges.

        Returns
        -------
        np.array
            The computed gradient of the beta parameter.
        """
        abs_expectations = abs(np.array(edge_expectations))
        action_space = self._action_space(Q_action)

        betas_idx = action_space
        grad = np.zeros(len(self.beta))

        grad[betas_idx[idx]] += abs_expectations[idx]

        # Compute gradient by adjusting with policy values
        for i in range(len(action_space)):
            grad[betas_idx[i]] -= policy[i] * abs_expectations[i]

        return np.array(grad)

    def _cut_edge(self, selected_edge_idx, expectations, Q_action, Q_init):
        """
        Cuts the selected edge and returns the reduced QUBO matrix along with a matrix of the same size
        where the corresponding node values are set to zero.

        Parameters
        ----------
        selected_edge_idx : int
            Index of the selected edge to be cut.

        expectations : list
            Expectation values of ZZ interactions for all edges.

        Q_action : np.ndarray
            Current QUBO matrix tracking active nodes.

        Q_init : np.ndarray
            Initial QUBO matrix.

        Returns
        -------
        tuple
            Reduced QUBO matrix and an updated QUBO matrix with the selected nodes set to zero.
        """
        edge_list = [(i, j) for i in range(Q_init.shape[0]) for j in range(Q_init.shape[0]) if Q_init[i, j] != 0 and i != j]
        edge_to_cut = edge_list[selected_edge_idx]
        edge_to_cut = sorted(edge_to_cut)

        expectation = expectations[selected_edge_idx]

        i, j = edge_to_cut[0], edge_to_cut[1]

        for key in dict(sorted(self.node_assignments.items(), key=lambda item: item[0])):
            if i >= key:
                i += 1
            if j >= key:
                j += 1

        self.node_assignments[i] = 1

        new_Q, Q_action = reduce_hamiltonian(Q_init, edge_to_cut[0], edge_to_cut[1], self.node_assignments, int(np.sign(expectation)))
        if expectation > 0:
            self.same_list.append((i, j))
        else:
            self.diff_list.append((i, j))

        self._tree_action(self.tree, expectations, selected_edge_idx, Q_init)
        return new_Q, Q_action


    def _tree_action(self,tree, expectations,selected_edge_idx,Q_init):
        edge_list = [(i, j) for i in range(Q_init.shape[0]) for j in range(Q_init.shape[0]) if Q_init[i, j] != 0 and i != j]
        edge_to_cut = edge_list[selected_edge_idx]
        edge_to_cut = sorted(edge_to_cut)

        expectation = expectations[selected_edge_idx]

        i, j = edge_to_cut[0], edge_to_cut[1]

        for key in dict(sorted(self.node_assignments.items(), key=lambda item: item[0])):
            if i >= key:
                i += 1
            if j >= key:
                j += 1

        if expectation > 0:
            self.key = f'({i},{j})'
            if tree.has_child(self.key):
                tree.move(self.key)

            else:
                tree.create(self.key,None)
                tree.move(self.key)
        else:
            self.key = f'({-i},{-j})'
            if tree.has_child(self.key):
                tree.move(self.key)
            else:
                tree.create(self.key,None)
                tree.move(self.key)


    def _action_space(self, Q_action):
        """
        Maps the edges in the reduced graph to their original positions in the full graph.

        This function is used to track which edges in the reduced graph correspond to the original
        graph's edges after node elimination. When a node is removed, the edge indices in the
        reduced graph will shift, and this function helps maintain consistency with the original
        edge indexing.

        Example:
        --------
        Suppose the original graph has nodes [1, 2, 3, 4, 5] with edges:
            (1,2), (2,3), (3,4), (4,5)

        If node 3 is removed, the reduced graph has edges:
            (1,2), (4,5)

        The reduced graph will renumber nodes as:
            (1,2) -> (1,2), (4,5) -> (2,3)

        This function ensures the correct mapping to the original graph using `Q_action`.

        Parameters
        ----------
        Q_action : np.ndarray
            The original QUBO matrix with node elimination information, used to track active nodes.

        Returns
        -------
        list
            A list of indices indicating which edges in the reduced graph correspond to the original
            graph structure.
        """
        action_space_list = []
        index = 0  # Tracks the original edge indices

        for i in range(Q_action.shape[0]):
            for j in range(Q_action.shape[0]):
                if i != j:  # Avoid self-loops
                    if Q_action[i, j] != 0:  # Check if the edge exists in the original graph
                        action_space_list.append(index)  # Store the original edge index
                    index += 1  # Increment index for original edge mapping

        return action_space_list

    def _qaoa_edge_expectations(self, Q, idx):
        """
        Computes the expectation values of ZZ interactions for each edge in the given QUBO matrix.

        Parameters
        ----------
        Q : np.ndarray
            The QUBO matrix representing the optimization problem.

        idx : int
            Index for selecting the QAOA parameters.

        Returns
        -------
        list
            A list of expectation values for ZZ interactions of the edges in the QUBO matrix.
        """
        self.qaoa_layer = QAOA_layer(self.p, Q)

        @qml.qnode(self.qaoa_layer.dev)
        def circuit(param):
            self.qaoa_layer.qaoa_circuit(param)
            return [qml.expval(qml.PauliZ(i) @ qml.PauliZ(j))
                    for i in range(Q.shape[0])
                    for j in range(Q.shape[0])
                    if Q[i, j] != 0 and i != j]

        return circuit(self.param[idx])

    def _qaoa_edge_expectations_gradients(self, Q, idx):
        """
        Computes the gradients of the expectation values of ZZ interactions for each edge.

        Parameters
        ----------
        Q : np.ndarray
            The QUBO matrix representing the optimization problem.

        idx : int
            Index for selecting the QAOA parameters.

        Returns
        -------
        list
            A list of gradient values for the expectation values of ZZ interactions.
        """
        self.qaoa_layer = QAOA_layer(self.p, Q)
        res = []

        @qml.qnode(self.qaoa_layer.dev)
        def circuit(params, i, j):
            self.qaoa_layer.qaoa_circuit(params[idx])
            return qml.expval(qml.PauliZ(i) @ qml.PauliZ(j))

        # Compute gradients for each valid edge
        for i in range(Q.shape[0]):
            for j in range(Q.shape[0]):
                if Q[i, j] != 0 and i != j:
                    gradients = qml.grad(circuit)(self.param, i, j)
                    res.append(gradients)

        return res

    def _qaoa_edge_expectations_gradient(self, Q, idx, index):
        """
        Computes the gradient of a specific ZZ interaction expectation value for a selected edge.

        Parameters
        ----------
        Q : np.ndarray
            The QUBO matrix representing the optimization problem.

        idx : int
            Index for selecting the QAOA parameters.

        index : int
            The specific edge index to compute the gradient for.

        Returns
        -------
        float
            The gradient of the ZZ expectation value for the selected edge.
        """
        self.qaoa_layer = QAOA_layer(self.p, Q)
        number = 0

        @qml.qnode(self.qaoa_layer.dev)
        def circuit(params, i, j):
            self.qaoa_layer.qaoa_circuit(params[idx])
            return qml.expval(qml.PauliZ(i) @ qml.PauliZ(j))

        for i in range(Q.shape[0]):
            for j in range(Q.shape[0]):
                if Q[i, j] != 0 and i != j:
                    if number == index:
                        gradients = qml.grad(circuit)(self.param, i, j)
                        return gradients
                    number += 1

    def _brute_force_optimal(self, Q):
        """
        Finds the optimal solution using brute force when the graph size is small.

        Parameters
        ----------
        Q : np.ndarray
            The reduced QUBO matrix.

        Updates
        -------
        self.node_assignments : dict
            Stores the optimal node assignments obtained through brute-force search.
        """
        n = self.Q.shape[0]
        configs = list(itertools.product([-1, 1], repeat=n))
        best_value = np.inf
        res_node = None

        # Find all valid combinations considering the same and different constraints
        comb_list = get_case(self.same_list, self.diff_list,n)

        for comb in comb_list:
            value = self._state_energy(np.array(comb), self.Q)
            if value < best_value:
                best_value = value
                res_node = copy.copy(comb)

        # Store the optimal assignment
        self.node_assignments = res_node

    def _state_energy(self, state, Q):
        """
        Computes the energy of a given state based on the QUBO matrix.

        Parameters
        ----------
        state : np.ndarray
            Binary state vector (e.g. [-1, 1, -1, 1]).

        Q : np.ndarray
            The QUBO matrix representing the optimization problem.

        Returns
        -------
        float
            The computed energy value of the given state.
        """
        # Create an identity matrix of the same size
        identity_matrix = np.eye(Q.shape[0], dtype=bool)

        # Remove diagonal elements from the QUBO matrix to isolate interactions
        interaction = np.where(identity_matrix, 0, Q)
        diagonal_elements = np.diag(Q)

        # Compute the energy using the QUBO formulation
        value = diagonal_elements @ state + state.T @ interaction @ state
        return value


def cut_k_matrices_randomly(matrix, m, k):
    n = matrix.shape[0]
    if m >= n:
        raise ValueError("m must be smaller than the size of the matrix")

    selected_subsets = []
    remaining_indices = list(range(n))
    combinations_list = generate_combinations(n, m)
    random_indices = random.sample(range(len(combinations_list)), k)
    indices = [combinations_list[i] for i in random_indices]
    for indice in indices:
        matrix_copy = copy.deepcopy(matrix)
        remaining_indices = list(range(n))
        remaining_indices = sorted(set(remaining_indices) - set(indice))
        reduced_matrix = matrix_copy[np.ix_(remaining_indices, remaining_indices)]
        selected_subsets.append(reduced_matrix)

    return selected_subsets

def generate_combinations(n, m):
    return list(combinations(range(n), m))



def generate_upper_triangular_qubo(size, low=-10, high=10, integer=True, seed=None):
    """
    Generates an upper-triangular QUBO (Quadratic Unconstrained Binary Optimization) matrix.

    Args:
        size (int): The number of variables (size of the QUBO matrix).
        low (int/float): Minimum value of the random elements.
        high (int/float): Maximum value of the random elements.
        integer (bool): If True, generates integer values; otherwise, generates float values.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        np.ndarray: An upper-triangular QUBO matrix of the specified size.
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate random values for the upper triangular part including diagonal
    if integer:
        Q = np.random.randint(low, high, (size, size))
    else:
        Q = np.random.uniform(low, high, (size, size))

    # Keep only the upper triangle values (including diagonal), set lower triangle to zero
    Q = np.triu(Q)


    # Ensure diagonal values are positive (bias terms)
    np.fill_diagonal(Q,np.diagonal(Q))


    return Q

class AdamOptimizer:
    """
    Stochastic gradient descent optimizer using the Adam optimization algorithm.

    Note: All default values are based on the original Adam paper.

    Parameters
    ----------
    params : list
        A concatenated list containing coefs_ and intercepts_ in the MLP model.
        Used for initializing velocities and updating parameters.

    learning_rate_init : float, default=0.001
        The initial learning rate used to control the step size in updating the weights.

    beta_1 : float, default=0.9
        Exponential decay rate for estimates of the first moment vector, should be in [0, 1).

    beta_2 : float, default=0.999
        Exponential decay rate for estimates of the second moment vector, should be in [0, 1).

    epsilon : float, default=1e-8
        A small value to ensure numerical stability and avoid division by zero.

    amsgrad : bool, default=False
        Whether to use the AMSGrad variant of Adam.

    Attributes
    ----------
    learning_rate : float
        The current learning rate after applying bias correction.

    t : int
        The optimization step count (timestep).

    ms : list
        First moment vectors (moving average of gradients).

    vs : list
        Second moment vectors (moving average of squared gradients).

    max_vs : list
        Maximum of past squared gradients used in AMSGrad.

    References
    ----------
    Kingma, Diederik, and Jimmy Ba.
    "Adam: A method for stochastic optimization."
    arXiv preprint arXiv:1412.6980 (2014).
    """

    def __init__(self, params, learning_rate_init=0.001, beta_1=0.9,
                 beta_2=0.999, epsilon=1e-8, amsgrad=False):

        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        # Initialize learning rate as an array if provided, else a scalar
        if isinstance(learning_rate_init, float):
            self.learning_rate_init = np.ones(len(params)) * learning_rate_init
        else:
            self.learning_rate_init = np.array(learning_rate_init)

        self.t = 0  # Timestep initialization
        self.ms = [np.zeros_like(param) for param in params]  # First moment vector (m)
        self.vs = [np.zeros_like(param) for param in params]  # Second moment vector (v)
        self.amsgrad = amsgrad
        self.max_vs = [np.zeros_like(param) for param in params]  # For AMSGrad correction

    def get_updates(self, grads):
        """
        Computes the parameter updates based on the provided gradients.

        Parameters
        ----------
        grads : list
            Gradients with respect to coefs_ and intercepts_ in the model.

        Returns
        -------
        updates : list
            The values to be added to params for optimization.
        """
        self.t += 1  # Increment timestep

        # Update biased first moment estimate (m)
        self.ms = [self.beta_1 * m + (1 - self.beta_1) * grad
                   for m, grad in zip(self.ms, grads)]

        # Update biased second moment estimate (v)
        self.vs = [self.beta_2 * v + (1 - self.beta_2) * (grad ** 2)
                   for v, grad in zip(self.vs, grads)]

        # Update maximum second moment for AMSGrad if enabled
        self.max_vs = [np.maximum(v, max_v) for v, max_v in zip(self.vs, self.max_vs)]

        # Compute bias-corrected learning rate
        self.learning_rate = (self.learning_rate_init *
                              np.sqrt(1 - self.beta_2 ** self.t) /
                              (1 - self.beta_1 ** self.t))

        # Compute update step based on AMSGrad condition
        if self.amsgrad:
            updates = [lr * m / (np.sqrt(max_v) + self.epsilon)
                       for lr, m, max_v in zip(self.learning_rate, self.ms, self.max_vs)]
        else:
            updates = [lr * m / (np.sqrt(v) + self.epsilon)
                       for lr, m, v in zip(self.learning_rate, self.ms, self.vs)]

        return updates


class QAOA_layer:

    def __init__(self, depth, Q):
        """
        A class to represent a layer of a Quantum Approximate Optimization Algorithm (QAOA).

        Parameters
        ----------
        depth : int
            The number of QAOA layers (depth of the circuit).

        Q : np.ndarray
            The QUBO matrix representing the quadratic unconstrained binary optimization problem.

        Attributes
        ----------
        Q : np.ndarray
            The QUBO matrix.

        p : int
            The depth of the QAOA circuit.

        ham : qml.Hamiltonian
            The cost Hamiltonian for the given QUBO problem.

        dev : qml.device
            The quantum device used for simulation.

        """
        self.Q = Q  # Store the QUBO matrix
        self.p = depth  # Store the QAOA depth
        self.ham = self.prepare_cost_hamiltonian()  # Prepare the cost Hamiltonian based on QUBO matrix
        self.dev = qml.device("default.qubit", wires=Q.shape[0])  # Quantum device with qubits equal to Q size

    def qaoa_circuit(self, params):
        """
        Constructs the QAOA circuit based on given parameters.

        Parameters
        ----------
        params : list
            A list containing gamma and beta values for parameterized QAOA layers.
        """
        n = self.Q.shape[0]  # Number of qubits based on QUBO matrix size
        gammas = params[:self.p]  # Extract gamma parameters
        betas = params[self.p:]  # Extract beta parameters

        # Apply Hadamard gates to all qubits for uniform superposition
        for i in range(n):
            qml.Hadamard(wires=i)

        # Apply QAOA layers consisting of cost and mixer Hamiltonians
        for layer in range(self.p):
            self.qubo_cost(gammas[layer])
            self.mixer(betas[layer])

    def qubo_cost(self, gamma):
        """
        Implements the cost Hamiltonian evolution for the QUBO problem.

        Parameters
        ----------
        gamma : float
            Parameter for cost Hamiltonian evolution.
        """
        n = self.Q.shape[0]
        for i in range(n):
            for j in range(n):
                if self.Q[i, j] != 0:
                    if i == j:
                        qml.RZ(2 * gamma * float(self.Q[i, j]), wires=i)
                    else:
                        qml.CNOT(wires=[i, j])
                        qml.RZ(2 * gamma * float(self.Q[i, j]), wires=j)
                        qml.CNOT(wires=[i, j])

    def mixer(self, beta):
        """
        Implements the mixer Hamiltonian for QAOA.

        Parameters
        ----------
        beta : float
            Parameter for mixer Hamiltonian evolution.
        """
        for i in range(self.Q.shape[0]):
            qml.RX(2 * beta, wires=i)

    def prepare_cost_hamiltonian(self):
        """
        Constructs the cost Hamiltonian for the QUBO problem.

        Returns
        -------
        qml.Hamiltonian
            The constructed cost Hamiltonian.
        """
        n = self.Q.shape[0]
        coeffs = []
        ops = []

        for i in range(n):
            for j in range(n):
                if self.Q[i, j] != 0:
                    if i == j:
                        coeffs.append(self.Q[i, j])
                        ops.append(qml.PauliZ(i))
                    else:
                        coeffs.append(self.Q[i, j])
                        ops.append(qml.PauliZ(i) @ qml.PauliZ(j))

        return qml.Hamiltonian(coeffs, ops)



def add_zero_row_col(matrix, m):
    """
    Adds a new row and column filled with zeros at the specified position
    in an n x n matrix, resulting in an (n+1) x (n+1) matrix.

    Args:
        matrix (np.array): The original n x n matrix.
        m (int): The index (0-based) where the new row and column will be inserted.

    Returns:
        np.array: The expanded (n+1) x (n+1) matrix with the new row and column filled with zeros.
    """
    n = matrix.shape[0]  # Get the size of the original matrix

    # Create a new (n+1)x(n+1) matrix initialized with zeros
    new_matrix = np.zeros((n + 1, n + 1))

    # Copy the top-left submatrix (before row m and column m)
    new_matrix[:m, :m] = matrix[:m, :m]

    # Copy the top-right submatrix (after column m)
    new_matrix[:m, m+1:] = matrix[:m, m:]

    # Copy the bottom-left submatrix (after row m)
    new_matrix[m+1:, :m] = matrix[m:, :m]

    # Copy the bottom-right submatrix (after row m and column m)
    new_matrix[m+1:, m+1:] = matrix[m:, m:]

    # The new row (index m) and column (index m) remain zeros by default

    return new_matrix

import copy
import random
from itertools import combinations
import pennylane as qml

class QAOA_pretrain:
    """
    A class to pre-train QAOA parameters by averaging expectation values
    over randomly sampled submatrices from the given QUBO matrix.

    Parameters
    ----------
    Q : np.ndarray
        The full QUBO matrix representing the optimization problem.

    size : int
        The size of each randomly selected submatrix.

    depth : int
        The depth of the QAOA circuit (number of layers).

    number : int
        The number of submatrices to generate for averaging.

    Methods
    -------
    qaoa_exp(param)
        Computes the average expectation value of the QAOA circuit over randomly
        selected submatrices.

    cost_function(params)
        Computes the cost function based on the averaged expectation value.
    """

    def __init__(self, Q, size, depth, number):
        self.Q = Q  # Full QUBO matrix
        self.size = size  # Size of submatrices
        self.p = depth  # Depth of QAOA circuit
        self.number = number  # Number of submatrices to sample

    def qaoa_exp(self, param):
        """
        Computes the average expectation value of the QAOA circuit over randomly selected submatrices.

        Parameters
        ----------
        param : np.ndarray
            Array of QAOA parameters (gammas and betas).

        Returns
        -------
        float
            The averaged expectation value over selected submatrices.
        """
        Q_action = copy.deepcopy(self.Q)

        # Generate k random submatrices of size `size` from QUBO matrix
        Q_list = cut_k_matrices_randomly(Q_action, self.size, self.number)

        res = 0
        for Q in Q_list:
            qaoa = QAOA_layer(self.p, Q)

            @qml.qnode(qaoa.dev)
            def qaoa_expectation(param):
                """
                Quantum node that runs the QAOA circuit and measures the expectation value
                of the cost Hamiltonian.

                Parameters
                ----------
                param : np.ndarray
                    Array of QAOA parameters (gammas and betas).

                Returns
                -------
                float
                    Expectation value of the cost Hamiltonian.
                """
                qaoa.qaoa_circuit(param)
                return qml.expval(qaoa.ham)

            # Accumulate the expectation values from submatrices
            res += qaoa_expectation(param)

        return res / len(Q_list)  # Return the average expectation value

    def cost_function(self, params):
        """
        Computes the expectation value of the cost Hamiltonian to be used as the cost function.

        Parameters
        ----------
        params : np.ndarray
            Array of QAOA parameters (gammas and betas).

        Returns
        -------
        float
            Expectation value of the cost Hamiltonian (to be minimized).
        """
        return self.qaoa_exp(params)


def cut_k_matrices_randomly(matrix, m, k):
    """
    Randomly selects `k` submatrices of size `m x m` from the given `n x n` matrix.

    Parameters
    ----------
    matrix : np.ndarray
        The original QUBO matrix.

    m : int
        The size of the submatrices to be extracted.

    k : int
        The number of submatrices to be extracted.

    Returns
    -------
    list of np.ndarray
        A list containing `k` submatrices of size `m x m`.

    Raises
    ------
    ValueError
        If the desired submatrix size `m` is larger than the original matrix size `n`.
    """
    n = matrix.shape[0]

    if m >= n:
        raise ValueError("The submatrix size 'm' must be smaller than the original matrix size.")

    selected_subsets = []
    remaining_indices = list(range(n))

    # Generate all possible combinations of selecting `m` nodes out of `n`
    combinations_list = generate_combinations(n, m)

    # Randomly sample `k` combinations
    random_indices = random.sample(range(len(combinations_list)), k)
    indices = [combinations_list[i] for i in random_indices]

    for indice in indices:
        matrix_copy = copy.deepcopy(matrix)
        remaining_indices = list(range(n))

        # Remove the selected indices to extract the submatrix
        remaining_indices = sorted(set(remaining_indices) - set(indice))

        # Create a submatrix by selecting only the remaining indices
        reduced_matrix = matrix_copy[np.ix_(remaining_indices, remaining_indices)]
        selected_subsets.append(reduced_matrix)

    return selected_subsets


def generate_combinations(n, m):
    """
    Generates all possible combinations of selecting `m` elements from `n` elements.

    Parameters
    ----------
    n : int
        The total number of elements.

    m : int
        The number of elements to select.

    Returns
    -------
    list of tuples
        A list of all possible combinations of size `m` from `n` elements.
    """
    return list(combinations(range(n), m))


def zero_lower_triangle(matrix):
    """
    Set the lower triangular elements (below the diagonal) of a given numpy matrix to zero.

    Args:
        matrix (np.array): The input matrix.

    Returns:
        np.array: A matrix with the lower triangular elements set to zero.
    """
    result = np.copy(matrix)
    rows, cols = result.shape

    for i in range(rows):
        for j in range(i):
            result[i, j] = 0

    return result

def reduce_hamiltonian(J, k, l, node_assignments, sign):
    """
    Reduces the given Hamiltonian matrix by applying the constraint Z_k = sign * Z_l.

    Args:
        J (np.array): The initial Hamiltonian matrix (including diagonal terms).
        k (int): Index of the variable to be removed.
        l (int): Index of the variable to be replaced.
        node_assignments (dict): Dictionary mapping node indices to assignments.
        sign (int): Relationship (1 if identical, -1 if opposite).

    Returns:
        tuple:
            - np.array: The reduced Hamiltonian matrix with the k-th variable removed.
            - np.array: An expanded version of the reduced matrix with extra rows and columns added back.
    """
    # Update interactions: J[i, l] = J[i, l] + sign * J[i, k]
    for i in range(J.shape[0]):
        if i != k and i != l:
            J[i, l] += sign * J[i, k]  # Update row
            J[l, i] += sign * J[k, i]  # Update column

    # Update diagonal elements (self-interaction term)
    J[l, l] = sign * J[k, k] + J[l, l]

    # Zero out lower triangular elements to maintain upper triangular form
    J = zero_lower_triangle(J)

    # Sort keys for correct row/column addition
    key_list = sorted(node_assignments.keys())

    # Set the removed row and column to zero before deletion
    J[:, k] = 0
    J[k, :] = 0

    # Remove the k-th row and column
    J = np.delete(J, k, axis=0)
    J = np.delete(J, k, axis=1)

    # Create a copy for expansion
    R = copy.deepcopy(J)

    # Add back zero rows and columns at specified indices
    for key in key_list:
        R = add_zero_row_col(R, key)

    return J, R

def add_zero_row_col(matrix, m):
    """
    Adds a new row and column filled with zeros at the specified position
    in an n x n matrix, resulting in an (n+1) x (n+1) matrix.

    Args:
        matrix (np.array): The original n x n matrix.
        m (int): The index (0-based) where the new row and column will be inserted.

    Returns:
        np.array: The expanded (n+1) x (n+1) matrix with the new row and column filled with zeros.
    """
    n = matrix.shape[0]

    # Create a new (n+1)x(n+1) matrix initialized with zeros
    new_matrix = np.zeros((n + 1, n + 1))

    # Copy the existing elements to the new matrix
    new_matrix[:m, :m] = matrix[:m, :m]      # Top-left block
    new_matrix[:m, m+1:] = matrix[:m, m:]    # Top-right block
    new_matrix[m+1:, :m] = matrix[m:, :m]    # Bottom-left block
    new_matrix[m+1:, m+1:] = matrix[m:, m:]  # Bottom-right block

    return new_matrix

def signed_softmax_rewards(rewards, beta=15.0):
    """
    Apply softmax transformation to absolute values of rewards
    while preserving their original sign.

    Args:
        rewards (np.ndarray): Array of reward values.
        beta (float): Temperature parameter to control sharpness of softmax.

    Returns:
        np.ndarray: Transformed reward values with preserved sign.
    """
    rewards = np.array(rewards)

    # Step 1: Compute absolute values and apply softmax
    abs_rewards = np.abs(rewards)
    scaled_rewards = beta * abs_rewards
    exp_rewards = np.exp(scaled_rewards - np.max(scaled_rewards))  # Numerical stability
    softmax_vals = exp_rewards / np.sum(exp_rewards)

    # Step 2: Restore original sign
    signed_rewards = np.sign(rewards) * softmax_vals
    return signed_rewards


import copy

class Edge:
    """
    Represents an edge in the graph connecting nodes with a "same" or "different" condition.

    Attributes:
        value (int): The target node index.
        same (bool): True if nodes must have the same value, False if they must have different values.
    """
    def __init__(self, edge_type, value):
        self.value = value
        self.same = edge_type == "same"  # True if 'same', False if 'diff'

def dfs(node, node_number, edge_list, tmp_list, result_list):
    """
    Depth-First Search (DFS) function to explore valid assignments of values to nodes
    based on the given same/different constraints.

    Args:
        node (int): Current node index being processed.
        node_number (int): Total number of nodes.
        edge_list (dict): Adjacency list representation of constraints.
        tmp_list (list): Temporary list storing current node assignments.
        result_list (list): List to store valid assignment combinations.

    Returns:
        None
    """
    if node == node_number:
        result_list.append(copy.deepcopy(tmp_list))
        return

    if len(edge_list[node]) == 0:
        # Assign possible values (-1 or 1) and continue exploring
        tmp_list[node] = 1
        dfs(node + 1, node_number, edge_list, tmp_list, result_list)
        tmp_list[node] = -1
        dfs(node + 1, node_number, edge_list, tmp_list, result_list)
        return

    tmp_list[node] = 0  # Reset node value
    for e in edge_list[node]:
        if e.same:
            res = tmp_list[e.value]
        else:
            res = -1 * tmp_list[e.value]

        # Conflict check
        if tmp_list[node] != 0 and tmp_list[node] != res:
            return

        tmp_list[node] = res

    dfs(node + 1, node_number, edge_list, tmp_list, result_list)
    return

def get_case(same_list, diff_list, node_number):
    """
    Generates all possible assignments of values to nodes that satisfy given same/different constraints.

    Args:
        same_list (list of tuples): List of node pairs that must have the same value.
        diff_list (list of tuples): List of node pairs that must have different values.
        node_number (int): Total number of nodes.

    Returns:
        list: A list of valid node value assignments satisfying all constraints.
    """
    edge_list = {i: [] for i in range(node_number)}

    # Create "same" edges (bi-directional relationships)
    for e in same_list:
        if e[0] < e[1]:
            edge_list[e[1]].append(Edge('same', e[0]))
        else:
            edge_list[e[0]].append(Edge('same', e[1]))

    # Create "different" edges (bi-directional relationships)
    for e in diff_list:
        if e[0] < e[1]:
            edge_list[e[1]].append(Edge('diff', e[0]))
        else:
            edge_list[e[0]].append(Edge('diff', e[1]))

    # Initialize temporary storage and results list
    tmp_list = [0] * node_number
    result_list = []

    # Start DFS traversal
    dfs(0, node_number, edge_list, tmp_list, result_list)

    return result_list

