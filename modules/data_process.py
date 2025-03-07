from pennylane import numpy as np
import numpy as npo
import matplotlib.pyplot as plt
import copy
import numpy as np

def data_to_QUBO(matrix, hamming_weight, lamb, relative_diff=None):
    if relative_diff is None:
        return -np.diag([1] + [1] * (len(matrix) - 1)) + matrix / hamming_weight * lamb
    else:
        return -np.diag(relative_diff) + matrix / hamming_weight * lamb

def qubo_to_ising(Q):
    """
    Converts a QUBO matrix Q (symmetric np.array) in 0/1 basis
    to an Ising model in -1/1 basis, returning the linear term h
    and quadratic interaction matrix J.
    
    Using the transformation: x_i = (1 - z_i) / 2
    
    The transformed Ising Hamiltonian:
       H(z) = constant + sum_i h_i * z_i + sum_{i<j} J_{ij} * z_i * z_j.
    
    Parameters:
        Q (np.array): QUBO matrix (symmetric, diagonal elements represent 1-body terms,
                      off-diagonal elements represent 2-body interactions)
    
    Returns:
        h (np.array): Coefficients of linear terms in the Ising model (length n)
        J (np.array): Coefficients of interaction terms in the Ising model (n x n,
                      symmetric, diagonal elements are zero)
    """
    n = Q.shape[0]
    h = np.zeros(n)
    J = np.zeros((n, n))
    
    # Compute quadratic interaction terms for i < j
    for i in range(n):
        for j in range(n):
            if i != j:
                h[i] -= Q[i, j] / 4.0
                h[j] -= Q[i, j] / 4.0
                J[i, j] = Q[i, j] / 4.0
    
    # Compute linear terms
    for i in range(n):
        h[i] -= Q[i, i] / 2.0
    
    Q_res = J
    np.fill_diagonal(Q_res, h)
    return Q_res

def ising_to_qubo(ising_matrix):
    """
    Converts an Ising matrix (h_i and J_{ij}) into a QUBO matrix using a for-loop.
    
    Args:
        ising_matrix (np.ndarray): Symmetric n x n matrix
    
    Returns:
        np.ndarray: Transformed QUBO matrix
    """
    n = ising_matrix.shape[0]
    Q = np.zeros_like(ising_matrix)
    
    # Off-diagonal elements: Q_{ij} = 4 * J_{ij} (for i != j)
    for i in range(n):
        for j in range(n):
            if i != j:
                Q[i, j] = 4 * ising_matrix[i, j]
                Q[i, i] -= 2 * ising_matrix[i, j]
                Q[j, j] -= 2 * ising_matrix[i, j]
    
    # Diagonal elements: Q_{ii} = 2*h_i - 2 * sum_{j != i} J_{ij}
    for i in range(n):
        h_i = ising_matrix[i, i]
        Q[i, i] -= 2 * h_i 
    
    return Q


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
        for j in range(i+1,cols):
            result[i, j] += result[j, i]

    for i in range(rows):
        for j in range(i):
            result[i, j] = 0

    return result

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
def add_constraint(node_hamming_weights, hamming_weights):
    """
    Adds a Hamming weight constraint to the QUBO formulation.
    
    Args:
        node_hamming_weights (list or np.array): Weights for each node.
        hamming_weights (float): Total Hamming weight constraint.
    
    Returns:
        np.array: QUBO matrix with Hamming weight constraint applied.
    """
    size = len(node_hamming_weights)
    Q = npo.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if i == j:
                Q[i, j] = node_hamming_weights[i]**2 - 2*hamming_weights*node_hamming_weights[i]
            else:
                Q[i, j] = node_hamming_weights[i]*node_hamming_weights[j]
    
    return Q

def make_check(list_seq):
    """
    Processes the compressed Hamiltonian structure from RQAOA.
    Given the reduction process in Hamiltonian (e.g., Z1=-Z2=Z3),
    it generates the compressed values representing node relationships (e.g., Z1=-Z2, Z2=Z3).
    
    Args:
        list_seq (list of lists): Compressed Hamiltonian sequences.
    
    Returns:
        list of lists: Processed compressed node relationships.
    """
    list_seq = copy.deepcopy(list_seq)
    full_list = []
    add_comp = None
    for seq in list_seq:
        if len(full_list) == 0:
            full_list = [seq]
        else:
            break_comp = False
            seq_test = abs(npo.array(seq))
            for list_comp in full_list:
                if break_comp:
                    break
                for index, comp in enumerate(list_comp):
                    if break_comp:
                        break
                    for test_idx, test in enumerate(seq_test):
                        if test == abs(comp):
                            test = seq.pop(test_idx)
                            add_comp = npo.sign(test) * npo.sign(comp) * npo.array(seq)
                            break_comp = True
                            list_comp += list(add_comp)
                            break
    
        if add_comp is None and len(seq) >= 1:
            full_list.append(seq)
        add_comp = None
    return full_list

def make_node_weights(full_list):
    """
    Computes the Hamming weight adjustments for each node
    based on the compressed Hamiltonian structure.
    
    Args:
        full_list (list of lists): Compressed node relationships.
    
    Returns:
        tuple: (list of node weights, default Hamming weight sum)
    """
    node_weights = []
    hamming_weights_default = 0
    for seq in full_list:
        node_weights.append(np.sum(np.array(seq) >= 0) - np.sum(np.array(seq) < 0))
        hamming_weights_default += np.sum(np.array(seq) < 0)
    return node_weights, hamming_weights_default

def off_diagonal_median(matrix):
    """
    Computes twice the median of the off-diagonal elements in a symmetric matrix.
    
    Args:
        matrix (list or np.ndarray): Input matrix.
    
    Returns:
        float: Twice the median of off-diagonal elements.
    """
    matrix = np.array(matrix)  # Convert input to numpy array for flexibility
    matrix = (matrix + matrix.T) / 2  # Ensure symmetry
    rows, cols = matrix.shape
    
    # Select only off-diagonal elements
    mask = ~np.eye(rows, dtype=bool)
    off_diagonal_values = matrix[mask]
    
    # Compute and return twice the median of off-diagonal elements
    return np.median(off_diagonal_values) * 2


def add_constraint(node_hamming_weights, hamming_weights):
    """
    Adds a Hamming weight constraint to the QUBO formulation.
    
    Args:
        node_hamming_weights (list or np.array): Weights for each node.
        hamming_weights (float): Total Hamming weight constraint.
    
    Returns:
        np.array: QUBO matrix with Hamming weight constraint applied.
    """
    size = len(node_hamming_weights)
    Q = npo.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if i == j:
                Q[i, j] = node_hamming_weights[i]**2 - 2*hamming_weights*node_hamming_weights[i]
            else:
                Q[i, j] = node_hamming_weights[i]*node_hamming_weights[j]
    
    return Q

def make_check(list_seq):
    """
    Processes the compressed Hamiltonian structure from RQAOA.
    Given the reduction process in Hamiltonian (e.g., Z1=-Z2=Z3),
    it generates the compressed values representing node relationships (e.g., Z1=-Z2, Z2=Z3).
    
    Args:
        list_seq (list of lists): Compressed Hamiltonian sequences.
    
    Returns:
        list of lists: Processed compressed node relationships.
    """
    list_seq = copy.deepcopy(list_seq)
    full_list = []
    add_comp = None
    for seq in list_seq:
        if len(full_list) == 0:
            full_list = [seq]
        else:
            break_comp = False
            seq_test = abs(npo.array(seq))
            for list_comp in full_list:
                if break_comp:
                    break
                for index, comp in enumerate(list_comp):
                    if break_comp:
                        break
                    for test_idx, test in enumerate(seq_test):
                        if test == abs(comp):
                            test = seq.pop(test_idx)
                            add_comp = npo.sign(test) * npo.sign(comp) * npo.array(seq)
                            break_comp = True
                            list_comp += list(add_comp)
                            break
    
        if add_comp is None and len(seq) >= 1:
            full_list.append(seq)
        add_comp = None
    return full_list

def make_node_weights(full_list):
    """
    Computes the Hamming weight adjustments for each node
    based on the compressed Hamiltonian structure.
    
    Args:
        full_list (list of lists): Compressed node relationships.
    
    Returns:
        tuple: (list of node weights, default Hamming weight sum)
    """
    node_weights = []
    hamming_weights_default = 0
    for seq in full_list:
        node_weights.append(np.sum(np.array(seq) >= 0) - np.sum(np.array(seq) < 0))
        hamming_weights_default += np.sum(np.array(seq) < 0)
    return node_weights, hamming_weights_default

def off_diagonal_median(matrix):
    """
    Computes twice the median of the off-diagonal elements in a symmetric matrix.
    
    Args:
        matrix (list or np.ndarray): Input matrix.
    
    Returns:
        float: Twice the median of off-diagonal elements.
    """
    matrix = np.array(matrix)  # Convert input to numpy array for flexibility
    matrix = (matrix + matrix.T) / 2  # Ensure symmetry
    rows, cols = matrix.shape
    
    # Select only off-diagonal elements
    mask = ~np.eye(rows, dtype=bool)
    off_diagonal_values = matrix[mask]
    
    # Compute and return twice the median of off-diagonal elements
    return np.median(off_diagonal_values) * 2

def plot_rl_qaoa_results(avg_values, min_values, prob_values, label="start"):
    """
    Plots the training values of RL_QAOA over epochs with margins.
    
    - The first plot shows the average values over epochs.
    - The second plot displays the minimum values over epochs.
    - The third plot represents the probability of finding the correct solution.
    
    Args:
        avg_values (list): List of average values over epochs.
        min_values (list): List of minimum values over epochs.
        prob_values (list): List of probabilities over epochs.
        label (str): Label for the plots.
    """
    epochs = range(1, len(avg_values) + 1)
    
    # Plot average values
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, avg_values, label=f"{label} start Average Reward", color='blue')
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.title(f"{label} start Average Reward")
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # Plot minimum values
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, min_values, label=f"{label} start Minimum Reward", color='red')
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # Plot probability values
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, prob_values, label=f"{label} start Probability", color='green')
    plt.xlabel("Epochs")
    plt.ylabel("Probability")
    plt.title(f"{label} start Probability of Finding Correct Solution")
    plt.ylim(0, 1)  # Set y-axis range between 0 and 1
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_bitstring_counts(input_data, bitstring_counts, label, hamming_weight=None, node_weights=1):
    """
    Plots a histogram of bitstring counts.
    
    - Highlights specific bitstrings from input_data in red.
    - Other bitstrings are shown in blue.
    
    Args:
        input_data (list of tuples): List of (bitstring, value) pairs.
        bitstring_counts (dict): Dictionary mapping bitstrings to counts.
        label (str): Plot title.
        hamming_weight (int, optional): Hamming weight constraint.
        node_weights (int or list, optional): Weights for nodes.
    """
    # Extract highlighted bitstrings
    highlighted_strings = {bitstring for bitstring, _ in input_data}
    bit_str = {}
    
    # Prepare data
    for key in bitstring_counts.keys():
        count = 0
        list_value = [int(bit) for bit in key]
        if hamming_weight is None:
            bit_str[key] = bitstring_counts[key]
        else:
            count = np.dot(np.array(list_value), np.array(node_weights))
            if count == hamming_weight:
                bit_str[key] = bitstring_counts[key]
    
    bitstrings = list(bit_str.keys())
    counts = np.array(list(bit_str.values())) / np.sum(list(bitstring_counts.values()))
    colors = ['red' if bitstring in highlighted_strings else 'blue' for bitstring in bitstrings]
    print(f'pass prob : {np.sum(list(bit_str.values())) / np.sum(list(bitstring_counts.values()))}')
    
    # Plot histogram
    plt.figure(figsize=(10, 5))
    plt.bar(bitstrings, counts, color=colors)
    plt.xlabel("Bitstrings")
    plt.ylabel("Counts")
    plt.title(label)
    plt.xticks(rotation=90)
    plt.show()
