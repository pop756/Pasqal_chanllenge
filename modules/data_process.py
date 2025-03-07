from pennylane import numpy as np
import numpy as npo
import matplotlib.pyplot as plt
import copy

def data_to_QUBO(matrix,hamming_weight,lamb,relative_diff = None):
    if relative_diff is None:
        return -np.diag([1]+[1]*(len(matrix)-1))+matrix/hamming_weight*lamb
    else:
        return -np.diag(relative_diff)+matrix/hamming_weight*lamb


def qubo_to_ising(Q):
    """
    0/1 basis의 QUBO matrix Q (대칭 np.array)를 받아,
    -1/1 basis (Ising)로 변환하여 linear term h와 quadratic interaction matrix J를 반환합니다.
    
    x_i = (1 - z_i) / 2 변환을 이용.
    
    변환된 Ising Hamiltonian:
       H(z) = constant + sum_i h_i * z_i + sum_{i<j} J_{ij} * z_i * z_j.
    
    Parameters:
        Q (np.array): QUBO matrix (대칭, diagonal에 1-body term, off-diagonal에 2-body interaction)

    Returns:
        h (np.array): Ising 모델의 선형 항 계수 (길이 n)
        J (np.array): Ising 모델의 상호작용 계수 (n x n, 대칭, diagonal은 0)
    """
    n = Q.shape[0]
    h = np.zeros(n)
    J = np.zeros((n, n))
    # i<j에 대해 quadratic interaction term 계산
    for i in range(n):
        for j in range(n):
          if i!=j:
            h[i] -= Q[i, j] / 4.0
            h[j] -= Q[i, j] /4
            J[i, j] = Q[i, j] / 4.0

    for i in range(n):
        h[i] -= Q[i, i] / 2.0
    Q_res = J
    np.fill_diagonal(Q_res, h)
    return Q_res


def ising_to_qubo(ising_matrix):
    """
    for-loop를 이용하여 Ising matrix (h_i와 J_{ij})를 QUBO matrix로 변환.
    
    Args:
        ising_matrix (np.ndarray): n x n 대칭 행렬
    Returns:
        np.ndarray: 변환된 QUBO 행렬
    """
    n = ising_matrix.shape[0]
    Q = np.zeros_like(ising_matrix)
    
    # off-diagonal 항: Q_{ij} = 4 * J_{ij} (i != j)
    for i in range(n):
        for j in range(n):
            if i != j:
                Q[i, j] = 4 * ising_matrix[i, j]
                Q[i,i] -= 2 * ising_matrix[i, j]
                Q[j,j] -= 2 * ising_matrix[i, j]
    # 대각 원소: Q_{ii} = 2*h_i - 2 * sum_{j != i} J_{ij}
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
    list_seq = copy.deepcopy(list_seq)
    full_list = []
    add_comp = None
    for seq in list_seq:
        if len(full_list) == 0:
            full_list =[seq]
        else:
            break_comp = False
            seq_test = abs(npo.array(seq))
            for list_comp in full_list:
                if break_comp:
                    break
                for index,comp in enumerate(list_comp):
                    if break_comp:
                        break
                    for test_idx,test in enumerate(seq_test):
                        if test == abs(comp):
                            test = seq.pop(test_idx)
                            add_comp = npo.sign(test)*npo.sign(comp)*npo.array(seq)
                            break_comp = True
                            list_comp+=list(add_comp)
                            break
                                
                
                
                
            if add_comp is None and len(seq) >= 1:
                full_list.append(seq)
            add_comp = None
    return full_list

def make_node_weights(full_list):
    node_weights = []
    hamming_weights_default = 0
    for seq in full_list:
        node_weights.append(np.sum(np.array(seq) >= 0) - np.sum(np.array(seq) < 0))
        hamming_weights_default += np.sum(np.array(seq) < 0)
    return node_weights, hamming_weights_default





def off_diagonal_median(matrix):
    matrix = np.array(matrix)  # 리스트 입력 가능하도록 변환
    matrix = (matrix + matrix.T)/2
    rows, cols = matrix.shape


    # 대각 성분을 제외한 값들만 선택
    mask = ~np.eye(rows, dtype=bool)
    off_diagonal_values = matrix[mask]

    # 비대각 성분의 중앙값 계산
    return np.median(off_diagonal_values)*2

def plot_rl_qaoa_results(avg_values, min_values, prob_values,lable = "start"):
    """
    Plots the stored training values of RL_QAOA across epochs with margin.

    - The first plot shows the average values over epochs with margin.
    - The second plot displays the minimum values over epochs with margin.
    - The third plot represents the probability of finding the correct solution.

    Parameters
    ----------
    avg_values : list
        List of average values over epochs.
    min_values : list
        List of minimum values over epochs.
    prob_values : list
        List of probabilities over epochs.

    The x-axis represents the number of epochs in all graphs.
    """

    epochs = range(1, len(avg_values) + 1)

    # Plot average values with margin
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, avg_values, label=f"{lable} start Average Reward", color='blue')
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.title(f"{lable} start Average Reward")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot minimum values with margin
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, min_values, label=f"{lable} start Minimum Reward", color='red')
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot probability values
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, prob_values, label=f"{lable} start Probability", color='green')
    plt.xlabel("Epochs")
    plt.ylabel("Probability")
    plt.title(f"{lable} start Probability of Finding Correct Solution")
    plt.ylim(0, 1)  # Setting the y-axis range between 0 and 1
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_bitstring_counts(input_data, bitstring_counts,label,hamming_weight = None,node_weights = 1):
    """
    입력된 bitstring 목록을 기반으로 주어진 bitstring_counts에서 해당 bitstring을
    빨간색 막대로 강조하고, 나머지는 파란색으로 표시하는 그래프를 생성합니다.

    :param input_data: List of tuples (bitstring, value)
    :param bitstring_counts: Dictionary (bitstring -> count)
    """
    import matplotlib.pyplot as plt

    # 빨간색으로 강조할 bitstring 목록 추출
    highlighted_strings = {bitstring for bitstring, _ in input_data}
    bit_str = {}
    # 데이터 준비
    for key in bitstring_counts.keys():
        count = 0
        list_value = []
        for bit in key:
            list_value+=[int(bit)]
        if hamming_weight is None:
            bit_str[key] = bitstring_counts[key]
        else:
            try:
                count = np.array(list_value)@np.array(node_weights)
            except:
                count = np.sum(np.array(list_value)*np.array(node_weights))
            if  count == hamming_weight:
                bit_str[key] = bitstring_counts[key]

    bitstrings = list(bit_str.keys())
    counts = np.array(list(bit_str.values()))/np.sum(list(bitstring_counts.values()))
    colors = ['red' if bitstring in highlighted_strings else 'blue' for bitstring in bitstrings]
    print(f'pass prob : {np.sum(list(bit_str.values()))/np.sum(list(bitstring_counts.values()))}')
    # 막대 그래프 그리기
    plt.figure(figsize=(10, 5))
    plt.bar(bitstrings, counts, color=colors)
    plt.xlabel("Bitstrings")
    plt.ylabel("Counts")
    plt.title(label)
    plt.xticks(rotation=90)
    plt.show()