import copy
import pennylane as qml
import pulser
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from pulser import Pulse, Sequence, Register
from pulser.devices import MockDevice as DigitalAnalogDevice
from pulser.waveforms import InterpolatedWaveform
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from modules.data_process import zero_lower_triangle,qubo_to_ising,off_diagonal_median




class Pulse_simulation:
    """
    A class that simulates quantum pulse evolution based on an input QUBO matrix.
    
    Given an arbitrary QUBO matrix, this class:
    - Generates pulse waveforms for amplitude and detuning.
    - Applies interpolation to create continuous pulse functions.
    - Simulates the quantum state evolution over a given duration.
    """
    
    def __init__(self, Q, amplitude, detuning, duration, step_time=50):
        """
        Initializes the simulation with the given QUBO matrix and pulse parameters.
        
        Args:
            Q (np.ndarray): The QUBO matrix.
            amplitude (list): List of amplitude values for the pulse.
            detuning (list): List of detuning values for the pulse.
            duration (int): Total pulse duration.
            step_time (int): Time step interval.
        """
        self.amplitude = amplitude
        self.detuning = detuning
        self.duration = duration
        self.step_time = step_time
        
        # Generate interpolation points
        self.points = np.linspace(0, 1, int(duration / step_time))
        self.points = (self.points[:-1] + self.points[1:]) / 2
        
        # Convert QUBO to Ising model
        Q_copy = copy.deepcopy(Q)
        np.fill_diagonal(Q_copy, 0)
        self.Q_ising = zero_lower_triangle(qubo_to_ising(Q_copy / 2))
        
        # Generate Hamiltonians
        self.generate_hamiltonians()
    
    def generate_hamiltonians(self):
        """
        Generates a list of Hamiltonians based on the interpolated pulse values.
        """
        coeffs_ZZ, ops_ZZ, coeffs_Z, ops_Z = Q_to_ham(self.Q_ising)
        amp, detune = self.interpolate_1d()
        self.ham = []
        for time in range(len(amp[0])):
            coeffs = list(coeffs_ZZ)
            ops = list(ops_ZZ)
            for q_index in range(len(amp)):
                coeffs.append(amp[q_index][time] / 2)
                ops.append(qml.PauliX(q_index))
                coeffs.append(detune[q_index][time] / 2 + coeffs_Z[q_index])
                ops.append(qml.PauliZ(q_index))
            self.ham.append(qml.Hamiltonian(coeffs, ops))
    
    def simulate_time_evolution(self):
        """
        Simulates the quantum state evolution by applying the Hamiltonians sequentially.
        """
        dev = qml.device("default.qubit", wires=len(self.amplitude))
        
        @qml.qnode(dev)
        def circuit():
            for H in self.ham:
                qml.ApproxTimeEvolution(H, self.step_time / 1000, 1)
            return [qml.expval(qml.PauliZ(i)) for i in range(len(self.amplitude))]
        
        return circuit()
    
    def interpolate_1d(self):
        """
        Performs 1D interpolation on amplitude and detuning values.
        """
        amp_list = []
        detune_list = []
        for i in range(len(self.amplitude)):
            amp = interp.PchipInterpolator(np.linspace(0, 1, len(self.amplitude[i])), self.amplitude[i])
            detune = interp.PchipInterpolator(np.linspace(0, 1, len(self.detuning[i])), self.detuning[i])
            amp_list.append(amp(self.points))
            detune_list.append(detune(self.points))
        return amp_list, detune_list
    
    def draw(self):
        """
        Visualizes the quantum pulse sequence.
        """
        reg = create_square_register(len(self.amplitude))
        seq_temp = Sequence(reg, DigitalAnalogDevice)
        
        for i in range(len(self.amplitude)):
            pulse = Pulse(
                InterpolatedWaveform(self.duration, self.amplitude[i]),
                InterpolatedWaveform(self.duration, self.detuning[i]),
                0,
            )
            seq_temp.declare_channel(f"ch{i}", "rydberg_local")
            seq_temp.target(f"q{i}", f"ch{i}")
            seq_temp.add(pulse, f"ch{i}")
        
        seq_temp.draw(mode="input")





class PulseSimulationFixed(Pulse_simulation):
    """
    A fixed-parameter quantum pulse simulation class for solving constrained QUBO problems.
    
    This class experimentally determines fixed values for amplitude and detuning
    to simulate quantum state evolution under predefined pulse settings.
    """
    def __init__(self, Q, step_time=10):
        """
        Initializes the simulation with predefined amplitude and detuning parameters.
        
        Args:
            Q (np.ndarray): The QUBO matrix.
            step_time (int): Time step interval for the simulation.
        """
        duration = 4000  # Fixed total duration of the pulse sequence
        
        # Normalize the QUBO matrix for experimental tuning
        Q_cal = Q / off_diagonal_median(Q) * 20
        Q_diag = np.median(np.diag(Q_cal))
        
        # Define detuning values based on the QUBO matrix diagonal terms
        if 0.5 < np.min(abs(np.diag(Q_cal))):
            detuning = [[Q_diag / 4, -Q_cal[i][i] / 2 + 3] for i in range(len(Q_cal))]
        else:
            detuning = [[Q_diag / 4, -Q_cal[i][i] / 2] for i in range(len(Q_cal))]
        
        # Set a fixed amplitude sequence
        amplitude = [[0, 8, 0] for _ in range(len(Q_cal))]
        
        self.amplitude = amplitude
        self.detuning = detuning
        self.duration = duration
        self.step_time = step_time
        
        # Generate interpolation points
        self.points = np.linspace(0, 1, int(duration / step_time))
        self.points = (self.points[:-1] + self.points[1:]) / 2
        
        # Convert QUBO to Ising model
        Q_copy = copy.deepcopy(Q_cal)
        np.fill_diagonal(Q_copy, 0)
        self.Q_ising = zero_lower_triangle(qubo_to_ising(Q_copy / 2))
        
        # Generate Hamiltonians
        self.generate_hamiltonians()

def create_square_register(N):
    """
    Function to randomly generate a register for Pulser simulation drawing.
    
    Args:
        N (int): Number of qubits.
    
    Returns:
        Register: A register created from generated coordinates.
    """
    # Calculate the number of rows based on the square root of N
    rows = int(np.floor(np.sqrt(N)))  
    
    # Calculate the number of columns to fit all qubits
    cols = int(np.ceil(N / rows))     
    
    coordinates = []
    for i in range(rows):
        for j in range(cols):
            # Ensure the total number of coordinates does not exceed N
            if len(coordinates) < N:
                # Assign coordinates with a spacing of 5 units
                coordinates.append((i * 5, j * 5))  
    
    # Create and return a register using the generated coordinates
    return Register.from_coordinates(coordinates, prefix="q")

def Q_to_ham(Q):
    """
    Converts a given Q matrix into Hamiltonian coefficients and operators.
    
    Args:
        Q (np.ndarray): The matrix representing interactions.
    
    Returns:
        tuple:
            - coeffs_ZZ: list of coefficients for ZZ interactions.
            - ops_ZZ: list of PauliZ ⊗ PauliZ operators.
            - coeffs_Z: list of coefficients for Z interactions.
            - ops_Z: list of PauliZ operators.
    """
    coeffs_ZZ = []
    ops_ZZ = []
    coeffs_Z = []
    ops_Z = []
    for i in range(len(Q)):
        for j in range(len(Q)):
            if i != j and Q[i][j] != 0:
                coeffs_ZZ.append(Q[i][j])
                ops_ZZ.append(qml.PauliZ(i) @ qml.PauliZ(j))
            if i == j:
                coeffs_Z.append(Q[i][i])
                ops_Z.append(qml.PauliZ(i))
    return coeffs_ZZ, ops_ZZ, coeffs_Z, ops_Z