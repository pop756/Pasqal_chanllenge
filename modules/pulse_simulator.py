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
    def __init__(self, Q,amplitude, detuning, duration,step_time = 50):
        self.amplitude = amplitude
        self.detuning = detuning
        self.x_amp = [np.linspace(0,1,len(amplitude[i]))  for i in range(len(amplitude))]
        self.x_detune = [np.linspace(0,1,len(detuning[i]))  for i in range(len(detuning))]
        self.duration = duration
        self.step_time = step_time
        points = np.linspace(0, 1, int(duration/step_time))
        points = (points[0:-1]+points[1:])/2

        self.points = points

        Q_copy = copy.deepcopy(Q)
        np.fill_diagonal(Q_copy,0)
        Q_ising = zero_lower_triangle(qubo_to_ising(Q_copy/2))
        self.Q_ising = Q_ising
        self.generate_hamiltonians()
    def generate_hamiltonians(self):
      """
      Hamiltonian 리스트를 생성하는 함수.

      Returns:
          list[qml.Hamiltonian]: pulse + cite Hamiltonian 리스트.
      """
      coeffs_ZZ, ops_ZZ,coeffs_Z,ops_Z = Q_to_ham(self.Q_ising)
      amp,detune = self.interpolate_1d()
      hamiltonian_list = []
      for time in range(len(amp[0])):
        coeffs =  list(coeffs_ZZ)
        ops = list(ops_ZZ)
        for q_index in range(len(amp)):
            # 랜덤 계수 및 Pauli 연산자 선택
            coeffs.append(amp[q_index][time]/2)
            ops.append(qml.PauliX(q_index))
            coeffs.append(detune[q_index][time]/2+coeffs_Z[q_index])
            ops.append(qml.PauliZ(q_index))
        H = qml.Hamiltonian(coeffs, ops)
        hamiltonian_list.append(H)
      self.ham = hamiltonian_list

      

    def simulate_time_evolution(self):
      """
      주어진 Hamiltonian 리스트를 step_time 간격으로 순차적으로 적용하여 최종 상태 계산.

      Args:
          hamiltonian_list (list[qml.Hamiltonian]): 시간에 따른 Hamiltonian 리스트.
          step_time (float): 각 Hamiltonian을 적용할 시간 (ns 단위).

      Returns:
          list[float]: 각 큐비트의 최종 <Z> 기대값.
      """
      # 전체 큐비트 개수 찾기
      qubits = set()
      for H in self.ham:
          for op in H.ops:
              qubits.update(op.wires)


      # Correct instantiation of SparseHamiltonian with the 'wires' argument

      num_qubits = len(qubits)  # 총 큐비트 개수

      dev = qml.device("default.qubit", wires=num_qubits)


      def QAA_circuit():
          # 2. Hamiltonian 리스트를 순차적으로 적용하여 시간 발전 수행
          for H in self.ham:
              qml.ApproxTimeEvolution(H, self.step_time/1000, 1)  # 1st-order Trotter step  
      return QAA_circuit()

    def interpolate_1d(self):
      """
      1차원 보간(interpolation) 함수.

      Args:
          x_points (array-like): 원래 데이터의 x 좌표들.
          y_points (array-like): 원래 데이터의 y 값들.
          x_new (array-like): 보간할 새로운 x 좌표들.
          method (str): 보간 방법 선택. ("linear", "polynomial", "spline")

      Returns:
          np.ndarray: 보간된 y 값들.
      """
      amp_list = []
      detune_list = []
      for i in range(len(self.amplitude)):
        amp = interp.PchipInterpolator(self.x_amp[i], self.amplitude[i])
        detune = interp.PchipInterpolator(self.x_detune[i], self.detuning[i])
        amp_list.append(amp(self.points))
        detune_list.append(detune(self.points))
      return amp_list,detune_list

    def draw(self):
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




class Pulse_simulation_fixed(Pulse_simulation):
    def __init__(self, Q,step_time = 10):
        duration = 4000
        


        Q_cal = Q/off_diagonal_median(Q)*32
        Q_diag = np.median(np.diag(Q_cal))

        if 0.5 < np.min(abs(np.diag(Q_cal))):
            detuning = [[Q_diag/4 ,-Q_cal[i][i]/2+3] for i in range(len(Q_cal))]
        else:
            detuning = [[Q_diag/4 ,-Q_cal[i][i]/2] for i in range(len(Q_cal))]
        amplitude = [[0,8,0] for i in range(len(Q_cal))]
        self.amplitude = amplitude
        self.detuning = detuning
        self.x_amp = [np.linspace(0,1,len(amplitude[i]))  for i in range(len(amplitude))]
        self.x_detune = [np.linspace(0,1,len(detuning[i]))  for i in range(len(detuning))]
        self.duration = duration
        self.step_time = step_time
        points = np.linspace(0, 1, int(duration/step_time))
        points = (points[0:-1]+points[1:])/2

        self.points = points
        Q_copy = copy.deepcopy(Q_cal)
        np.fill_diagonal(Q_copy,0)
        Q_ising = zero_lower_triangle(qubo_to_ising(Q_copy/2))
        self.Q_ising = Q_ising
        self.generate_hamiltonians()


def create_square_register(N):
    """
    Function to randomly generate a register for Pulser simulation drawing.

    Parameters:
    N : int
        Number of qubits.

    Returns:
    Register
        A register created from generated coordinates.
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

    Parameters:
    Q : numpy.ndarray
        The matrix representing interactions.

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