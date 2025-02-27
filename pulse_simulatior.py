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


def create_square_register(N, spacing=5.0):
    """
    주어진 N개의 원자를 사각형 형태로 배치하여 Pulser Register 생성

    Args:
        N (int): 원자의 개수
        spacing (float): 원자 간격 (기본값: 5μm)

    Returns:
        Register: Pulser Register 객체
    """
    # 1. 가장 가까운 정사각형 행렬 크기 찾기
    rows = int(np.floor(np.sqrt(N)))  # ⌊√N⌋
    cols = int(np.ceil(N / rows))     # ⌈N / rows⌉

    # 2. 원자 좌표 리스트 생성
    coordinates = []
    for i in range(rows):
        for j in range(cols):
            if len(coordinates) < N:
                coordinates.append((i * spacing, j * spacing))  # (x, y) 좌표 생성

    # 3. Pulser Register 생성
    return Register.from_coordinates(coordinates, prefix="q")

def qubo_to_ising(Q):
    """
    0/1 basis의 QUBO matrix Q (대칭 np.array)를 받아,
    -1/1 basis (Ising)로 변환하여 linear term h와 quadratic interaction matrix J를 반환합니다.
    
    x_i = (1 + z_i) / 2 변환을 이용.
    
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
            h[i] -= Q[i, j] / 8.0
            h[j] -= Q[i, j] /8
            J[i, j] = Q[i, j] / 8.0

    Q_res = J
    np.fill_diagonal(Q_res, h)
    return Q_res

def Q_to_ham(Q):
    coeffs = []
    ops = []
    for i in range(len(Q)):
      for j in range(len(Q)):
        if i!=j:
          coeffs.append(Q[i][j])
          ops.append(qml.PauliZ(i)@qml.PauliZ(j))
        else:
          coeffs.append(Q[i][i])
          ops.append(qml.PauliZ(i))
    return coeffs, ops

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
        Q_ising = qubo_to_ising(Q_copy)
        self.Q_ising = Q_ising
        self.generate_random_hamiltonians()
    def generate_random_hamiltonians(self):
      """
      Hamiltonian 리스트를 생성하는 함수.

      Returns:
          list[qml.Hamiltonian]: pulse + cite Hamiltonian 리스트.
      """
      Q_coeffs,Q_opts = Q_to_ham(self.Q_ising)
      amp,detune = self.interpolate_1d()
      hamiltonian_list = []
      for index in range(len(amp[0])):
        coeffs =  copy.deepcopy(Q_coeffs)
        ops = copy.deepcopy(Q_opts)
        for q_index in range(len(amp)):
            # 랜덤 계수 및 Pauli 연산자 선택
            coeffs.append(amp[q_index][index]/2)
            ops.append(qml.PauliX(q_index))
            coeffs.append(detune[q_index][index]/2)
            ops.append(qml.PauliZ(q_index))
        H = qml.Hamiltonian(coeffs, ops)
        H = H.simplify()
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

      @qml.qnode(dev)
      def circuit():
          # 1. 초기 상태 |000...0> 설정
          qml.BasisState(np.zeros(num_qubits, dtype=int), wires=range(num_qubits))

          # 2. Hamiltonian 리스트를 순차적으로 적용하여 시간 발전 수행
          for H in self.ham:
              qml.ApproxTimeEvolution(H, self.step_time/1000, 1)  # 1st-order Trotter step  
              #qml.adjoint(qml.TrotterProduct(H, self.step_time/1000, order=1, n=1))

          # 3. 최종 상태에서 각 큐비트의 <Z> 값 반환
          return qml.probs(wires=range(num_qubits))

      return circuit()

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
        amp = interp.interp1d(self.x_amp[i], self.amplitude[i], kind="linear", fill_value="extrapolate")
        detune = interp.interp1d(self.x_detune[i], self.detuning[i], kind="linear", fill_value="extrapolate")
        amp_list.append(amp(self.points))
        detune_list.append(detune(self.points))
      return amp_list,detune_list

    def draw(self):
      reg = create_square_register(len(self.amplitude))
      seq_temp = Sequence(reg, DigitalAnalogDevice)


      for i in range(len(self.amplitude)):
          pulse = Pulse(
              InterpolatedWaveform(self.duration, self.amplitude[i],interpolator='interp1d'),
              InterpolatedWaveform(self.duration, self.detuning[i],interpolator='interp1d'),
              0,
          )
          seq_temp.declare_channel(f"ch{i}", "rydberg_local")
          seq_temp.target(f"q{i}", f"ch{i}")
          seq_temp.add(pulse, f"ch{i}")

      seq_temp.draw(mode="input")