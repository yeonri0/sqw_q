import timeit
import warnings
from collections.abc import Iterator, Sequence
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import quimb.tensor as qtn
import scipy.optimize
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction, ParameterVector, Qubit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_addon_aqc_tensor import generate_ansatz_from_circuit
from qiskit_addon_aqc_tensor.objective import MaximizeStateFidelity
from qiskit_addon_aqc_tensor.simulation import tensornetwork_from_circuit
from qiskit_addon_aqc_tensor.simulation.quimb import QuimbSimulator
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_quimb import quimb_circuit
from scipy.sparse import SparseEfficiencyWarning

# scipy.sparse.linalg.expm (reached via PauliEvolutionGate.to_matrix) prefers
# CSC and warns when handed the CSR matrix from SparsePauliOp.to_matrix; the
# result is unaffected, so silence the cosmetic warning.　計算結果に影響しない警告を表示しない
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

# XXZ Hamiltonian MPO ただし昇降演算子で記述
def xxz_hamiltonian_mpo(
    n_qubits: int, interaction: float = 1.0, anisotropy: float = 1.0
) -> qtn.MatrixProductOperator:
    """1D XXZ Hamiltonian as a quimb MPO.

    Builds the Hamiltonian using ``qtn.SpinHam1D``.

    Args:
        n_qubits: Number of sites.
        interaction: Overall interaction strength (J in the paper).
        anisotropy: XY/Z anisotropy (ε in the paper). ``ε = 1`` is the
            isotropic Heisenberg point; ``ε = 0`` is the Ising limit.

    Returns:
        The Hamiltonian as a matrix product operator.
    """
    builder = qtn.SpinHam1D(S=1 / 2)
    builder += interaction * anisotropy * 0.5, "+", "-"
    builder += interaction * anisotropy * 0.5, "-", "+"
    builder += interaction, "Z", "Z"
    return builder.build_mpo(L=n_qubits)

# DMRGで求めた基底状態を近似するためのパラメータ付き回路(HVA の量子回路)を構築する。
def build_ground_state_ansatz(n_qubits: int, n_layers: int) -> QuantumCircuit:
    """Hamiltonian variational ansatz (HVA) circuit for ground-state preparation.

    Starts from a product of singlet pairs and applies alternating
    odd/even layers of parameterized XXZ pair-evolution gates.

    The returned circuit is parameterized: it carries a ``ParameterVector``
    named ``"theta"`` of length ``2 * n_layers`` whose values must be
    assigned (e.g. via ``circuit.assign_parameters``) before simulation.
    Element ``2 * r`` is the odd-layer angle and element ``2 * r + 1`` is the
    even-layer angle of layer ``r``.

    Args:
        n_qubits: Number of qubits (must be even).
        n_layers: Number of HVA layers.

    Returns:
        The parameterized HVA preparation circuit.
    """
    theta = ParameterVector("theta", 2 * n_layers)
    circuit = QuantumCircuit(n_qubits)
    # Initial singlet product state
    for i in range(n_qubits // 2):
        circuit.x(2 * i)
        circuit.x(2 * i + 1)
        circuit.h(2 * i + 1)
        circuit.cx(2 * i + 1, 2 * i)
    # Variational layers: each pair gate is exp(-i theta (XX+YY+ZZ)/2)
    pair_ham = SparsePauliOp(
        ["XX", "YY", "ZZ"], coeffs=[0.5, 0.5, 0.5]
    )  # H_pair (HVA form)
    for r in range(n_layers):
        for i in range(1, (n_qubits + 1) // 2):  # odd layer
            circuit.append(
                PauliEvolutionGate(pair_ham, time=theta[2 * r]),
                [2 * i - 1, 2 * i],
            )
        for i in range(n_qubits // 2):  # even layer
            circuit.append(
                PauliEvolutionGate(pair_ham, time=theta[2 * r + 1]),
                [2 * i, 2 * i + 1],
            )
    return circuit

# 回路パラメータを最適化し、与えられた target_mps (DMRG の基底状態)に HVA の状態を近づける。
def optimize_ground_state_ansatz(
    ansatz: QuantumCircuit,
    x0: NDArray[np.floating],
    target_mps: qtn.MatrixProductState,
    *,
    max_bond: int | None = None,
    cutoff: float = 1e-10,
    method: str = "COBYQA",
    options: dict | None = None,
) -> scipy.optimize.OptimizeResult:
    """Optimize HVA parameters by maximizing fidelity with a target MPS.

    The HVA circuit is simulated as a matrix product state with the given
    bond-dimension truncation, and the parameters are optimized to maximize
    the state fidelity ``|<psi_HVA | target_mps>|**2`` with the DMRG ground
    state. Both states are normalized, so the minimized objective is the
    infidelity ``1 - |<psi_HVA | target_mps>|**2``.

    Args:
        ansatz: Parameterized HVA circuit from ``build_ground_state_ansatz``.
            The length of ``x0`` must equal ``ansatz.num_parameters``.
        x0: Initial parameters.
        target_mps: Target MPS (DMRG ground state) to maximize fidelity with.
        max_bond: Maximum MPS bond dimension during gate application.
        cutoff: Singular-value cutoff during gate application.
        method: ``scipy.optimize.minimize`` method.
        options: Options dict forwarded to ``scipy.optimize.minimize``.

    Returns:
        The Scipy OptimizeResult.
    """
    # パラメータを最適化するための目的関数
    def infidelity(params: NDArray[np.floating]) -> float:
        circuit = ansatz.assign_parameters(params)
        circuit_mps = quimb_circuit(
            circuit.decompose(["PauliEvolution"]),
            quimb_circuit_class=qtn.CircuitMPS,
            max_bond=max_bond,
            cutoff=cutoff,
        )
        return 1 - abs(circuit_mps.psi.H @ target_mps) ** 2
    # 目的関数が最小になるようにパラメータを最適化する
    return scipy.optimize.minimize(
        infidelity, np.asarray(x0), method=method, options=options
    )

# 遅延Green 関数から動的構造因子を計算	Fourier 変換などの後処理
def get_dsf(
    n_qubits: int,
    rgf_mat: NDArray[np.floating],
    time_step: float,
    n_steps: int,
    n_points_momentum: int,
    n_points_frequency: int,
) -> NDArray[np.floating]:
    """Compute the dynamical structure factor from the retarded Green's function.

    Uses the center-site approximation and a discrete Fourier transform.
    The result is symmetrized about the momentum axis and clipped to
    non-negative values, ready for plotting.

    Args:
        n_qubits: Number of qubits (sites).
        rgf_mat: RGF matrix of shape ``(n_steps, n_qubits)``.
        time_step: Trotter time-step size.
        n_steps: Number of time steps.
        n_points_momentum: Number of momentum points.
        n_points_frequency: Number of frequency points.

    Returns:
        DSF array of shape ``(n_points_frequency, n_points_momentum)``,
        symmetrized about the momentum axis and clipped to non-negative values.
    """
    max_frequency = np.pi / time_step
    momentum_range = np.linspace(0, 2 * np.pi, n_points_momentum)
    frequency_range = np.linspace(0, max_frequency, n_points_frequency)
    result = np.zeros((frequency_range.shape[0], momentum_range.shape[0]))
    center = n_qubits // 2 - 1
    for iw, w in enumerate(frequency_range):
        exponent = np.exp(1j * w * time_step * np.arange(1, n_steps + 1))
        # S = sigma/2, so two spin operators contribute a factor of (1/2)(1/2) = 1/4.
        rgf_omega = (
            np.dot(rgf_mat.T, exponent) * time_step / 4
        )  # S(omega): time Fourier slice of the Green's function
        for iq, q in enumerate(momentum_range):
            momentum_phases = np.exp(
                -1j * q * np.arange(-center, center + 2, 1)
            )
            result[iw, iq] = np.imag(np.dot(rgf_omega, momentum_phases))
    result = -(result + result[:, ::-1]) / 2
    result = np.clip(result, a_min=0, a_max=None)
    return result

# グラフデザイン
plt.rcParams.update({
    "axes.labelsize": 16,    # x軸・y軸のラベル
    "xtick.labelsize": 14,   # x軸の目盛り
    "ytick.labelsize": 14,   # y軸の目盛り
    "axes.titlesize": 14,    # グラフタイトル
    "legend.fontsize": 12,   # 凡例
    "figure.titlesize": 14,  # Figure全体のタイトル
})

# RGFのヒートマップを描画
def plot_rgf(
    n_qubits: int,
    rgf_mat: NDArray[np.floating],
    time_step: float,
    n_steps: int,
    title: str | None = None,
) -> None:
    """Heat-map of the retarded Green's function in real space and time.

    Args:
        n_qubits: Number of qubits (sites).
        rgf_mat: RGF matrix of shape ``(n_steps, n_qubits)``.
        time_step: Trotter time-step size.
        n_steps: Number of time steps.
        title: Optional plot title.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    qubit_axis = np.arange(n_qubits)
    t_axis = np.arange(1, n_steps + 1) * time_step
    x, y = np.meshgrid(qubit_axis, t_axis)
    c = ax.pcolormesh(x,y,np.real(rgf_mat),cmap="RdBu",vmax=0.5,vmin=-0.5,shading="auto")
    cbar = fig.colorbar(c, ax=ax)
    cbar.set_label(r"Re $G^R(j, j_c, t)$", fontsize=16)
    cbar.ax.tick_params(labelsize=12)
    ax.set_xlabel("Qubit")
    ax.xaxis.set_major_locator(
        plt.matplotlib.ticker.MaxNLocator(integer=True)
    )
    ax.set_ylabel(r"$Jt$")
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig("rgf_lightcone_10qubits.png", dpi=300)
    plt.show()


# S(q,w)のヒートマップを描画
def plot_dsf(
    dsf: NDArray[np.floating],
    time_step: float,
    n_points_momentum: int,
    n_points_frequency: int,
    title: str | None = None,
) -> None:
    """Heat-map of the dynamical structure factor.

    Args:
        dsf: DSF array of shape ``(n_points_frequency, n_points_momentum)``.
        time_step: Trotter time-step size.
        n_points_momentum: Number of momentum points.
        n_points_frequency: Number of frequency points.
        title: Optional plot title.
    """
    max_frequency = np.pi / time_step
    momentum_range = np.linspace(0, 2 * np.pi, n_points_momentum)
    frequency_range = np.linspace(0, max_frequency, n_points_frequency)
    x, y = np.meshgrid(momentum_range, frequency_range)
    fig, ax = plt.subplots(figsize=(8, 5))
    q_exact = np.linspace(0, 2 * np.pi, 500)
    J = 1.0  # Example value for J
    omega_exact1 = np.pi * J * np.abs(np.sin(q_exact / 2))
    omega_exact2 = np.pi * J / 2 * np.abs(np.sin(q_exact))
    plt.plot(q_exact,omega_exact1, "b-", linewidth=1.8, label=r"$\omega_{\mathrm{U}}(q)=\pi J|sin\frac{q}{2}|$")
    plt.plot(q_exact,omega_exact2, "y-", linewidth=1.8, label=r"$\omega_{\mathrm{L}}(q)=\frac{\pi J}{2}|\sin q|$")
    plt.legend(loc="upper right")
    c = ax.pcolormesh(x, y, dsf / np.max(dsf), cmap="turbo", shading="auto")
    cbar = plt.colorbar(c, label=r"$S^{zz}(q,\omega)$ normalized")
    cbar.set_label(r"$S^{zz}(q,\omega)$ normalized", fontsize=16)
    c.set_clim(0, 1)
    cbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.ax.tick_params(labelsize=14)


    ax.set_ylim(0, 3.5)
    ax.set_xlim(0, 2 * np.pi)
    ax.set_xlabel(r"$q$")
    ax.set_ylabel(r"$\omega / J$")
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig("sqw_q_10qubits.png", dpi=300)
    plt.show()


# Trotter evolution of the XXZ Hamiltonian
def trotter_evolution(
    qubits: Sequence[Qubit],
    interaction: float,
    anisotropy: float,
    time_step: float,
    n_steps: int,
) -> Iterator[CircuitInstruction]:
    """Second-order Trotter steps of the XXZ pair Hamiltonian.

    While the paper used a hand-optimized circuit for the Trotter steps, we use
    PauliEvolutionGate here for simplicity and generality. The final two-qubit gate
    count and gate depth are equivalent when transpiled with ``optimization_level=3``.

    Args:
        qubits: Qubits to act on (length ``n_qubits``).
        interaction: Overall interaction strength (J in the paper).
        anisotropy: XY/Z anisotropy (ε in the paper). ``ε = 1`` is the
            isotropic Heisenberg point; ``ε = 0`` is the Ising limit.
        time_step: Per-step Trotter time.
        n_steps: Number of Trotter steps.

    Yields:
        ``CircuitInstruction``s implementing the Trotter steps.
    """
    if n_steps == 0:
        return
    n_qubits = len(qubits)
    pair_ham = SparsePauliOp(
        ["XX", "YY", "ZZ"],
        coeffs=[
            0.25 * interaction * anisotropy,
            0.25 * interaction * anisotropy,
            0.25 * interaction,
        ],
    )
    half_evo = PauliEvolutionGate(pair_ham, time=time_step / 2)
    full_evo = PauliEvolutionGate(pair_ham, time=time_step)
    for i in range(n_qubits // 2):  # half even layer
        yield CircuitInstruction(half_evo, (qubits[2 * i], qubits[2 * i + 1]))
    for i in range(n_qubits // 2 - 1):  # full odd layer
        yield CircuitInstruction(
            full_evo, (qubits[2 * i + 1], qubits[2 * i + 2])
        )
    for _ in range(n_steps - 1):  # interior steps
        for i in range(n_qubits // 2):
            yield CircuitInstruction(
                full_evo, (qubits[2 * i], qubits[2 * i + 1])
            )
        for i in range(n_qubits // 2 - 1):
            yield CircuitInstruction(
                full_evo, (qubits[2 * i + 1], qubits[2 * i + 2])
            )
    for i in range(n_qubits // 2):  # half even layer
        yield CircuitInstruction(half_evo, (qubits[2 * i], qubits[2 * i + 1]))


# 2量子ビットゲートの深さを求め、回路の実行コストを評価
def uniform_2q_depth(circuit: QuantumCircuit) -> int:
    """Two-qubit gate depth in a standardized basis."""
    pass_manager = generate_preset_pass_manager(
        optimization_level=0, basis_gates=["cz", "id", "rz", "sx", "x"]
    )
    return pass_manager.run(circuit).depth(
        lambda inst: inst.operation.num_qubits == 2
    )

# ── Retrieve the existing QPU result ──
service = QiskitRuntimeService(name="ibm-kobe-probe")
job = service.job("db6uem84qg6s73c5dqp0")
result = job.result()

# ── Parameters ──────────────────────────────────────────────────────────────
n_qubits = 10
interaction = 1.0  # J
anisotropy = 1.0  # ε (isotropic point)
time_step = 0.6
n_steps = 10  # 10 → 20
aqc_n_steps_1 = 3  # 3 → 6
aqc_n_steps_2 = 2  # 2 → 4
aqc_n_steps_total = aqc_n_steps_1 + aqc_n_steps_2
gs_n_layers = 3  # 3 → 5
mps_max_bond = 32  # 32 -> 128
mps_cutoff = 1e-8
center = n_qubits // 2 - 1


# ── Step 4: Post-process ──
rgf_mat = np.stack([pub_result.data.evs for pub_result in result])

n_points_momentum = 100
n_points_frequency = 100

spectrum = get_dsf(
    n_qubits,
    rgf_mat,
    time_step,
    n_steps,
    n_points_momentum,
    n_points_frequency,
)

plot_rgf(
    n_qubits,
    rgf_mat,
    time_step,
    n_steps,
    title=f"Retarded Green's function — {n_qubits} qubits (QPU)",
)


plot_dsf(
    spectrum,
    time_step,
    n_points_momentum,
    n_points_frequency,
    title=rf"KCuF$_3$ DSF — {n_qubits} qubits (QPU)"
)