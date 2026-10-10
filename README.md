# Dynamical Structure Factor $S(q,\omega)$

This repository contains Python(Qiskit) codes to calculate the dynamical structure factor $S(q,\omega)$ of the 1D spin-1/2 Heisenberg model using quantum computer.

参考:
https://quantum.cloud.ibm.com/docs/en/tutorials/simulate-neutron-scattering?utm_source=chatgpt.com

量子コンピュータを用いて、スピン鎖の動的構造因子 $S(q,\omega)$ を計算する。

古典計算(MPS/DMRG)を用いて基底状態を準備し、量子回路で時間発展を行い、測定結果から相関関数を求めて Fourier transform する。

## Model: 1D Spin-1/2 Heisenberg Model

The 1D spin-1/2 Heisenberg Hamiltonian is following:

$$
H=J\sum_{i=1}^{L-1}
\mathbf{S}_i\cdot\mathbf{S}_{i+1}
$$

where $J>0$ is the antiferromagnetic exchange coupling. \
Open boundary conditions (OBC) are used.

### 計算の流れ

1. **基底状態の計算**  
   古典計算で DMRG / MPS を用いて基底状態 `|ψ₀⟩` を求める。

2. **量子回路の構築**  
   基底状態の準備と時間発展を量子回路で表現する。Approximate Quantum Compiling（AQC）を用いて回路を近似・圧縮する。

3. **局所摂動と時間発展**  
   特定のサイトに局所摂動を加え、2次の Trotter 分解によって時間発展演算子を量子ゲート列に近似する。

4. **測定**  
   各時刻・各サイトで `σᶻ` を測定し、繰り返し測定した結果から期待値を推定する。

5. **RetardedGreen 関数の計算**  
   測定結果から Retarded Green function `Gᴿ(r,t)` を求める。

6. **動的構造因子の計算**  
   空間・時間方向に Fourier transform を行い、`S(q,ω)` を求める。

### 主な手法・ツール

- **基底状態：** DMRG / MPS
- **量子回路の近似・圧縮：** Approximate Quantum Compiling（AQC）
- **時間発展：** 2次の Trotter 分解
- **量子回路の構築・実行：** Qiskit / IBM Quantum
- **測定結果の解析：** 期待値推定、Fourier transform

この手法は、古典計算によるテンソルネットワーク法と量子コンピュータを組み合わせたハイブリッド計算である。