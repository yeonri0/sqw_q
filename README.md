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

1. **モデルの定義**  
   1D spin-1/2 Heisenberg model のハミルトニアンを定義する。

2. **基底状態の計算と量子回路で準備**  
   DMRG / MPS を用いてreference基底状態 $|\psi_0\rangle$ を求める。

   この基底状態を、量子回路ansatzで実現したい。まず初期状態を、隣接サイトごとのシングレット状態の積として準備する。その後、ハミルトニアンに基づくゲート
   $$U_{\mathrm{pair}}(\theta)=e^{-i\theta H_{\mathrm{pair}}},    H_{pair} = J/4 (XX+YY+ZZ)$$

   を複数層作用させ、HVA回路の状態 $|\psi_{\mathrm{HVA}}(\boldsymbol{\theta})\rangle$ を得る。DMRGで求めた基底状態 $|\psi_0\rangle$ との忠実度$F(\boldsymbol{\theta})=\left|\langle\psi_0|\psi_{\mathrm{HVA}}(\boldsymbol{\theta})\rangle\right|^2$
   が最大となるようにパラメータ $\boldsymbol{\theta}$ を最適化し、基底状態を近似的に再現する。
   $|\psi_{\mathrm{HVA}}(\boldsymbol{\theta}^*)\rangle \approx |\psi_0\rangle$

3. **局所摂動と時間発展**

   中央サイト $c$ に摂動(回転ゲート $R^z(\pi/2)$ )を加え、摂動後の初期状態

   $$|\phi(0)\rangle=R^Z_c(\pi/2)|\psi_{\mathrm{HVA}}(\boldsymbol{\theta}^*)\rangle$$

   を準備する。2次の Trotter 分解を用いて、時間発展を量子回路として構成する。
   $$|\phi(t)\rangle=(e^{-iH\Delta t})^n|\phi(0)\rangle$$
   ここで、実機での回路深さを削減するため、Approximate Quantum Compilation（AQC）によって、初期の深い回路を短いパラメータ付き回路で近似・圧縮する。AQCでは、元の回路が生成する目標状態 $|\psi_{\mathrm{target}}\rangle$ と圧縮後の状態 $|\psi_{\mathrm{AQC}}(\boldsymbol{\theta})\rangle$ の忠実度が最大となるようにパラメータを最適化する。 

4. **測定**  
   各時刻・各サイトで $\sigma^z$ を測定し、繰り返し測定した結果から期待値を推定する。

5. **RetardedGreen 関数の計算**  
   測定結果から Retarded Green function $G^R(r,t)$ を求める。

6. **動的構造因子の計算**  
   空間・時間方向に Fourier transform を行い、$S(q,ω)$ を求める。

### 主な手法・ツール

- **基底状態：** DMRG / MPS
- **時間発展：** 2次の Trotter 分解
- **量子回路の近似・圧縮：** Approximate Quantum Compiling（AQC）
- **量子回路の構築・実行：** Qiskit / IBM Quantum
- **測定結果の解析：** 期待値推定、Fourier transform


古典計算機： 基底状態の変分パラメータの最適化、AQC による回路圧縮、結果のフーリエ変換

量子コンピュータ： 圧縮した量子回路を実行し、各サイトの $\langle Z_i\rangle$ を測定

この手法は、古典計算によるテンソルネットワーク法と量子コンピュータを組み合わせたハイブリッド計算である。