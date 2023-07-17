### Q1: 量子化学模拟 (VQE)

1. 最优化分子结构求解
2. 激发态能求解的浅层线路设计


### theory

⚪ Notes of [vqe_for_quantum_chemistry](https://www.mindspore.cn/mindquantum/docs/zh-CN/r0.8/vqe_for_quantum_chemistry.html)

```
量子化学的核心问题：求解薛定谔方程 (Schrödinger Equation)
引入玻恩-奥本海默近似 (Born-Oppenheimer approximation, BO approximation) 将 含时薛定谔方程 近似为 不含时/定态薛定谔方程
  <H> = 电子动能 <K_e> + 电子-电子势能 <V_ee> + 电子-核势能 <V_Ne>
其中 <H> 为哈密顿量 H 的期望

三种波函数方法用于求解定态薛定谔方程：
  - HF: Hartree-Fock
  - CI: Configuration Interaction
    - CISD: Configuration Interaction with Singles and Doubles, 只考虑单激发和双激发
    - FCI: Full Configuration Interaction, 将基态 HF 波函数一直到 N 激发波函数全部考虑在内, FCI 波函数是定态薛定谔方程在给定基函数下的精确解
  - CC: Coupled-Cluster theory
    - CCSD: 改进的 CISD
    - UCC: 幺正的 CC 以用于量子计算

方法分析：
  - 运行时间: T_HF < T_CCSD << T_FCI
  - 运行结果能量值: E_HF > E_CCSD > E_FCI
  - 随着体系大小（电子数、原子数）的增加，求解 FCI 波函数和基态能量的时间消耗大约以 2^N 增长，即使是较小的分子如乙烯分子等，进行 FCI 计算也并不容易。

[量子变分求解器]
变分原理
  E0 <= <ψ|H|ψ> / <ψ|ψ>
其中 H 为系统的哈密顿量, |ψ> 表示试探波函数, E0 为真实基态能量

即有，任意试探波函数得到的基态能量总是大于等于真实的基态能量
故，变分原理为求解分子基态薛定谔方程提供了一种方法：使用一个参数化的函数 f(θ) 作为精确基态波函数 |ψ> 的近似，通过优化参数 θ 来逼近精确的基态能量 E0

[VQE算法流程]
- 制备 HF 初态 |00..11..>; |0> 表示电子不占据轨道, |1> 表示占据
- 定义 UCCSD 等波函数拟设，转化为参数化量子线路 vqc(θ)
- 初始化所有的变分参数 θ
- 重复直到满足收敛条件
  - 在量子计算机上多次测量得到目标分子哈密顿量在当前 vqc(θ) 下的能量 E(θ) 及其梯度 {∂E/∂θ} 
  - 在经典计算机上使用优化算法 (SDG、BFGS等) 更新变分参数 θ
```

⚪ Notes of thesis `A variational eigenvalue solver on a quantum processor`

```
[Quantum Expectation Estimation (QEE)]
This algorithm computes the expectation value of <H> a given Hamiltionian H and an input state |psi>
  <H> = <psi|H|psi>
it works like a parallel version of QPE that reduces required coherent evolution time to O(1)

Any Hamiltionian H (a matrix) can be decomposed using Pauli operators:
  H = Σ[iα] h_α^i * σ_α^i + Σ[ijαβ] h_αβ^ij * σ_α^i * σ_β^j + ...
where 
  - h are term coeffs
  - the Roman letter i/j/.. denotes the i-th subspace on which the operator acts
  - the Greek letter α/β/.. is chosen from [x, y, z] so that σ_α is a Pauli operator
Then the expectation of H denoted as <H>, following the linearity of quantum observables, is:
  <H> = Σ[iα] h_α^i * <σ_α^i> + Σ[ijαβ] h_αβ^ij * <σ_α^i * σ_β^j> + ...
When terms of this H is polynomial in size of the system, it can be well arpproximated by several matrix production
That's to say, evaluation of any given <H> reduces to polynomial number of expections of simple Pauli operators for a state |psi>
  (*) 倒着说就是仅靠 O(N^p) 个简单的 Pauli-XYZ 算子的组合嵌套，可以逼近任意哈密顿量 H
Quantum device can efficiently evaluate a 2^n-scale Pauli operator composed H using only n-qubits
The time complexity is: 
  - we have M decomposed terms to evaluate
  - each term needs O(|h|^2/p^2) repeations to reach precision p
    - FIXME: what is a repeation, how is this done?
    - indicates the n_iters of function optimizing?
  - hence in total: O(|h_max|^2*M/p^2)

[Quantum Variational Eigensolver (QVE)]
This algorithm is a variational method to prepare the eigenstate
We need it beacuse both QPE and QEE require a approximated ground state wavefunction to compute the ground state eigenvalue,
and the known methods to get the ground state all require long coherent evolution:
  - adiabatic evolution
  - quantum metropolis algorithm

Respecting to Rayleigh-Ritz quotient, the eigenvector |psi> corresponding to the lowest eigenvalue is the |psi> that
  min. <psi|H|psi> / <psi|psi>
ℹ ref:
  - https://en.wikipedia.org/wiki/Rayleigh%E2%80%93Ritz_method
  - https://en.wikipedia.org/wiki/Rayleigh_quotient
To prepare this unknown |psi>, we can use a variaional method of parametrize fucntion optimizing:
  i.e. let |psi> = f(|init>, theta), then optimize parameters theta of f
Choose the unitary coupled cluster (UCC) ansatz as the f above:
  |psi> = e^(T - T.daggar) |init>
where 
  - T is the cluster operator, UCC contains polynomial amout of parameters (rather than expotienal)
  - ref state |init> is set to the Hartree-Fock ground state

Finding excited states:
repeating the procedure of Hλ = (H-λ)^2, the **folded spetrum method** allows a variational method to converge to the eigenvec closest to the shift parameter λ
therefore, search in a range of λ to find any he is interested with, but this blind search is slow :(
```

⚪ Notes of thesis `Variational Quantum Computation of Excited States`

```
[Variational Quantum Deflation (VQD)]
ℹ Also known as Orthogonal-Restrained VQE

In original VQE, the parameters λ of ansatz ψ are optimized with respect to the expetation value:
  E(λ) = <ψ(λ)|H|ψ(λ)> = Σj cj <ψ(λ)|P(j)|ψ(λ)>
where H = Σj cj*P(j), decomposed to a set of linear-additive simple Paulis P(j)

Then VQD extends VQE to find the k-th exicted state by optimzing parameters λk for ansatz ψ to prepare state |ψ(λk)> sat.
  mim. F(λk) = <ψ(λk)|H|ψ(λk)> + Σi[0, k-1] βi |<ψ(λk)|ψ(λi)>|^2
             = Ek + reg_term
this can be taken as searching for one more state that is diagonal to all formerly unknown states, yet still minimizes the expectation
note that to ensure |<ψ(λk)|ψ(λi)>|^2 -> 0, the βi must be large enough; 
and the ansatz ψ must be enough expressive to produce adequate different states by varying λ

Note that this algorithm is recursive, in order to get E(λk), you MUST solve all E(λj) where j < k.
```

⚪ Notes of scipy.optimze

```
[optimize.minimize() methods]
⚪ not need derivatives
- Nelder-Mead simplex: Nelder-Mead 单纯形法
  - 把输入向量 x0 视为 N 维度空间里的一个单纯型 (凸包), 反射/收缩这个凸包各顶点来寻找最优解；不保证最优
  - ref: https://blog.csdn.net/Accelerating/article/details/121508984
- Powell: 
  - 双向网格搜索；在输入向量 x0 的每个维度，双向等距打点搜索
  - ref: https://en.wikipedia.org/wiki/Powell%27s_method
- COBYLA: Constrained Optimization BY Linear Approximation algorithm
  - 解线性规划问题去近似解，然后缩减步幅到收敛
  - ref: https://handwiki.org/wiki/COBYLA

⚪ need derivatives (quasi-Newton methods)
- CG / conjugate gradient
  - 用于解线性方程 Ax=b
  - ref: https://en.wikipedia.org/wiki/Conjugate_gradient_method 
- Newton-CG
- TNC: truncated Newton
  - aka. Hessian-free optimization, 假设输入向量 x 各维度是独立无关的
  - ref: https://en.wikipedia.org/wiki/Truncated_Newton_method
- BFGS / Broyden–Fletcher–Goldfarb–Shanno algorithm
  - 类似梯度下降，但越来越逼近 Hessian 矩阵
  - ref: https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm
- L-BFGS-B
  - limited-momery box-bounded BFGS
  - ref: https://en.wikipedia.org/wiki/Limited-memory_BFGS
- SLSQP: Sequential Least Squares Programming
  - 对于无约束情况将退化到朴素 Newton 法
  - ref: https://en.wikipedia.org/wiki/Sequential_quadratic_programming
- dogleg
  - 需要一个 trust region
  - ref: https://en.wikipedia.org/wiki/Powell%27s_dog_leg_method
- trust-constr: trust-region algorithm for constrained optimization
- trust-ncg： trust-region algorithm with Newton conjugate gradient
- trust-exact: trust-region algorithm using a nearly exact trust-region
- trust-krylov: trust-region algorithm with exact region that only requires matrix vector products with the hessian matrix
```

⚪ Notes of mindquantum framework

```
[算符]
FermionOperator: 描述一个费米系统，如分子系统；遵循反对易关系
  a_4.dagger a_3 a_9 a_3.dagger  <=>  FermionOperator("4^ 3 9 3^")
QubitOperator (PauliOperator)：描述一个泡利算符
  0.5 * "X1 X5" + 0.3 * "Z1 Z2"  <=>  QubitOperator("X1 X5", 0.5) + QubitOperator("Z1 Z2", 0.3)
Hamiltonian: QubitOperator 的一个wrapper，不知为何

[如何以物理的思考方式构建一个QNN]
Encoder: 从经典数据映射到量子数据，制备一个初态 |phi>
Ansatz: 进行态演化，制备一个末态 |psi>
Loss: 通常是最小化一个可观测量期望值 <psi|H|psi>
  - Hamiltonian 量就是个矩阵 (密度矩阵?)
  - 也可以认为就是一个观测投影方式 (?)

[哈密顿量到底tm是个啥]
  - https://zhuanlan.zhihu.com/p/150292241

化学分子结构处理之后可以得到一个哈密顿量 H
  - dim(H) = 2^n_qubits ; n_qubits = 2 * n_electrons (?)
  - H is (a real) diag
  - SCF/HF energy == min(diag(H)) = λmin
  - a lot of pair-wise duplicated values exists in diag(H)
```


### references

- thesis
  - [A variational eigenvalue solver on a quantum processor](https://arxiv.org/abs/1304.3061)
  - [Variational Quantum Computation of Excited States](https://arxiv.org/abs/1805.08138)
  - [Subspace-search variational quantum eigensolver for excited states](https://arxiv.org/abs/1810.09434)
- Q framework
  - [pychemiq](https://pychemiq-tutorial.readthedocs.io/en/latest/index.html)
  - [mindquantum - ansatz zoo](https://gitee.com/mindspore/mindquantum/tree/research/ansatz_zoo)
- VQE tutorial
  - [pychemiq - vqe introduction](https://pychemiq-tutorial.readthedocs.io/en/latest/05theory/vqeintroduction.html)
  - [mindquantum - vqe for quantum chemistry](https://www.mindspore.cn/mindquantum/docs/zh-CN/r0.8/vqe_for_quantum_chemistry.html)
