# marl-first-price-auctions

# Project Proposal: Multi-Agent Learning Dynamics in First-Price Auctions

## 1. Abstract
We propose to study the convergence properties of coupled optimization algorithms in multi-agent first-price auctions. Building on Han et al.'s (2020) single-agent online learning framework, we extend their gradient-based bidding algorithm to settings where multiple agents simultaneously optimize their bidding strategies. This creates a system of coupled optimizers where each agent's objective function is non-stationary, depending on the evolving strategies of competitors. We will implement the auction environment as a PettingZoo-compatible MARL benchmark, reproduce Han et al.'s single-agent results, and empirically analyze convergence to Nash equilibrium (NE) under different auction mechanisms and optimization algorithms.

## 2. Background and Motivation

### 2.1 Problem Context
First-price auctions are ubiquitous in online advertising, spectrum allocation, and procurement. Unlike second-price auctions where truthful bidding is optimal, first-price auctions require strategic bidding where agents must balance:
1.  Bidding high enough to win.
2.  Bidding low enough to maximize surplus.

This creates a non-trivial optimization problem, especially when valuations vary over time, competitors' strategies are unknown, and the environment is non-stationary.

### 2.2 Single-Agent Learning (Han et al., 2020)
Han et al. developed the first minimax-optimal algorithm for learning to bid in adversarial first-price auctions, achieving $O(\sqrt{T})$ regret. Their approach uses online gradient ascent with expert chaining and Lipschitz constraints. However, their work assumes a single learning agent competing against fixed or adversarial opponents, ignoring the dynamics of mutual adaptation.

### 2.3 The Multi-Agent Gap
Real-world auction platforms feature multiple adaptive bidders. This creates a fundamentally different problem:
* Each agent's optimization objective depends on others' current strategies.
* Standard convergence guarantees for static optimization do not necessarily hold.

**Research Gap:** When all agents optimize simultaneously using gradient-based methods, do they converge? If so, does the resulting equilibrium align with the theoretical Nash Equilibrium?

## 3. Problem Formulation

### 3.1 Single-Agent Optimization Problem
Each bidder $i$ at time $t$ solves:

$$
\max_{b_{i,t} \in [0,1]} \mathbb{E}[U_{i,t}(b_{i,t})] = \mathbb{E}[v_{i,t} \cdot \mathbb{1}(b_{i,t} \geq m_{i,t}) - b_{i,t} \cdot \mathbb{1}(b_{i,t} \geq m_{i,t})]
$$

Where:
* $b_{i,t}$: Agent $i$'s bid (decision variable).
* $v_{i,t}$: Agent $i$'s private valuation.
* $m_{i,t}$: Highest competing bid against agent $i$.
* $\mathbb{1}(\cdot)$: Indicator function for winning (step function).

**Regret Criterion:**

$$
R_T = \sum_{t=1}^T U^{\ast}_t - \sum_{t=1}^T U_t
$$

Where $U^{\ast}_t$ is the utility of the best fixed Lipschitz bidding policy in hindsight.

### 3.2 Multi-Agent Coupled Optimization
With $N$ learning agents, the system becomes coupled. For each agent $i$, the "market price" $m_{i,t}$ is no longer exogenous but defined as:

$$
m_{i,t} = \max_{j \neq i} b_{j,t}
$$

The optimization goal becomes:

$$
\forall i \in \{1, \ldots, N\}: \quad \max_{\theta_i} \sum_{t=1}^T U_i(b_i(\theta_i; v_{i,t}), b_{-i}(\theta_{-i}; v_{-i,t}))
$$

### 3.3 Equilibrium as Fixed Point
A Nash equilibrium $\theta^{\ast} = (\theta^{\ast}_1, \ldots, \theta^{\ast}_N)$ satisfies:

$$
\theta^{\ast}_i \in \arg\max_{\theta_i} U_i(\theta_i, \theta^{\ast}_{-i}) \quad \forall i
$$

**Research Question:** Do gradient-based learning dynamics converge to this fixed point, or do they exhibit cyclic/chaotic behavior?
## 4. Methodology

### 4.1 Environment Development
We will develop a **PettingZoo-Compatible Auction Environment**:

```python
class AuctionEnvironment(ParallelEnv):
    """
    Multi-agent first-price auction environment
    - N agents bid simultaneously
    - Supports multiple mechanisms (1st price, 2nd price, VCG)
    """
    def __init__(self, n_agents, mechanism='first_price'):
        self.n_agents = n_agents
        self.mechanism = mechanism
        
    def step(self, actions):
        # actions = bids from all agents
        # Returns: observations, rewards, dones, infos
```

### 4.2 Optimization Algorithms
A key challenge is that the auction outcome $\mathbb{1}(b_i > m_i)$ is non-differentiable. We will implement:

**1. Gradient Ascent with Sigmoid Smoothing:**
To enable gradient-based updates, we approximate the hard win/loss step function with a differentiable sigmoid:
$$
P(\text{win}) \approx \sigma(\beta(b_i - m_i)) = \frac{1}{1 + e^{-\beta(b_i - m_i)}}
$$
Where $\beta$ is a temperature parameter controlling the steepness.
* **Update:** $\theta_{t+1} = \Pi_{\Theta}[\theta_t + \alpha_t \nabla_\theta \tilde{U}_t(\theta_t)]$

**2. Multiplicative Weights Update (MWU):**
* Discretize the bid space.
* Maintain a probability distribution over bids.
* Update weights exponentially based on realized utility.
* *Note:* Serves as a robust baseline with known regret guarantees.

**3. Policy Gradient (REINFORCE/PPO):**
* Treat the bid generation as a stochastic policy $\pi_\theta(b|v)$.
* Optimize expected return via score function estimator.

### 4.3 Convergence Analysis Methods
**1. Analytical Benchmark Validation:**
For symmetric agents with valuations $v \sim U[0,1]$, the theoretical symmetric Nash Equilibrium (SNE) is known:
$$
b^*(v) = \frac{N-1}{N}v
$$
We will measure deviation from this analytical solution to validate convergence.

**2. Strategy Distance Metrics:**
$$
d_t = \frac{1}{N} \sum_{i=1}^N \|\theta_{i,t} - \theta_{i,t-1}\|_2
$$

**3. $\epsilon$-Nash Deviation:**
Using a discretized solver (Linear Program) to calculate the best response $BR_i$ to current opponent strategies and measuring the utility gap.

## 5. Experimental Design

### 5.1 Phase 1: Reproduction & Validation
* **Goal:** Validate implementation against Han et al. (single-agent) and Analytical SNE (multi-agent).
* **Setup:**
    1.  Reproduce $O(\sqrt{T})$ regret for 1 agent vs. adversary.
    2.  Validate that 2 symmetric agents converge to bidding $b = v/2$ (Uniform priors).

### 5.2 Phase 2: Multi-Agent Convergence
* **Goal:** Study convergence when all agents learn simultaneously.
* **Variables:**
    * Agents: $N \in \{2, 3, 5, 10\}$
    * Algorithms: Gradient Ascent (Smoothed) vs. MWU
    * Valuation Distributions: Uniform (symmetric) vs. Beta (asymmetric).
* **Metrics:** Convergence rate to stable strategy; variance of strategies near convergence.

### 5.3 Phase 3: Mechanism Comparison
* **Hypothesis:** VCG (truthful) converges fastest as strategies are decoupled. First-price auctions (coupled) will exhibit slower or oscillatory convergence.
* **Mechanisms:** First-price, Second-price, VCG.

## 6. Expected Contributions
1.  **Empirical Characterization:** Detailed analysis of the regions of attraction and stability for gradient dynamics in auctions.
2.  **Algorithm Comparison:** Evaluation of "Smoothed Gradient Ascent" vs. "Multiplicative Weights" in non-stationary auction games.
3.  **Open Source Benchmark:** A robust PettingZoo environment for auction research.

## 7. Technical Approach

### 7.1 Code Structure
```text
marl-auctions/
├── environments/
│   ├── auction_env.py          # PettingZoo environment
│   └── mechanisms.py           # 1st price, 2nd price, VCG
├── agents/
│   ├── gradient_ascent.py      # Smoothed gradient approach
│   ├── multiplicative_weights.py
│   └── policy_gradient.py      # Deep RL approach
├── analysis/
│   ├── equilibrium_solver.py   # Computes ground-truth via LP/Nashpy
│   ├── convergence.py          # Metrics and plotting
│   └── welfare.py              # Social welfare analysis
└── experiments/
    ├── reproduce_han.py
    └── multi_agent.py
```

### 7.2 Validation Strategy
* **Unit Tests:** Verify mechanism payments and allocation rules.
* **Theoretical Checks:** Ensure single-agent regret matches theoretical bounds; ensure symmetric multi-agent cases match $b = \frac{N-1}{N}v$.

## 8. Timeline
* **Week 1-2:** Environment setup & Single-agent baseline (Han et al.).
* **Week 3-4:** Implementation of Smoothed Gradient Ascent & MWU. Validation against analytical NE.
* **Week 5-7:** Multi-agent experiments (Phase 2).
* **Week 8-9:** Mechanism comparison (Phase 3) & Sensitivity analysis.
* **Week 10-12:** Final analysis, plotting, and report writing.
