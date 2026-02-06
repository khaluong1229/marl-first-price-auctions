# Week 4 Report: Differentiable Learning in Multi-Agent Auctions

## 1. Problem Statement
We are investigating the convergence dynamics of coupled optimization algorithms in **First-Price Sealed-Bid Auctions**. In this setting, $N$ agents compete for an item with a private valuation $v_i$. The winner pays their bid, while losers pay nothing.

This presents a unique optimization challenge because:
1.  **Non-Stationarity:** Each agent's optimal strategy depends on the changing strategies of competitors.
2.  **Non-Differentiability:** The winner determination is a discrete step function (win/loss), making standard gradient-based optimization impossible without modification.
3.  **Coupled Objectives:** The system is a non-cooperative game where we seek a Nash Equilibrium (NE) rather than a global maximum.

**Success Metric:** We measure success by the agents' ability to converge to the theoretical Symmetric Nash Equilibrium (SNE). For $N$ risk-neutral bidders with valuations $v \sim U[0,1]$, the theoretical optimal bid is known to be:

$$
b^*(v) = \frac{N-1}{N} v
$$

## 2. Technical Approach
To enable gradient-based learning in this discrete auction environment, we implemented a **differentiable relaxation** of the auction mechanism.

### Mathematical Formulation
Instead of a hard indicator function $\mathbb{1}(b_i > \max b_{-i})$, we approximate the probability of winning using a sigmoid function:

$$
P(\text{win}) \approx \sigma(\beta (b_i - m_i)) = \frac{1}{1 + e^{-\beta(b_i - m_i)}}
$$

Where $m_i$ is the highest competing bid and $\beta$ is a temperature parameter controlling the approximation's steepness.

### Optimization Algorithm
* **Algorithm:** Online Gradient Ascent.
* **Policy:** We parameterize the bidding strategy as a linear function $b_i = w_i \cdot v_i$, where $w_i$ is a learnable scalar.
* **Loss Function:** Negative Expected Utility.

$$
\mathcal{L}(\theta) = - \mathbb{E}[(v_i - b_i) \cdot P(\text{win})]
$$

* **Implementation:** PyTorch with `torch.sigmoid` for the relaxation and `SGD/Adam` for the parameter updates.

## 3. Initial Results
We successfully implemented a standalone differentiable auction loop in a Jupyter Notebook.

### Convergence Verification
We simulated a 2-agent auction ($N=2$) with Uniform valuations. The theoretical prediction is that agents should learn to bid half their valuation ($w = 0.5$).

**Visual Evidence:**
![Convergence Results](path/to/your_screenshot.png)
*(Note: Replace this path with your actual uploaded image file)*

**Observations:**
1.  **Fast Adaptation:** As shown in the left plot, both agents started with a naive high bid strategy ($w \approx 0.9$) and rapidly descended towards the equilibrium.
2.  **Stability:** The strategies stabilized after approximately 500 epochs.
3.  **Slight Over-Shading:** The agents converged to $w \approx 0.48$, slightly below the theoretical $0.50$. This is an expected artifact of the sigmoid smoothing; the "soft" boundary creates a small incentive to bid conservatively to maximize expected smooth utility.
4.  **Linearity:** The right plot confirms that the agents learned the correct *structure* of the bidding function (linear w.r.t valuation).

## 4. Next Steps & Challenges
With the core optimization loop validated, the next phase focuses on scaling complexity:

1.  **PettingZoo Integration:** Port the standalone logic into a formal `PettingZoo` environment to support standard MARL libraries.
2.  **Mechanism Comparison:** Implement Second-Price and VCG mechanisms to compare convergence rates (Phase 2 of proposal).
3.  **Neural Network Policies:** Replace the simple linear parameter $w$ with a Neural Network ( $\pi_\theta(v)$ ) to allow for learning non-linear strategies in asymmetric valuation environments.
4.  **Temperature Scheduling:** Implement an annealing schedule for the parameter $\beta$ to reduce the "over-shading" bias as training progresses.
