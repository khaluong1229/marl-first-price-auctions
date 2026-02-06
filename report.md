# Week 4 Report: Differentiable Learning in Multi-Agent Auctions

## Problem Statement

We are looking at whether gradient-based learning agents converge to Nash Equilibrium when simultaneously optimizing bidding strategies in first-price auctions. In a first-price auction, each agent $i$ imaximizes expected utility

$$
U_i(b_i, b_{-i}) = (v_i - b_i) \cdot \mathbb{1}(b_i > \max_{j \neq i} b_j)
$$

, where $v_i$ is a private valuation drawn from $U[0,1]$ and $b_i$ is the submitted bid. The interesting aspect of this problem is that each agent's objective depends on the strategies of all other agents. This creates a coupled optimization problem with non-stationary dynamics.

This problem matters because first-price auctions are prevalent in online advertising and procurement markets. Theoretical guaranetees from single-agent online learning (e.g., Han et al., 2020) do not extend to settingsd where all particpants learn simultaneously. Our success metric is convergence of learned bid multipliers to the symmetric Nash Equilibrium $b^*(v) = \frac{N-1}{N}v$, which for $N=2$ agents yields $w^* = 0.5$. The constraint of this problem is differentiability. The indicator function in the utility requires smooth appropximation since it is a step function with zero gradient almost everywhere. This week, we used synthetically sampled valuations from $U[0,1]$, so no external data is needed. However, if we focus on a particular application of this probelm to the real-world, we can look at data that is representative of valuations. The primary risk is that simultaneous gradient ascent in this setting may oscillate or diverge rather than converge.

## Technical Approach

Each agent $i$ paramterizes a linear bidding strategy $b_i = \sigma(w_i) \cdot v_i$, where $\sigma$ is a sigmoid ensuring the multiplier stays in $(0, 1)$. To handle the non-differentiable winner-determination step function, we replace $\mathbb{1}(b_i > m_i)$ with a sigmoid relaxation $P(\text{win}) \approx \sigma(\beta(b_i - m_i))$, where $\beta = 50$ controls steepness. The smoothed expected utility becomes $\tilde{U}_i = (v_i - b_i) \cdot \sigma(\beta(b_i - m_i))$, which is differentiable with respect to $w_i$.

We implement each agent as aa PuyTorch 'nn.Module' with a single learnable parameter $w_i$, optimized via Adam ($\text{lr} = 0.01$). Each epoch, we sample a batch of 128 valuation vectors from $U[0,1]^N$, compute bids for all agents, evaluate the smoothed utility, and update each agent's parameter by gradient ascent (minimizing negative utility). We use 'retain_graph=True' to allow gradients to go through the shared bid tensor. We validate by comparing learned multipliers against the analytical Nash Equilibrium $w^* = (N-1)/N$ and by plotting the learned bidding function $b(v)$ against the theoretical optimal $b^*(v) = w^* \cdot v$.

## Results

For $N=2$ symmetric agents with $U[0,1]$ valuations, training over 2,000 epochs shows rapid initial convergence from $w \approx 0.71$ down toward the NE of $0.5$ within the first 200 epochs. However, the agents settle into a range of approximately $0.46$–$0.49$, consistently slightly below the theoretical $w^* = 0.5$. Both agents track each other closely, which is expected given symmetric initialization and identical valuation distributions.

The learned bidding function is approximately linear and close to the theorical line $b = 0.5v$, but sits slightly below it. We think this systematic downard bias is likely due to sigmoid smoothing, which softens the winner-takes-all mechanism and incentivizes slightly more conservative bids. The average utility at convergence is $0.17$, consistent with the theoretical expected utility of $1/12 \approx 0.083$ per agent per auction. A current limitation is that we have only tested $N=2$ with uniform valuations and a single optimization algorithm. Resource usage is minimal and training was completed in under a minute on CPU.

## Next Steps

The simplest improvement is a sensivity analysis on $\beta$. The gap between learned multipliers and the theoretical NE might shrink as $\beta$ increases, but a very high $\beta$ may cause gradient instability. Testing $\beta \in \{10, 25, 50, 100, 200\}$ will help us understand this tradeoff and determine whether the bias is part of smoothing or a property of simultaneous gradient dynamics.

We need to scale to $N \in \{3, 5, 10\}$ agents and verify convergence to $w^* = (N-1)/N$ in each case. Additionally, implementing Multiplicative Weights Update (MWU) as a comparison algorithm is important because MWU has known regret guarantees in adversarial settings and will serve as a baseline to contextualize gradient ascent performance. We plan to introduce asymmetric valuation distributions to test whether convergence to asymmetric Nash Equilibrium is achievable.

An open question is whether the oscillation observed in the convergence plot represents actual cycling behavior inherent to simultaneous gradient dynamics in games or if it is noise from stochastic batching. Increasing batch size or using exponential moving averages of the multipliers could help us determine which it is. We also need to implement the $\epsilon$-Nash deviation metric (via a linear program best-response solver) to quantify proximity to equilibriu, beyond simple multiplier comparison.
