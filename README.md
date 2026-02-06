# marl-first-price-auctions

## Abstract
We are studying the convergence properties of coupled optimization algorithms in multi-agent first-price auctions. Building on Han et al.'s (2020) single-agent online learning framework, we will extend their gradient-based bidding algorithm to settings where multiple agents simultaneously optimize their bidding strategies. This creates a system of coupled optimizers where each agent's optimization problem depends on others' evolving strategies, a form of distributed game-theoretic optimization. We will implement the acution environment as a PettingZoo-compatible MARL benchmark, reproducing the paper's single-agent results, and empirically analyze convergence to Nash equilibrium under different auction mechanisms and optimization algorithms.

## Background and Motivation

### Problem Context
First-price auctions are common in many areas, from online advertising to procurement. Unlike second-price auctions where truthful bidding is optimal, first-price auctions require strategic bidding where agents must balance between bidding high enough to win and bidding low enough to maximize surplus. This creates a non-trivial optimization problem for each bidder, especially when valuations vary over time, competitors' strategies are unknown, and the environment is non-stationary.

### Single-Agent Learning
Han et al. (2020) developed a minimax-optimal algorithm for learning to bid in adversarial first-price auctions, achieving O(√T) regret. Their approach uses online gradient ascent with expert chaining, Lipschitz constraints on bidding policies, and no distributional assumptions about compeitors. However, their work assumes a single learning agent competing against fixed or adversarial opponents.

### The Multi-Agent Gap
Real-world auction platforms feature multiple adaptive bidders, each running optimization algorithms. This creates a different problem where each agent's optimization objective depends on others' current strategies, the optimization landscape is non-stationary for all agents, and standard convergence guarantees may not hold.

Key Question: When all agents optimize simultaneously using gradient-based methods, do they converge? To what equilibrium? How do different auction mechanisms affect convergence dynamics?

## Problem Formulation

### Single-Agent Optimization Problem
Each bidder i at time t solves:

$$\max_{b_{i,t} \in [0,1]} \mathbb{E}[U_{i,t}(b_{i,t})] = \mathbb{E}[v_{i,t} \cdot \mathbb{1}(b_{i,t} \geq m_{i,t}) - b_{i,t} \cdot \mathbb{1}(b_{i,t} \geq m_{i,t})]$$

where:
- $b_{i,t}$ = agent i's bid (decision variable)
- $v_{i,t}$ = agent i's private valuation
- $m_{i,t}$ = highest competing bid against agent i
- $\mathbb{1}(\cdot)$ = indicator function for winning

Optimization Method: Online gradient ascent on the cumulative utility

Regret Criterion:
$$R_T = \sum_{t=1}^T U^*_t - \sum_{t=1}^T U_t$$

where $U^*_t$ is the utility of the best fixed Lipschitz bidding policy in hindsight.

### Multi-Agent Coupled Optimization
With N learning agents, the system becomes:

$$\text{For each } i \in \{1, \ldots, N\}: \quad \max_{\theta_i} \sum_{t=1}^T U_i(b_i(\theta_i; v_{i,t}), b_{-i}(\theta_{-i}; v_{-i,t}))$$

where:
- $\theta_i$ = parameters of agent i's bidding policy
- $b_i(\theta_i; v)$ = agent i's bid given valuation v
- $b_{-i}$ = other agents' bids (which depend on their learning)

Key Challenge: Each agent's optimization problem is **coupled** through the auction outcome.

### Equilibrium as Fixed Point

A Nash equilibrium $\theta^* = (\theta^*_1, \ldots, \theta^*_N)$ satisfies:

$$\theta^{*}_{i} \in \arg\max_{\theta_i} U_i(\theta_i, \theta^*_{-i}) \quad \forall i$$

This is a fixed-point problem in the space of bidding strategies.

Research Question: Do gradient-based learning dynamics converge to this fixed point?

## Methodology

### Environment Development
We will create a multi-agent first-price auction environment where $N$ agents bid simultaneously and that supports multiple mechanisms (1st price, 2nd price, VCG). Features include switchable auction mechanisms, configurable valuation distributions, and support for partial observability (win/loss only vs full feedback).

### Optimization Algorithms to Implement
We will implement online gradient ascent, multiplicative weights updating, and policy gradients.

### Multi-Agent Training Protocol
Each agent computes a bid. The Auction determines winner and payments. Each agent updates independently.

### Convergence Analysis Methods
* Strategy Distance Metrics: $$d_t = \frac{1}{N} \sum_{i=1}^N \|\theta_{i,t} - \theta_{i,t-1}\|_2$$
* Nash Equilibrium Distance:
  * Compute best-response for each agent given others' current strategies
  * Measure: $\epsilon$-Nash deviation = $\max_i [U_i(BR_i, \theta_{-i}) - U_i(\theta_i, \theta_{-i})]$
* Social Welfare Metrics:
  * Total surplus: $\sum_{i=1}^N U_i$
  * Auction revenue
  * Efficiency: Ratio of value to winner vs maximum possible value
* Lyapunov Function (if possible):
  * Design function $V(\theta)$ that decreases along optimization trajectory
  * Proves convergence if $V$ is bounded below
 
## Experimental Design

### Reproduction Strategy
Goal: Validate our implementation against Han et al.'s single-agent results

Setup:
* 1 learning agent vs. fixed bidding strategies
* Adversarial competitor bids
* Measure regret: $R_T = O(\sqrt{T})$?

**Success Criterion:** Match reported regret bounds from Han et al.

### Multi-Agent Convergence
Goal: Study convergence when all agents learn

Variables:
* Number of agents: N ∈ {2, 3, 5, 10}
* Optimization algorithms: {Gradient Ascent, Multiplicative Weights, Policy Gradient}
* Learning rates: {0.01, 0.1, 0.5}
* Valuation distributions: {Uniform[0,1], Beta(2,2), Asymmetric}

### Mechanism Comparison
* First-price auction
* Second-price auction (Vickrey)
* VCG mechanism

### Sensitivity Analysis
Variables to test for sensitivity:
* Learning rate schedules: constant vs. $O(1/\sqrt{t})$ decay
* Exploration noise
* Information feedback: full vs. binary (win/loss only)

## Research Questions
1. Convergence: Do coupled gradient-based optimizers converge in multi-agent auctions? Under what conditions (learning rates, agents, mechanisms)?
2. Convergence rate: How fast? Compare to single0agent rate. Does adding agents slow convergence?
3. Equilibrium quality: What do they converge to? Nash equilibrium? Socially optimal? revenue-maximizing for auctioneer?
4. Algorithm comparison: Which optimization method works best? Gradient ascent vs multiplicative weights vs deep RL. Trade-offs of speed vs equilibrium quality.
5. Mechanism design: Which auction mechanism leads to best learning dynamics? VCG vs first-price vs second-price. Relationship to incentive compatibility.

