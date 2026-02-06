# marl-first-price-auctions

# Auction Mechanism Design for Carbon Permits Under Information Asymmetry

## Executive Summary
Carbon permit auctions are the backbone of emissions trading systems like the EU ETS, which generates over €40 billion annually. A debate that appears in environmental economics is determining which auction format best allocates these permits when firms have private information about their costs. While the EU currently uses Uniform-Price auctions, economic theory suggests this format may suffer from demand reduction, where large firms intentionally underbid on permits to artificially lower the market-clearing price.

This project utilizes Multi-Agent Reinforcement Learning (MARL) to simulate and compare three different auction mechanisms: Uniform-Price, Discriminatory (Pay-as-Bid), and Vickrey-Clarke-Groves (VCG). By modeling firms as learning agents that submit two-tier demand schedules, we can empirically test whether the strategic inefficiences predicted by theory actually occure in a dynamic market enviornment.

## Theoretical Background

### The Economics of Carbon Markets
In a "Cap-and-Trade" system, regulators set a strict limit (Cap) on total emissions and issue permits corresponding to that limit. Firms must hold a permit for every ton of CO2 they emit. This creates a market: firms that can reduce emissions cheaply will do so and sell their permits, while firms with high reduction costs will buy permits.

The crucial decision for regulators is how to distribute these permits initially. Modern systems favor auctions over giving permits away for free (grandfathering) to avoid windfall profits for polluters and to generate government revenue for climate initiatives.

### The Three Auction Mechanisms
We compare three distinct formats for selling these permits:
* Uniform-Price Auction:
  * How it works: Everyone submits bids. The auctioneer ranks them highest to lowest and accepts bids until the supply is gone. The price for everyone is set by the lowest winning bid (the clearing price).
  * The flaw (demand reduction): Because the price is set by the marginal (last) bid, large bidders have an incentive to bid truthfully on their "must-have" permits but bid artificially low on their extra permits. If they are successful, this low bid sets the price for all the permits they won, saving them massive amounts of money.
* Discriminatory Auction:
  * How it works: Winners pay exactly what they bid. If you bid $80, you pay $80, even if the market clearing price was $40.
  * The flaw (demand reduction): There is no incentive for "demand reduction" because lowering one bid doesn't change the price you pay for other bids. However, firms face the "Winner's Curse"—the fear of overpaying—which leads everyone to shade their bids downwards.
* Vickrey-Clarke-Groves (VCG) Auction:
  * How it works: This is a theoretical ideal where winners pay an amount equal to the "harm" their participation caused to other bidders (the social externality).
  * The flaw (demand reduction): In theory, the optimal strategy in a VCG auction is to bid your true value exactly. It guarantees 100% efficiency but is complex and rarely used in the real world.

### Why Two-Tier Bids?
Firms have a decreasing marginal value for permits. The first permits are critical and without them, a factory might have to shut down. A firm is willing to pay a very high price for these. The later permits cover marginal production. If the price is too high, the firm can just cut back production slightly, so these are less valuable.

To capture this, our model allows agents to submit a Two-Tier Demand Schedule. Agents will set a high price for their critical needs and a lower price for their optional needs. This structure is crucial for observing the Demand Reduction strategy.

## Mathematical Formulation

### Firm Characteristics
We model $N$ heterogeneous firms. Each firm $i$ has a private Marginal Abatement Cost (MAC) coefficient, $c_i$, which determines how expensive it is for them to reduce pollution. This cost is private information, and the regulator does not know it.

The cost to reduce emissions by amount $a$ is quadratic: $$C_i(a) = \frac{c_i}{2}a^2$$ This implies a linear Marginal Abatement Cost curve: $MAC = c_i \cdot a$. As a firm reduces more emissions, it becomes increasingly expensive to reduce the next unit.

### Valuation of Permits
The value of receiving a permit is equal to the abatement cost the firm avoids paying. Because costs are quadratic, the value of permits is decreasing. The value $v_i(k)$ of the $k$-th permit is: $$v_i(k) = c_i \cdot (e_i^0 - k + 1)$$ Where $e_i^0$ is the firm's baseline emissions.

### Optimization Problem
Each firm acts as an independent agent trying to maximize its own profit $\pi_i$: $$\pi_i = \text{Total Value of Permits Won} - \text{Payment to Regulator}$$ $$\pi_i = \sum_{k=1}^{x_i} v_i(k) - \text{Payment}_i$$ The Payment depends on the auction mechanism rules (Uniform, Discriminatory, or VCG).

## Methodology: MARL

### Learning Approach
We model the auction as a partially observable Markov game. We use Independent Proximal Policy Optimization (IPPO), a standard Reinforcement Learning algorithm.
* Decentralized Learning: Each firm has its own brain (neural network policy). They do not share information or coordinate training, mimicking the real world where firms are competitors.
* Partial Observability: An agent knows its own costs and the history of market prices, but it cannot see the private costs or bids of its competitors.

### Action Space
Unlike simple models where agents pick a single number, our agents output a multi-dimensional action representing their demand schedule:
* $p^H$: The price for their high-value tier (critical permits).
* $q^H$: The quantity of critical permits needed.
* $p^L$: The price for their low-value tier (optional permits).
* $q^L$: The quantity of optional permits needed.

The environment enforces a monotonicity constraint ($p^H \ge p^L$) to ensure the demand curve logically slopes downward.

## Implementation Strategy
The project is divided into four phases.

### Environment Development
We will build a custom simulation environment that can switch between the three auction rules.
* Objective: Validate that the "Market Clearing" logic works. For example, in a Uniform-Price auction, does shading the low-tier bid correctly lower the clearing price for the high-tier permits?
* Deliverable: A suite of unit tests verifying the economic logic against theoretical examples.

### Single-Agent Training
We will train a single learning agent against "dummy" opponents with fixed strategies.
* Objective: Confirm that the agent can actually learn to make money. It should learn to bid truthful values when playing against non-strategic opponents.
* Deliverable: Learning curves showing profit maximization.

### Multi-Agent Learning
We will introduce 10 learning agents into the same market.
* Objective: Observe emergent behavior. Do Uniform-Price agents learn to shade their bids more than Discriminatory agents? Does the market stabilize?
* Deliverable: Trained policies for all 10 agents across all three mechanism types.

### Analysis and Metrics
We will evaluate the trained markets using two primary metrics:
* Allocative Efficiency: A ratio (0 to 1) measuring if the permits went to the firms who valued them most (High Efficiency) or if they were misallocated to low-value firms (Low Efficiency).
* Demand Reduction Index (DRI): A novel metric calculating the difference between shading on the high tier vs. the low tier. A positive DRI indicates that firms are manipulating the market.
