# Self-Critique

## Strengths
- Very clear technical approach with clear, actionable steps.
- Modeled complex market dynamics effectively as mathematical functions that formalize our definition of utility, private values, and our benchmark \(w^*\).
- We recognized the issues with a hard-win function and pivoted by implementing a softer sigmoid function instead.

## Areas for Improvement
- Real-world applicability could be clarified. The current model doesn’t account for many real-world factors such as budgets, bidder heterogeneity, and reserve prices which can strongly affect how applicable our results are beyond the toy setting.
- Results may depend heavily on the smoothing parameter \(\beta\) and the limited setting (\(N=2\)); a \(\beta\) sensitivity sweep and experiments with \(N>2\) are needed to show robustness.
- Our current strategy for evaluating optimality by comparing our learned multiplier with \(w^*\) isn’t robust enough. We can improve this strategy by using an \(\varepsilon\)-Nash metric to better evaluate our current policy beyond simply checking against the multiplier.

## Critical Risks/Assumptions
Some of the critical risks are that we replaced a typical auction winning hard objective (1 → win, 0 → loss) with a softer sigmoid function since gradient-based models do not learn as well with a hard function. As a result, our project may not closely reflect real-world scenarios because we are optimizing a different function than the true auction objective. We are also assuming that each bidder is extremely rational, focused solely on profit maximization and ignoring factors such as irrationality and loss aversion which are highly present in real-world markets, which can cause deviations between our results and real-world scenarios.

## Concrete Next Actions
- Sweep \(\beta \in \{10, 25, 50, 100, 200\}\), plot \(w\) convergence/error vs. \(\beta\), and retrain \(N=3\) to check \(w^* \approx 0.67\); complete by EOW with a Jupyter notebook.
- Implement \(\varepsilon\)-Nash metric via PuLP LP solver for best-response deviation; validate current policy quality beyond multiplier check.
- Polish report: spellcheck, derive \(\partial \tilde{U}_i / \partial w_i\) explicitly, embed convergence/bid plots.

## Resource Needs
Dedicate 1–2 hours to hands-on tutorials specifically for \(\varepsilon\)-Nash computation in games using PuLP's linear programming solver—start with Towards Data Science articles on "PuLP Nash equilibrium Python" or YouTube demos like "game theory LP solver tutorial." This directly unblocks quantifying policy quality beyond simple \(w\) comparison, as current report lacks this metric.

Profile \(N=10\) agent runs on local CPU first (expect ~5–10× slowdown vs. \(N=2\)); if >5 min per 2000 epochs, migrate to Google Colab's free T4 GPU with a self-contained notebook uploading your current PyTorch code. Include `torch.cuda.is_available()` check and a `batch_size=256` scaling test to confirm feasibility before \(\beta\) sweeps.

Review concise resources (30–60 min total): Roughgarden's "Multiplicative Weights Survey" (arXiv:0907.2874) for MWU pseudocode, plus a 10-line Python snippet adapting it to bid updates via regret matching. This provides a no-regret baseline to contrast gradient ascent's convergence, addressing the report's single-algorithm limitation.
