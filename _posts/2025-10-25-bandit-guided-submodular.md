---
layout: post
title: "Bandit-Guided Submodular Curriculum for Adaptive Subset Selection"
math: true
categories: [Research, Curriculum Learning, Subset Selection]
---

### Problem Statement

Traditional curriculum learning proceeds from easy to hard samples, yet defining
a reliable notion of difficulty remains elusive. Prior work has used submodular
functions to induce difficulty scores in curriculum learning. We reinterpret adaptive
subset selection and formulate it as a multi-armed bandit problem, where each
arm corresponds to a submodular function guiding sample selection. We introduce
ONLINESUBMOD, a novel online greedy policy that optimizes a utility-driven re-
ward and provably achieves no-regret performance under various sampling regimes.
Empirically, ONLINESUBMOD outperforms both traditional curriculum learning
and bi-level optimization approaches across vision and language datasets, show-
ing superior accuracy-efficiency tradeoffs


Formally, given a dataset $D = \{(x_i, y_i)\}_{i=1}^N$, the goal is to select a subset  
$S_t \subseteq D$ at training step $t$ that maximizes the validation utility $U(S_t)$ while maintaining efficiency.  
Curriculum learning can be posed as the optimization:

$$
\max_{S_t \subseteq D} f(S_t) = \mathbb{E}_{(x,y)\sim S_t} [ \ell(x, y; \theta_t) ],
$$

where $f(\cdot)$ is a **submodular function** encoding representativeness and diversity.

---

## Our Solution: ONLINESUBMOD

### Framing Selection as a Multi-Armed Bandit

We maintain a pool of candidate submodular utility functions:

$$
\{f_1, f_2, \dots, f_K\},
$$

where each function $f_k$ models a different prior — **diversity**, **representativeness**, or **uncertainty**. At each iteration $t$, selecting a function is treated as pulling an arm in a multi-armed bandit:

- Each arm corresponds to a submodular function.
- The reward is the validation performance improvement after training on the selected subset.

---

### Step 1: Utility-Based Subset Selection

Given the dataset $D_t$ at step $t$, we select a subset $S_t$ of size at most $b$ by maximizing the chosen submodular function:

$$
S_t = \arg\max_{S \subseteq D_t, |S| \le b} f_{a_t}(S),
$$

where $f_{a_t}$ is the selected utility function.  
The **submodular property** ensures efficient greedy approximation:

$$
f_k(A \cup \{x\}) - f_k(A) \ge f_k(B \cup \{x\}) - f_k(B), \quad A \subseteq B,
$$

and guarantees a $(1 - 1/e)$-approximation:

$$
f_k(S_{\text{greedy}}) \ge (1 - 1/e) f_k(S^*).
$$

---

### Step 2: Validation-Driven Reward

After training on subset $S_t$, we measure the validation improvement:

$$
r_t = \Delta \ell_{\text{val}} = \ell_{\text{val}}(\theta_{t-1}) - \ell_{\text{val}}(\theta_t),
$$

which serves as our reward signal, linking subset quality to generalization performance.

---

### Step 3: Bandit Objective

Our goal is to maximize cumulative expected validation reward:

$$
\max_\pi \; \mathbb{E} \left[ \sum_{t=1}^T r_t \right],
$$

where $\pi$ is our arm-selection policy. The expected reward for arm $k$ is modeled as:

$$
\mathbb{E}[r_t \mid a_t = k] = g(f_k).
$$

---

### The Mathematics Behind the Utility Function

**Gradient-Based Utility Approximation:**

For a training batch $B_t$ and validation instance $z_{\text{val}}$:

$$
U_t(B_t, z_{\text{val}}) = \ell(z_{\text{val}}, \theta_t) - \ell(z_{\text{val}}, \tilde{\theta}_{t+1}(B_t)),
$$

where

$$
\tilde{\theta}_{t+1}(B_t) = \theta_t - \eta_t \nabla_\theta \left[ \frac{1}{|B_t|} \sum_{z \in B_t} \ell(z, \theta_t) \right].
$$

**First-Order Approximation:**

Marginal utility gain of adding instance $z_i$:

$$
\Delta U_t(z_i \mid B_t^{(<i)}, z_{\text{val}}) \approx \eta_t \nabla_\theta \ell(z_i, \theta_t) \cdot \nabla_\theta \ell(z_{\text{val}}, \theta_{t+1}(B_t^{(<i)})).
$$

**Second-Order Refinement:**

Using a Taylor expansion:

$$
\Delta U_t \approx 
\underbrace{\eta_t g_{\theta_t}(z_i) \cdot g_{\theta_t}(z_{\text{val}})}_{\text{Gradient Influence (Term I)}} 
- 
\underbrace{\eta_t^2 g_{\theta_t}(z_i)^\top H_{z_{\text{val}}}(\theta_t) \left(\frac{1}{|B_t^{(<i)}|} \sum_{z \in B_t^{(<i)}} g_{\theta_t}(z)\right)}_{\text{Hessian-Weighted Similarity (Term II)}}.
$$

---
