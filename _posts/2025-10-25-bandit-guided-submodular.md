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
ing superior accuracy-efficiency tradeoffs.

Formally, given a dataset $D = \{(x_i, y_i)\}_{i=1}^N$, the goal is to select a subset  
$S_t \subseteq D$ at training step $t$ that maximizes the validation utility $U(S_t)$ while maintaining efficiency.  
Curriculum learning can be posed as the optimization:

$$
\max_{S_t \subseteq D} f(S_t) = \mathbb{E}_{(x,y)\sim S_t} [ \ell(x, y; \theta_t) ],
$$

where $f(\cdot)$ is a **submodular function** encoding representativeness and diversity.

---

### Methodology in a Nutshell: ONLINESUBMOD

We propose **ONLINESUBMOD**, a *bandit-guided submodular curriculum* framework for adaptive data selection.  
The key idea is to dynamically choose *which submodular utility function* to optimize at each step — guided by validation feedback — rather than following a fixed or hand-crafted curriculum.

At a high level, ONLINESUBMOD maintains a pool of candidate **submodular utility functions**:

$$
\{f_1, f_2, \dots, f_K\},
$$

where each $f_k$ models a different prior such as **diversity**, **representativeness**, or **uncertainty**.  
At each iteration, the algorithm picks one function $f_{a_t}$ to construct the next training subset.

---

#### Utility-Based Subset Selection

Given a dataset at step $t$, $D_t$, the model selects a subset $S_t$ of size at most $b$ that maximizes a chosen submodular utility function:

$$
S_t = \arg\max_{S \subseteq D_t,\; |S| \le b} f_{a_t}(S),
$$

where $f_{a_t}(S)$ quantifies the “value” of subset $S$ in terms of coverage, representativeness, or information gain.  
This general formulation allows ONLINESUBMOD to encompass many existing curricula (e.g., loss-based, uncertainty-based, diversity-based) as special cases.

Once the subset $S_t$ is chosen, the model trains on it and evaluates the **validation improvement**:

$$
r_t = \Delta \ell_{\text{val}}
    = \ell_{\text{val}}(\theta_{t-1})
    - \ell_{\text{val}}(\theta_t),
$$

where $\ell_{\text{val}}(\theta)$ denotes the validation loss after parameters $\theta$.  
This validation gain $r_t$ serves as a *reward signal* to guide the next selection.

---

#### Bandit Objective

Each submodular utility function acts as an **arm** in a multi-armed bandit (MAB) framework.  
At each round $t$, ONLINESUBMOD must decide **which arm (utility function)** to pull.

The objective is to maximize the cumulative expected validation reward:

$$
\max_{\pi} \; \mathbb{E}\!\left[ \sum_{t=1}^{T} r_t \right],
$$

where $\pi$ is the *arm-selection policy*.  
Each arm corresponds to a utility $f_k$, and its expected reward is modeled as:

$$
\mathbb{E}[r_t \mid a_t = k] = g(f_k),
$$

linking the *quality of the chosen subset* directly to the *observed validation improvement*.

---

#### Utility Function Formulations

We compute **gradient-based approximations** of submodular utility functions to capture each instance's contribution:

**Batch-Level Utility:**

For a training batch $B_t$ and validation instance $z_{\text{val}}$:

$$
U_t(B_t, z_{\text{val}}) = \ell(z_{\text{val}}, \theta_t) - \ell(z_{\text{val}}, \tilde{\theta}_{t+1}(B_t)),
$$

where 

$$
\tilde{\theta}_{t+1}(B_t) = \theta_t - \eta_t \nabla_\theta \left[ \frac{1}{|B_t|} \sum_{z \in B_t} \ell(z, \theta_t) \right].
$$

**First-Order Approximation (Marginal Utility Gain):**

Adding a candidate instance $z_i$ to a partially constructed batch $B_t^{(<i)}$:

$$
\Delta U_t(z_i \mid B_t^{(<i)}, z_{\text{val}}) \approx \eta_t \nabla_\theta \ell(z_i, \theta_t) \cdot \nabla_\theta \ell(z_{\text{val}}, \theta_{t+1}(B_t^{(<i)})).
$$

**Second-Order Refinement (Gradient & Hessian):**

Using Taylor expansion, we can interpret contributions as:

$$
\Delta U_t \approx 
\underbrace{\eta_t g_{\theta_t}(z_i) \cdot g_{\theta_t}(z_{\text{val}})}_{\text{Gradient Influence (Term I)}} 
- 
\underbrace{\eta_t^2 g_{\theta_t}(z_i)^\top H_{z_{\text{val}}}(\theta_t) \left(\frac{1}{|B_t^{(<i)}|} \sum_{z \in B_t^{(<i)}} g_{\theta_t}(z)\right)}_{\text{Hessian-Weighted Similarity (Term II)}}.
$$

- **Term I** measures how much the gradient of $z_i$ aligns with the validation loss gradient.  
- **Term II** accounts for interactions between the candidate instance and the batch through the Hessian, penalizing redundancy.

These utility approximations allow **ONLINESUBMOD** to evaluate candidate samples efficiently and adaptively, combining representativeness, diversity, and validation relevance in a principled manner.

---

#### Submodular Maximization Step

For the selected function $f_k$, we solve:

$$
S_t = \arg\max_{S \subseteq D_t,\; |S| \le b} f_k(S),
$$

with diminishing-returns property:

$$
f_k(A \cup \{x\}) - f_k(A)
  \ge
  f_k(B \cup \{x\}) - f_k(B),
  \quad A \subseteq B.
$$

Greedy selection achieves a $(1 - 1/e)$-approximation:

$$
f_k(S_{\text{greedy}}) \ge (1 - 1/e) f_k(S^*).
$$

---

#### No-Regret Guarantee via EXP3

The arm-selection policy uses **EXP3**:

$$
w_{k,t+1} = w_{k,t} \exp\!\big(\eta\,\hat{r}_{k,t}\big),
$$

with

$$
\hat{r}_{k,t}
  = \frac{r_t\,\mathbb{I}[a_t = k]}{p_{k,t}}, \quad
  p_{k,t} = \frac{w_{k,t}}{\sum_j w_{j,t}}.
$$

The algorithm ensures **no-regret**:

$$
\text{Regret}(T) = \mathcal{O}\!\big(\sqrt{T\,K\,\log K}\big),
$$

allowing ONLINESUBMOD to converge to the best submodular function in hindsight.

---
