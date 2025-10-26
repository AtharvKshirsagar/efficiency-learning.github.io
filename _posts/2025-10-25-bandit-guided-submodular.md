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
### ⚙️ Methodology in a Nutshell: ONLINESUBMOD

We propose **ONLINESUBMOD**, a *bandit-guided submodular curriculum* framework for adaptive data selection.  
The key idea is to dynamically choose *which submodular utility function* to optimize at each step — guided by validation feedback — rather than following a fixed or hand-crafted curriculum.

At a high level, ONLINESUBMOD maintains a pool of candidate **submodular utility functions**

$$
\{f_1, f_2, \dots, f_K\},
$$

where each $f_k$ models a different prior such as **diversity**, **representativeness**, or **uncertainty**.  
At each iteration, the algorithm picks one function $f_{a_t}$ to construct the next training subset.

---

#### 🎯 Utility-Based Subset Selection

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

#### 🧠 Bandit Objective

Each submodular utility function acts as an **arm** in a multi-armed bandit (MAB) framework.  
At each round $t$, ONLINESUBMOD must decide **which arm (utility function)** to pull.

The objective is to maximize the cumulative expected validation reward:

$$
\max_{\pi} \; \mathbb{E}\!\left[ \sum_{t=1}^{T} r_t \right],
$$

where $\pi$ is the *arm-selection policy*.  
Each arm corresponds to a utility $f_k$, and its expected reward is modeled as

$$
\mathbb{E}[r_t \mid a_t = k] = g(f_k),
$$

which links the *quality of the chosen subset* directly to the *observed validation improvement*.

---

#### 🧮 Submodular Maximization Step

For a chosen function $f_k$, the framework solves a **budget-constrained submodular maximization** problem:

$$
S_t = \arg\max_{S \subseteq D_t,\; |S| \le b} f_k(S),
$$

subject to the **diminishing-returns property** of submodularity:

$$
f_k(A \cup \{x\}) - f_k(A)
  \ge
  f_k(B \cup \{x\}) - f_k(B),
  \quad A \subseteq B.
$$

This property ensures that greedy selection yields a provable approximation to the optimal subset.  
Specifically, the greedy algorithm achieves a $(1 - 1/e)$-approximation:

$$
f_k(S_{\text{greedy}}) \ge (1 - 1/e) f_k(S^*).
$$

This provides theoretical efficiency guarantees while remaining computationally lightweight (less than 1 ms overhead compared to a gradient step).

---

#### ⚖️ No-Regret Guarantee via EXP3

The arm-selection policy uses the **EXP3 (Exponential Weights for Exploration and Exploitation)** algorithm to adaptively update arm probabilities based on observed rewards.

Weights are updated as:

$$
w_{k,t+1} = w_{k,t} \exp\!\big(\eta\,\hat{r}_{k,t}\big),
$$

where  

$$
\hat{r}_{k,t}
  = \frac{r_t\,\mathbb{I}[a_t = k]}{p_{k,t}},
  \qquad
  p_{k,t}
  = \frac{w_{k,t}}{\sum_j w_{j,t}}.
$$

Here:
- $p_{k,t}$ is the probability of selecting arm $k$ at time $t$,  
- $\eta$ is the learning rate, and  
- $\mathbb{I}[a_t = k]$ is the indicator for the chosen arm.

The algorithm satisfies a **no-regret guarantee**:

$$
\text{Regret}(T)
  = \mathcal{O}\!\big(\sqrt{T\,K\,\log K}\big),
$$

ensuring convergence toward the best fixed submodular function in hindsight.  
This means ONLINESUBMOD adaptively learns *which notion of utility* is most effective for improving validation performance during training.

---

#### 🧩 Summary of Core Ideas

- **Bandit Formulation:** Treat each submodular utility as a competing “expert” and learn which one helps most.  
- **Utility-Driven Selection:** Subsets are explicitly chosen to maximize a validation-based utility function.  
- **No-Regret Policy:** EXP3 guarantees long-term performance close to the best static utility.  
- **Efficiency:** Submodular maximization adds negligible computational cost while improving generalization.
---
