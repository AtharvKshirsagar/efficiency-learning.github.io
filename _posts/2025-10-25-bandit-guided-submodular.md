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

We propose **ONLINESUBMOD**, a *bandit-guided submodular curriculum* framework that adaptively selects data subsets based on validation-driven utility feedback.  
Unlike standard curricula with fixed difficulty schedules, ONLINESUBMOD *learns what to learn next* using submodular optimization and online bandit updates.

---

#### 🎯 Utility-Based Subset Selection

At each training iteration $t$, the algorithm observes the model parameters $\theta_t$ and the available dataset $D_t$.  
It selects a subset $S_t \subseteq D_t$ of size at most $b$ that maximizes a composite **utility function**:

$$
S_t = \arg\max_{S \subseteq D_t,\; |S| \le b} f(S; \theta_t),
$$

where the overall utility $f(S; \theta_t)$ is defined as a **weighted combination** of multiple submodular criteria:

$$
f(S; \theta_t)
  = \lambda_1 f_{\text{rep}}(S)
  + \lambda_2 f_{\text{div}}(S)
  + \lambda_3 f_{\text{inf}}(S).
$$

Here:

- $f_{\text{rep}}(S)$ — **Representativeness**, measuring how well the subset covers the data distribution.  
- $f_{\text{div}}(S)$ — **Diversity**, encouraging varied examples to prevent redundancy.  
- $f_{\text{inf}}(S)$ — **Informativeness**, prioritizing uncertain or high-loss samples.  

The coefficients $\lambda_1, \lambda_2, \lambda_3$ are adaptive weights tuned by **validation feedback**, ensuring the model prioritizes subsets that yield real performance gains.

The utility satisfies the **submodular property** (diminishing returns):

$$
f(A \cup \{x\}) - f(A)
  \ge f(B \cup \{x\}) - f(B),
  \quad \text{for all } A \subseteq B.
$$

This allows efficient greedy maximization with a $(1 - 1/e)$ guarantee:

$$
f(S_{\text{greedy}}) \ge (1 - 1/e)\, f(S^*).
$$

---

#### 🧩 Validation Utility and Reward Function

After selecting $S_t$ and training on it, ONLINESUBMOD computes a **validation utility** $U_t$ to quantify improvement in model generalization:

$$
U_t = \mathbb{E}_{(x, y) \sim D_{\text{val}}}
      \big[ -\ell(x, y; \theta_t) \big],
$$

where $\ell(x, y; \theta_t)$ denotes the loss on the validation set.  
The **reward signal** $r_t$ is defined as the *improvement* in this validation utility between two consecutive steps:

$$
r_t = \Delta U_t
    = U_t - U_{t-1}
    = -\big( \ell_{\text{val}}(\theta_t)
             - \ell_{\text{val}}(\theta_{t-1}) \big).
$$

Equivalently, the same reward can be expressed in terms of validation loss reduction:

$$
r_t = \ell_{\text{val}}(\theta_{t-1}) - \ell_{\text{val}}(\theta_t).
$$

This reward is directly used to update the *bandit policy*, encouraging selection of utilities that yield the largest validation improvement.

---

#### 🧠 Bandit Objective

The utility functions $\{ f_1, f_2, \dots, f_K \}$ act as **arms** in a multi-armed bandit (MAB) framework.  
At each iteration $t$, one function $f_{a_t}$ is chosen to construct $S_t$.  
The bandit aims to maximize the cumulative expected reward:

$$
\max_{\pi} \;
\mathbb{E}\!\left[
  \sum_{t=1}^{T} r_t
\right],
$$

where $\pi$ is the arm-selection policy and $r_t$ is the observed validation gain.  
The expected reward for arm $k$ is:

$$
\mathbb{E}[r_t \mid a_t = k] = g(f_k),
$$

which links each submodular function’s subset quality to validation improvement.

---

#### 🧮 Submodular Maximization Step

For a selected arm $k$, ONLINESUBMOD solves:

$$
S_t = \arg\max_{S \subseteq D_t,\; |S| \le b} f_k(S),
$$

where $f_k(\cdot)$ satisfies submodularity and is efficiently optimized via a greedy algorithm.  
The result is a near-optimal subset used for model updates and reward computation.

---

#### ⚖️ No-Regret Bandit Update via EXP3

The policy weights are updated online using the **EXP3** algorithm (Exponential Weights for Exploration and Exploitation).  
Let $w_{k,t}$ denote the weight for arm $k$ at time $t$, and $p_{k,t}$ its sampling probability:

$$
p_{k,t} = \frac{w_{k,t}}{\sum_j w_{j,t}}.
$$

After observing the reward $r_t$, the weight update rule is:

$$
w_{k,t+1}
  = w_{k,t}
    \exp\!\big(\eta\, \hat{r}_{k,t}\big),
$$

where  

$$
\hat{r}_{k,t}
  = \frac{r_t \, \mathbb{I}[a_t = k]}{p_{k,t}}.
$$

Here, $\eta$ is the learning rate, and $\mathbb{I}[a_t = k]$ is an indicator for the selected arm.  
This update balances **exploration** (trying diverse submodular functions) and **exploitation** (favoring those that yield higher validation reward).

Theoretical analysis guarantees **no regret** compared to the best fixed submodular function:

$$
\text{Regret}(T)
  = \mathcal{O}\!\big(\sqrt{T K \log K}\big).
$$

Thus, over time, ONLINESUBMOD converges to the most beneficial utility definition for the task, ensuring efficient and adaptive data selection.

---

#### 🧩 Summary of Core Mechanisms

- **Utility Function:** Combines representativeness, diversity, and informativeness with adaptive validation-driven weighting.  
- **Validation Reward:** Computed as stepwise improvement in validation performance.  
- **Bandit Formulation:** Learns which utility yields highest expected gain via EXP3 updates.  
- **No-Regret Policy:** Ensures convergence to optimal submodular criterion with bounded regret.  
- **Efficiency:** Submodular maximization adds negligible computational cost (<1 ms overhead).

