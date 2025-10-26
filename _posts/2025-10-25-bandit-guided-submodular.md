---
layout: post
title: "Bandit-Guided Submodular Curriculum for Adaptive Subset Selection"
math: true
categories: [Research, Curriculum Learning, Subset Selection]
---

### 🧩 Problem Statement

Traditional **curriculum learning** assumes a fixed notion of “easy-to-hard” sample progression, yet defining *difficulty* is often arbitrary and domain-dependent.  
Meanwhile, **adaptive subset selection** methods—though powerful—can be computationally heavy.

We ask:  
> Can we design a *principled*, *efficient*, and *adaptive* curriculum that learns which samples to train on—guided directly by validation performance?

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

#### 🧮 Utility-Based Subset Selection

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

#### 🧠 Validation Utility and Reward Function

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

#### 🎯 Bandit Objective

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

#### 📈 Submodular Maximization Step

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

#### 📊 Experimental Results

**1. LLM Finetuning (LLaMA-2-7B, Mistral-7B on LESS → MMLU, TYDIQA)**

| Method | Avg. | Soc. | Pol. | Hist. | ML | Eth. | Bio | Chem | TydiQA |
|:-------|:-----:|:----:|:----:|:----:|:--:|:----:|:----:|:----:|:-------:|
| GradNorm | 46.4 | 61.0 | 62.5 | 52.1 | 40.5 | 40.2 | 46.7 | 42.9 | 54.6 |
| GREATS | 47.8 | 63.2 | 66.2 | 48.3 | 42.6 | 41.1 | 48.9 | 43.1 | 55.7 |
| **ONLINESUBMOD** | **49.6** | **65.3** | **67.4** | **52.1** | **45.2** | **42.7** | **50.9** | **45.1** | **55.9** |

*ONLINESUBMOD yields ~3% absolute gain on average, with faster convergence.*

<p align="center">
  <img src="/assets/images/ss1.png" width="80%">
  <br>
  <em>Figure 1: Bandit-guided selection of submodular functions during training.</em>
</p>

---

**2. Vision Tasks (CIFAR-10/100, TinyImageNet, MNIST, SVHN)**

| Dataset | 10% Budget | 30% Budget | 50% Budget |
|:---------|:-----------|:-----------|:-----------|
| **CIFAR-100** | 0.736 / 9.2× | 0.754 / 3.3× | 0.758 / 1.9× |
| **TinyImageNet** | 0.553 / 8.4× | 0.607 / 3.1× | 0.626 / 2.6× |
| **CIFAR-10** | 0.924 / 5.4× | 0.937 / 2.0× | 0.941 / 2.0× |

(Values: *accuracy / relative speedup*)  
Even with only 30% of the training data, ONLINESUBMOD matches or exceeds full-data accuracy.

<p align="center">
  <img src="/assets/images/ss2.png" width="85%">
  <br>
  <em>Figure 2: ONLINESUBMOD achieves top-1 accuracy and near-optimal speedup across datasets.</em>
</p>

---

### 🧠 Why It Works

- The **validation-aware reward** ties learning signals directly to generalization.  
- **Bandit-driven exploration** dynamically balances diversity and representativeness.  
- **Submodularity** ensures principled and efficient subset selection.  
- **No-regret updates** guarantee asymptotic optimality.

Formally:

$$
\sum_{t=1}^{T} r_t \ge \sum_{t=1}^{T} r_t^* - \mathcal{O}(\sqrt{T K \log K}),
$$

ensuring the learner asymptotically approaches the best submodular policy.

---

### 🚀 Summary

ONLINESUBMOD bridges *curriculum learning* and *adaptive data selection* under a unified theoretical framework.  
It efficiently learns *what to learn next*—offering a robust, scalable path for efficient model training across **vision** and **language** domains.

> 📂 **Code:** [https://github.com/SALT-NLP/Efficient_Unlearning](https://github.com/SALT-NLP/Efficient_Unlearning)

---

*© 2025 Efficiency Learning Group, IIT Bombay. All rights reserved.*
