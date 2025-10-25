
### 🧩 Problem Statement

Traditional **curriculum learning** assumes a fixed notion of “easy-to-hard” sample progression, yet defining *difficulty* is often arbitrary and domain-dependent.  
Meanwhile, **adaptive subset selection** methods—though powerful—can be computationally heavy.  
We ask:  
> Can we design a *principled*, *efficient*, and *adaptive* curriculum that learns which samples to train on—guided directly by validation performance?

### ⚙️ Methodology in a Nutshell: ONLINESUBMOD

We propose **ONLINESUBMOD**, a *bandit-guided submodular curriculum* framework.  
At every training step, ONLINESUBMOD views each **submodular function** (e.g., diversity, representativeness) as a **bandit arm**, and dynamically selects which one to use based on real-time validation rewards.

#### Key Ideas:
- **Bandit Formulation:** Each submodular function acts as an arm; arm rewards are derived from validation loss improvement.
- **No-Regret Policy:** The selection strategy provably achieves *no-regret* over time, ensuring convergence to optimal subset-selection behavior.
- **Utility-Driven Curriculum:** The policy adapts using validation-aware reward signals—linking data selection directly to generalization performance.
- **Computationally Efficient:** Submodular maximization adds <1 ms overhead compared to gradient computation.

<p align="center">
<img src="/images/ss1.png" width="80%">
<br><em>Figure: Bandit-guided selection of submodular functions over training.</em>
</p>

---

### 📊 Results Overview

#### 1. **LLM Finetuning (LLaMA-2-7B, Mistral-7B on LESS → MMLU, TYDIQA)**
ONLINESUBMOD achieves the best accuracy across most domains while maintaining low perplexity throughout training.

| Method | Avg. | Soc. | Pol. | Hist. | ML | Eth. | Bio | Chem | TydiQA |
|:-------|:-----:|:----:|:----:|:----:|:--:|:----:|:----:|:----:|:-------:|
| GradNorm | 46.4 | 61.0 | 62.5 | 52.1 | 40.5 | 40.2 | 46.7 | 42.9 | 54.6 |
| GREATS | 47.8 | 63.2 | 66.2 | 48.3 | 42.6 | 41.1 | 48.9 | 43.1 | 55.7 |
| **ONLINESUBMOD** | **49.6** | **65.3** | **67.4** | **52.1** | **45.2** | **42.7** | **50.9** | **45.1** | **55.9** |

*ONLINESUBMOD yields ~3% absolute gain on average, with faster convergence.*

#### 2. **Vision Tasks (CIFAR-10/100, TinyImageNet, MNIST, SVHN)**

| Dataset | 10% Budget | 30% Budget | 50% Budget |
|:---------|:-----------|:-----------|:-----------|
| **CIFAR-100** | 0.736 / 9.2× | 0.754 / 3.3× | 0.758 / 1.9× |
| **TinyImageNet** | 0.553 / 8.4× | 0.607 / 3.1× | 0.626 / 2.6× |
| **CIFAR-10** | 0.924 / 5.4× | 0.937 / 2.0× | 0.941 / 2.0× |

(Values: *accuracy / relative speedup*)  
Even with only 30% of the training data, ONLINESUBMOD matches or exceeds full-data accuracy.

<p align="center">
<img src="/images/ss2.png" width="85%">
<br><em>Figure: ONLINESUBMOD achieves top-1 accuracy and near-optimal speedup across datasets.</em>
</p>

---

### 🧠 Why It Works

- The **validation-aware reward** tightly aligns the training signal with generalization.
- The **bandit-driven exploration** ensures that useful submodular functions (e.g., diversity vs representativeness) are chosen at the right stage.
- **No-regret guarantees** formalize the learning efficiency theoretically.

---

### 🚀 Summary

ONLINESUBMOD bridges *curriculum learning* and *adaptive data selection* under a unified theoretical framework.  
It efficiently learns *what to learn next*—offering a robust, scalable path for efficient model training, especially in large-scale **vision and language** settings.

> 📂 **Code:** [https://github.com/efficiency-learning/banditsubmod](https://github.com/efficiency-learning/banditsubmod)

---

*© 2025 Efficiency Learning Group, IIT Bombay. All rights reserved.*
