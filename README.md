# 🎯 Improved Hybrid Hangman Agent

This repository contains an advanced machine learning agent designed to play the classic game of Hangman. The agent employs a **hybrid strategy**, combining a Deep Q-Network (DQN) Reinforcement Learning (RL) component with two sophisticated statistical models: an **Enhanced Trigram Hidden Markov Model (HMM)** and an **Enhanced Frequency Model**. [cite_start]This hybrid approach leverages the strength of linguistic statistics for accurate letter probability estimation and the adaptive learning capability of Deep RL for strategic guessing.

---

## 🚀 Key Features

* **Hybrid Architecture:** The core guessing logic combines outputs from three distinct models:
    * [cite_start]**Deep Q-Network (DQN):** A neural network trained via Reinforcement Learning to learn strategic guessing based on the game state.
    * [cite_start]**Enhanced Trigram HMM:** A statistical model focused on letter probabilities derived from character-level unigrams, bigrams, and trigrams within the training corpus.
    * [cite_start]**Enhanced Frequency Model:** A statistical model that tracks global, position-specific, and word-length-specific letter frequencies, incorporating **vowel/consonant boosting** heuristics.
* [cite_start]**Weighted Decision Making:** The models' outputs are combined using adjustable weights: **HMM (50%), Frequency (35%), and RL (15%)**, favoring the stable statistical models.
* [cite_start]**Intelligent Exploration:** During the RL training phase, the agent uses an **epsilon-greedy strategy** where exploration steps are guided by the combined probability distribution of the statistical models, leading to more informed exploration.
* [cite_start]**Detailed State Representation:** The game state is vectorized into a **35-dimensional feature tensor** for the DQN.
* [cite_start]**Improved Reward Shaping:** The reward function heavily penalizes **repeated guesses (-5.0)** and losses (**-20.0**) while providing boosted rewards for **correct guesses** based on remaining lives and a large bonus for a **win (+50.0)**.

---

## ⚙️ Model Details (DQN)

The Reinforcement Learning component uses a simplified DQN structure implemented in PyTorch:

| Layer | Input Size | Output Size | Activation | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `Linear` | 35 | 128 | `ReLU` | [cite_start]State vector to hidden layer. |
| `Dropout` | - | - | - | [cite_start]Dropout rate of **0.1** to prevent overfitting. |
| `Linear` | 128 | 128 | `ReLU` | [cite_start]Second hidden layer. |
| `Linear` | 128 | 26 | - | [cite_start]Output Q-values for all 26 possible actions (letters). |

### Key Hyperparameters

| Parameter | Value |
| :--- | :--- |
| `State Dimension` | [cite_start]35  |
| `Action Dimension` | [cite_start]26  |
| `Learning Rate (lr)` | [cite_start]0.0005  |
| `Discount Factor (gamma)` | [cite_start]0.99  |
| `Epsilon Decay` | [cite_start]0.9997  |
| `Batch Size` | [cite_start]64  |
| `Replay Capacity` | [cite_start]30,000  |

---

## 📊 Final Test Results

[cite_start]The agent was tested on **2,000 games** using the final saved model (`best_agent.pkl`).

| Metric | Value |
| :--- | :--- |
| Games Played | [cite_start]2000  |
| Wins | [cite_start]500  |
| **Success Rate** | [cite_start]**0.2500 (25.00%)**  |
| Total Wrong Guesses | [cite_start]10841  |
| Total Repeated Guesses | [cite_start]0  |
| Average Wrong per Game | [cite_start]5.42  |
| Average Repeated per Game | [cite_start]0.00  |
| **FINAL SCORE** | [cite_start]**-53705.00**  |

### Final Score Calculation

The final test score is calculated as:
$$
\text{Final Score} = (\text{Wins} \times 2000) - (\text{Total Wrong Guesses} \times 5) - (\text{Total Repeated Guesses} \times 2)
$$

---

## 🛠️ Requirements and Setup

To run this agent, you would typically need:

* Python 3.x
* `numpy`
* `torch` (PyTorch)
* `tqdm` (for progress bars during training)

[cite_start]The agent is configured to use a **GPU (CUDA)** if available, falling back to **CPU** otherwise.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
