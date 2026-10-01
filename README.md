# 🎮 Reinforcement Learning: Frozen Lake Optimal Policy

A Reinforcement Learning project that implements **Value Iteration** to calculate the optimal value function and policy for the **Frozen Lake environment** using Gymnasium.

The project demonstrates important Reinforcement Learning concepts such as the **Markov Decision Process (MDP), Bellman Optimality Equation, discount factor (γ), Value Iteration, optimal policy, and environment simulation**.

---

## 📌 Project Overview

Frozen Lake is a classic Reinforcement Learning problem where an agent must navigate through a grid and reach the goal while avoiding holes.

The environment contains different types of states:

* 🟦 **Start** — Initial position of the agent
* 🧊 **Frozen Lake** — Safe states
* 🕳️ **Hole** — Terminal failure state
* 🏁 **Goal** — Terminal success state

The objective is to determine the best sequence of actions that allows the agent to reach the goal while maximizing its expected reward.

---

## 🧠 Algorithm: Value Iteration

This project uses **Value Iteration** to find the optimal policy.

The value function is updated using the **Bellman Optimality Equation**:

```text
V(s) = maxₐ Σ P(s'|s,a) [R(s,a,s') + γV(s')]
```

Where:

* `V(s)` = Value of the current state
* `a` = Action
* `s'` = Next state
* `P(s'|s,a)` = Transition probability
* `R` = Reward
* `γ` = Discount factor

The algorithm repeatedly updates the value of every state until the values converge.

---

## 🔄 Working Process

```text
Frozen Lake Environment
          ↓
    Initialize V(s)
          ↓
   Select Each State
          ↓
   Evaluate All Actions
          ↓
 Bellman Optimality Update
          ↓
   Calculate Maximum Value
          ↓
 Check Convergence (Δ)
          ↓
   Optimal Value Function
          ↓
     Extract Policy
          ↓
    Run Environment
          ↓
      Reach the Goal
```

---

## 🎯 Actions

The Frozen Lake environment provides four possible actions:

```text
0 → Left
1 → Down
2 → Right
3 → Up
```

The optimal policy selects the action with the highest expected future value for each state.

---

## ⚙️ Discount Factor (γ)

The discount factor determines how much importance the algorithm gives to future rewards.

Example:

```text
γ = 0.90
```

A higher value of `γ` gives more importance to future rewards, while a lower value focuses more on immediate rewards.

The project can experiment with different discount factors such as:

```text
0.90
0.95
0.99
```

---

## 📊 Convergence

During Value Iteration, the maximum change between the old and new value functions is tracked using:

```text
Δ = max |V_new(s) - V_old(s)|
```

The algorithm stops when:

```text
Δ < θ
```

where `θ` is the chosen convergence threshold.

This ensures that the value function has sufficiently stabilized before extracting the optimal policy.

---

## 🕹️ Simulation

After calculating the optimal policy, the agent can interact with the Frozen Lake environment.

The simulation shows:

```text
Current State
      ↓
Optimal Action
      ↓
Next State
      ↓
Reward
      ↓
Continue Until
Goal / Hole
```

The project also provides an interactive **Streamlit interface** for running the simulation.

---

## 🛠️ Technologies Used

* **Python**
* **Gymnasium**
* **NumPy**
* **OpenCV**
* **Streamlit**
* **Reinforcement Learning**
* **Value Iteration**

---

## 📂 Project Structure

```text
reinforcement-learning-frozen-lake/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/Karamthot-anil-naik/reinforcement-learning-frozen-lake.git
```

Navigate to the project:

```bash
cd reinforcement-learning-frozen-lake
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application opens in your browser and allows you to run the Frozen Lake simulation.

---

## 📚 Key Concepts Demonstrated

This project demonstrates practical understanding of:

* Reinforcement Learning
* Markov Decision Processes
* States and Actions
* Rewards
* Transition Probabilities
* Bellman Equation
* Bellman Optimality
* Value Function
* Discount Factor
* Value Iteration
* Policy Extraction
* Optimal Policy
* Convergence
* Environment Simulation

---

## 🎯 Learning Outcome

Through this project, I gained practical understanding of how **Value Iteration** can be used to solve a Markov Decision Process and determine an optimal policy.

The project connects the theoretical concepts of **Bellman equations and dynamic programming** with an interactive Reinforcement Learning environment.

--

