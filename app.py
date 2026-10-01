import streamlit as st
import numpy as np
import pandas as pd
import gymnasium as gym
import time

ACTION_NAMES = {
    0: "Left",
    1: "Down",
    2: "Right",
    3: "Up",
}

def make_env(slippery: bool):
    return gym.make("FrozenLake-v1", is_slippery=slippery, render_mode="rgb_array")

def state_to_grid(env, state):
    desc = env.unwrapped.desc
    rows, cols = desc.shape
    board = []
    for r in range(rows):
        row = []
        for c in range(cols):
            cell = desc[r, c].decode("utf-8")
            idx = r * cols + c
            row.append("●" if idx == state else cell)
        board.append(row)
    return board

def get_prob_table(env, state):
    rows = []
    transitions = env.unwrapped.P[state]
    if hasattr(transitions, "items"):
        action_outcomes = transitions.items()
    else:
        action_outcomes = enumerate(transitions)

    for action, outcomes in action_outcomes:
        for outcome in outcomes:
            prob, next_state, reward, done = outcome
            rows.append({
                "Action": ACTION_NAMES.get(action, action),
                "Probability": round(float(prob), 4),
                "Next State": int(next_state),
                "Reward": float(reward),
                "Done": bool(done),
            })
    return pd.DataFrame(rows)

@st.cache_data
def solve_value_iteration(slippery: bool, gamma: float = 0.9, max_iter: int = 1000, tol: float = 1e-12):
    env = make_env(slippery)
    n_states = env.observation_space.n
    V = np.zeros(n_states, dtype=float)

    for _ in range(max_iter):
        new_V = np.zeros_like(V)
        for s in range(n_states):
            q_values = []
            for a in range(env.action_space.n):
                value = 0.0
                for prob, next_state, reward, done in env.unwrapped.P[s][a]:
                    if done:
                        value += prob * reward
                    else:
                        value += prob * (reward + gamma * V[next_state])
                q_values.append(value)
            new_V[s] = max(q_values)

        if np.max(np.abs(new_V - V)) < tol:
            V = new_V
            break
        V = new_V

    policy = []
    for s in range(n_states):
        q_values = []
        for a in range(env.action_space.n):
            value = 0.0
            for prob, next_state, reward, done in env.unwrapped.P[s][a]:
                if done:
                    value += prob * reward
                else:
                    value += prob * (reward + gamma * V[next_state])
            q_values.append(value)
        policy.append(int(np.argmax(q_values)))

    return V, policy
def main():
    st.set_page_config(page_title="reinforcement-learning-frozen-lake", page_icon="🎯", layout="wide")
    st.title("reinforcement-learning-frozen-lake")
    st.caption("Interactive FrozenLake environment + value iteration demo")

    with st.sidebar:
        st.header("Control panel")

        slippery = st.checkbox("Enable slippery movement", value=True)

        if "env" not in st.session_state or st.session_state.get("env_slippery") != slippery:
            st.session_state.env = make_env(slippery)
            st.session_state.env_slippery = slippery
            st.session_state.state, st.session_state.info = st.session_state.env.reset()
            st.session_state.last_reward = 0.0
            st.session_state.last_done = False

        if st.button("Reset environment"):
            st.session_state.state, st.session_state.info = st.session_state.env.reset()
            st.session_state.last_reward = 0.0
            st.session_state.last_done = False

        action_name = st.selectbox(
            "Action to take",
            ["Left", "Down", "Right", "Up"],
            index=2,
            help="Manual action for one environment step."
        )
        action_index = {"Left": 0, "Down": 1, "Right": 2, "Up": 3}[action_name]

        if st.button("Step environment"):
            try:
                next_state, reward, terminated, truncated, info = st.session_state.env.step(action_index)
                st.session_state.state = next_state
                st.session_state.last_reward = reward
                st.session_state.last_done = terminated or truncated
                st.session_state.info = info
                if terminated or truncated:
                    st.toast("Episode ended.")
            except Exception as e:
                st.warning(f"Environment step failed: {e}")

        if st.button("Random action"):
            try:
                random_action = int(st.session_state.env.action_space.sample())
                next_state, reward, terminated, truncated, info = st.session_state.env.step(random_action)
                st.session_state.state = next_state
                st.session_state.last_reward = reward
                st.session_state.last_done = terminated or truncated
                st.session_state.info = info
                if terminated or truncated:
                    st.toast("Episode ended.")
            except Exception as e:
                st.warning(f"Random step failed: {e}")

    env = st.session_state.env
    current_state = st.session_state.state

    col1, col2 = st.columns([1.2, 1.8])

    with col1:
        st.subheader("Environment state")
        st.write(f"Current state: {current_state}")
        st.write(f"Observation space: {env.observation_space.n}")
        st.write(f"Action space: {env.action_space.n}")
        st.write(f"Last reward: {st.session_state.last_reward}")
        st.write(f"Episode ended: {st.session_state.last_done}")

        render_frame = env.render()
        st.image(render_frame, caption="Frozen Lake board", use_container_width=True)

    with col2:
        st.subheader("Transition model for current state")
        transition_df = get_prob_table(env, current_state)
        st.dataframe(transition_df, use_container_width=True)

        st.subheader("Board map")
        grid = state_to_grid(env, current_state)
        st.dataframe(pd.DataFrame(grid), hide_index=True, use_container_width=True)

    st.markdown("---")

    st.subheader("Value iteration solution")
    gamma = st.slider("Discount factor (gamma)", 0.1, 0.99, 0.9, step=0.01)
    policy_option = st.segmented_control(
        "Policy option",
        ["Value iteration", "Random"],
        default="Value iteration",
        key="policy_option",
    )
    values, optimal_policy = solve_value_iteration(slippery, gamma=gamma)
    if policy_option == "Random":
        policy = np.random.default_rng(0).integers(0, env.action_space.n, size=env.observation_space.n)
    else:
        policy = optimal_policy

    value_df = pd.DataFrame({
        "State": list(range(env.observation_space.n)),
        "Value": values,
        "Best action": [ACTION_NAMES.get(a, a) for a in policy],
    })
    st.dataframe(value_df, use_container_width=True)

    st.subheader("Selected policy")
    rows = int(np.sqrt(env.observation_space.n))
    cols = rows
    policy_grid = []
    for r in range(rows):
        row = []
        for c in range(cols):
            state = r * cols + c
            row.append(ACTION_NAMES.get(policy[state], policy[state]))
        policy_grid.append(row)

    st.dataframe(pd.DataFrame(policy_grid), hide_index=True, use_container_width=True)

    st.subheader("Policy simulation")
    frame_placeholder = st.empty()
    status_placeholder = st.empty()

    if st.button("Start simulation"):
        simulation_env = make_env(slippery)
        state, _ = simulation_env.reset()
        done = False
        step = 0
        reward = 0.0
        status_placeholder.write("Simulation started...")

        while not done:
            action = int(policy[state])
            next_state, reward, terminated, truncated, _ = simulation_env.step(action)
            frame_placeholder.image(simulation_env.render(), width=400)
            state = next_state
            step += 1
            done = terminated or truncated
            status_placeholder.write(
                f"Step: {step} | State: {state} | "
                f"Action: {ACTION_NAMES[action]} | Reward: {reward}"
            )
            if not done:
                time.sleep(1)

        if reward == 1:
            status_placeholder.success(f"Goal reached in {step} steps.")
        else:
            status_placeholder.error(f"Episode ended after {step} steps.")
        simulation_env.close()

    st.markdown("---")
    st.subheader("Notebook logic summary")
  

if __name__ == "__main__":
    main()