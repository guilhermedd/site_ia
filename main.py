from src.ActorCritic import ActorCritic
from src.TicTacToeEnv import TicTacToeEnv
from src.Agent import Agent
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import custom_object_scope
import time


def run_episode(env, agent):
    state = env.reset()
    total_reward = 0
    total_loss = 0
    done = False
    player = 1  # X começa

    while not done:
        action = agent.select_action(state, env.get_legal_moves())
        next_state, reward, done = env.step(action, player)
        loss = agent.train_step(state, action, reward, next_state, done)

        total_reward += reward
        total_loss += loss.numpy()
        state = next_state
        player *= -1  # Alterna o jogador

    return total_loss, total_reward


def train_agents(env, agent, num_episodes):
    rewards = []
    losses = []
    elapsed_episodes = []

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    ax1.set_title('Recompensas por Episódio')
    ax1.set_xlabel('Episódio')
    ax1.set_ylabel('Recompensa')
    ax2.set_title('Perdas por Episódio')
    ax2.set_xlabel('Episódio')
    ax2.set_ylabel('Perda')

    reward_line, = ax1.plot([], [], label="Reward", color='blue')
    loss_line, = ax2.plot([], [], label="Loss", color='blue')

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper left")

    for episode in range(num_episodes):
        elapsed_episodes.append(episode + 1)
        loss, reward = run_episode(env, agent)
        rewards.append(reward)
        losses.append(loss)

        reward_line.set_data(elapsed_episodes, rewards)
        loss_line.set_data(elapsed_episodes, losses)

        ax1.set_xlim(1, max(1, episode + 1))
        ax1.set_ylim(min(rewards) - 5, max(rewards) + 5)
        ax2.set_xlim(1, max(1, episode + 1))
        ax2.set_ylim(min(losses) - 5, max(losses) + 5)

        plt.draw()
        plt.pause(0.01)
        time.sleep(0.1)

    plt.ioff()
    plt.show()


def load_actor_critic_model(filepath):
    try:
        print("Loading model...")
        return  tf.keras.models.load_model(filepath, custom_objects={"ActorCritic": ActorCritic})
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None


def save_actor_critic_model(agent, filepath):
    actor_critic.save(filepath)


if __name__ == "__main__":
    env = TicTacToeEnv(player=1)
    file_path = "actor_critic_model.keras"

    actor_model = load_actor_critic_model(file_path)
    gamma = 0.99
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    if actor_model is None:
        # Configurações do modelo
        num_actions = env.action_space
        num_hidden_units = 128

        actor_critic = ActorCritic(num_actions, num_hidden_units)
        agent = Agent(actor_critic=actor_critic, optimizer=optimizer, gamma=gamma, player=1)

        num_episodes = 3
        train_agents(env, agent, num_episodes)

        save_actor_critic_model(agent, file_path)
    else:
        print("Modelo carregado com sucesso!")
        agent = Agent(actor_model, optimizer=optimizer, gamma=gamma, player=-1)

        player = 1
        state = env.reset()
        done = False
        env.render()
        while not done:
            if player == 1:
                move = int(input("Escolha uma posição: "))
            else:
                move = agent.select_action(state, env.get_legal_moves())
            state, reward, done = env.step(move, player)
            player *= -1