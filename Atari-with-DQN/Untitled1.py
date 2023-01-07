import gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

env = gym.make('CartPole-v1')
model = DQN("MlpPolicy", env, device="cpu")
model.learn(total_timesteps = 1000)
#model = DQN.load("dqn_lunar")

# Test the agent
obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
env.close()

# from stable_baselines3 import PPO, DQN
# import gym
#
# env = gym.make("LunarLander-v2")
# ppo = DQN("MlpPolicy", env, device="mps")
#
# ppo.learn(total_timesteps = 1000)
# #Test the agent
# obs = env.reset()
# done = False
# while not done:
#     action, _states = ppo.predict(obs)
#     obs, rewards, done, info = env.step(action)
#     env.render()
# env.close()