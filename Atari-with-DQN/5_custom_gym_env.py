import gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import DQN

# Create the Atari environment
env = gym.make('CartPole-v1')

# Wrap the environment in a vectorized environment
vec_env = DummyVecEnv([lambda: env])

# Create the DQN model
model = DQN('MlpPolicy', vec_env, verbose=1)

# Train the model for a number of timesteps
model.learn(total_timesteps=100)

# Test the trained model
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = vec_env.step(action)
    if done:
        vec_env.reset()

# Close the environment
vec_env.close()
