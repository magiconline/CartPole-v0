import gym
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from collections import deque
import os

from keras.layers import Dense
from keras.models import Sequential, load_model

SHOW = True  # control env.render()
Train = False  # control model.fit, EPSILON
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # dont have to use gpu because the limit should be env.step() not model.fit()


EPISODES = 3000
DISCOUNT = 0.95
EPSILON = 1.0 if Train else 0.0  # explore or use
EPSILON_DACAY = 0.99
MODEL_PATH = 'model-1587539585'  # None or 'filename'
BATCH_SIZE = 64
EPOCH = 5
MAX_REPLAY_MEMORY = BATCH_SIZE * 5


def create_model(input_dim, output_dim):
    model = Sequential([
        Dense(24, activation='relu', input_dim=input_dim),
        Dense(24, activation='relu'),
        Dense(output_dim=output_dim)
    ])
    model.compile(loss='mse', optimizer='adam')
    return model


env = gym.make('CartPole-v0')

observation_n = env.observation_space.shape[0]
activation_n = env.action_space.n

if MODEL_PATH:
    model = load_model(MODEL_PATH)
else:
    model = create_model(observation_n, activation_n)

avg_survive_time = []
avg_reward = []
replay_memory = deque(maxlen=MAX_REPLAY_MEMORY)

for episode in range(EPISODES):
    survive_time = 0
    rewards = 0
    done = False
    observation = env.reset()
    observation = observation.reshape((-1, observation_n))

    while not done:
        if SHOW:
            env.render()

        observation = observation.reshape((-1, observation_n))
        q = model.predict(observation)

        if random.random() > EPSILON:
            action = np.argmax(q)
        else:
            action = env.action_space.sample()

        new_observation, reward, done, _ = env.step(action)
        rewards += reward
        new_observation = new_observation.reshape((-1, observation_n))
        replay_memory.append((observation, action, reward, done, new_observation))
        survive_time += 1
        observation = new_observation

    observation, action, reward, done, new_observation = zip(*random.sample(replay_memory, min(BATCH_SIZE, len(replay_memory))))
    observation = np.array(observation).reshape((-1, 4))
    new_observation = np.array(new_observation).reshape(-1, 4)
    q = model.predict(observation)
    max_future_q = np.amax(model.predict(new_observation), axis=1)

    for index in range(len(observation)):
        q[index][action[index]] = reward[index] if done[index] else reward[index] + max_future_q[index]

    if Train:
        model.fit(observation, q, epochs=EPOCH, verbose=0)

    print(f'episode: {episode}, survive time: {survive_time}, reward: {rewards}')
    avg_survive_time.append(survive_time)
    avg_reward.append(rewards)
    EPSILON *= EPSILON_DACAY


plt.plot(avg_survive_time)
plt.plot(avg_reward)
plt.legend(['survive time', 'reward'], loc='upper left')
plt.show()

model.save(f'model-{int(time.time())}')

env.close()
