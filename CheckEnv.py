from Reinforcement import LabyrinthEnv

env = LabyrinthEnv()
episodes = 50

for episode in range(episodes):
    done = False
    env.reset()
    while not done:
        random_action = env.action_space.sample()
        print("action", random_action)
        obs, reward, done, info = env.step(random_action)
        print("reward",reward)