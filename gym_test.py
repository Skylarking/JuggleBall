import gym
import unity_gym

if __name__ == '__main__':
    env = gym.make('Roller-v0', env_file='Roller')
    obs = env.reset()
    done = False

    for i in range(100):
        print("Episode {} start!".format(i))
        while not done:
            # max steps:50000
            action = env.action_space.sample()
            next_obs, reward, done, info = env.step(action)
            print(reward)

        done = False
        env.reset()