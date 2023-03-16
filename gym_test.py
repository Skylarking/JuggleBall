import gym
import roller_gym

if __name__ == '__main__':
    env = gym.make('Roller-v0', env_file='Roller')
    obs = env.reset()
    done = False

    for i in range(100):
        print("Episode {} start!".format(i))
        while not done:
            # max steps:50000
            next_obs, reward, done, info = env.step([0.3, 0])
            print(reward)

        done = False
        env.reset()