class Plot():
    def __init__(self, env_name):
        self.env_name = env_name

    def plot_success(self, eps, success, normalization):
    # def plot_reward(self, rewards):
        import numpy as np
        import matplotlib.pyplot as plt

        x = np.asarray(eps)
        y = np.asarray(success)

        plt.title(self.env_name)
        plt.xlabel('number of episode')
        plt.ylabel('reward')

        plt.plot(x, y, linewidth=2.0)
        plt.show()