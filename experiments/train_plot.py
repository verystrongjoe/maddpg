import numpy as np
from matplotlib import pyplot as plt
history_rewards = np.load('history_rewards.npy')

history_rewards.shape
plt.plot(history_rewards)
plt.show()

r = []
for i in range(len(history_rewards)//1000):
    x = history_rewards[(i * 1000):(i * 1000)+1000]
    r.append(np.mean(x))

r = np.array(r)
plt.plot(r)
plt.show()