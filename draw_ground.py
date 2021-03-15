import numpy as np
import matplotlib.pyplot as plt
import time

# %%
plt.ion()
x = np.linspace(0, 100, 400)
y1 = x * 3 - 300
y2 = x * (-3) + 200
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
ax1.plot(x, y1, label='legend')
ax1.plot(x, y2, label='haha')
ax1.legend()
ax1.set_xlabel('awef', fontsize=14)
ax1.set_ylabel('awefd', fontsize=14)
ax2.plot(x, y2)
ax1.set_title('This is a graph', fontsize=18)
ax2.set_title('This is another one')

s = input()
# %%
plt.close(fig)

