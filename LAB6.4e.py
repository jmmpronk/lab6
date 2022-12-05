import matplotlib.pyplot as plt
import numpy as np

h = 0.01
timesteps = np.linspace(0, 5, int(5 / h))


growth_rate = [2, 1, 2, 1]
degrade_rate = [3, 1.5, 2, 1]

for i in range(len(degrade_rate)):
    x = 0
    x_list = []
    for t in timesteps:
        k = degrade_rate[i]
        g = growth_rate[i]
        x = -(g / k) * (np.exp(-k * t) - 1)
        x_list.append(x)

    plt.plot(timesteps, x_list, label=f"g = {g}, k = {k}")
plt.xlim(0, 5)
plt.xlabel("time (t)")
plt.ylabel("x")
plt.legend()
plt.savefig("protein.png")
plt.show()
