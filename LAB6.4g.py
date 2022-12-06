import matplotlib.pyplot as plt
import numpy as np

h = 0.01
timesteps = np.linspace(0, 10, int(10 / h))

x = 0
x_list = []
dx_list = []
for t in timesteps:
    k = 3
    g = 2
    x = -(g / k) * (np.exp(-k * t) - 1)
    dx = g - k * x
    x_list.append(x)
    dx_list.append(dx)

plt.plot(x_list, dx_list, label=f"g = {g}, k = {k}")
plt.xlim(0, max(x_list))
plt.xlabel("x(t)")
plt.ylabel("dx/dt")
plt.legend()
plt.savefig("unstable.png")
plt.show()
