import matplotlib.pyplot as plt
import numpy as np

h = 0.01
timesteps = np.linspace(0, 3, int(3 / h))


def func1(t, x):
    return 1


def func2(t, x):
    return 2 * t


def func3(t, x):
    return -x


function_list = [func1, func2, func3]
start_value = [0, -4, 4]

for i in range(len(function_list)):
    x = start_value[i]
    x_list = []
    for t in timesteps:
        x += h * function_list[i](t, x)
        x_list.append(x)

    plt.xlim(0, 3)
    plt.xlabel("time (t)")
    plt.ylabel("x")
    plt.plot(timesteps, x_list)
    plt.savefig(f"function{i}.png")
    plt.show()
