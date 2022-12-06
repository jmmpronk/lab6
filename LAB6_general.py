#
# Runs all code for entire assignment of LAB6 and saves plots.
#

import numpy as np
import matplotlib.pyplot as plt

from modules import numericalFO1D


""" Exercise 3 """

# define the functions


def func1(t, x):
    return 1


def analytic1(t, start_value):
    return start_value + t


def func2(t, x):
    return 2 * t


def analytic2(t, start_value):
    return start_value + t ** 2


def func3(t, x):
    return -x


def analytic3(t, start_value):
    return start_value * np.exp(-t)


# create list of functions and analytic solutions
function_list = [func1, func2, func3]
analytic_list = [analytic1, analytic2, analytic3]

# indicate initial value for each function
start_value = [0, -4, 4]

# different values of the time steps to use
t_steps = [1, 0.1, 0.01]

# indicate labels for subplots
subplot_labels = [["a", "b", "c"], ["d", "e", "f"], ["g", "h", "i"]]

# create figure for the exercise
fig, axes = plt.subplots(len(t_steps), len(function_list), figsize=[10, 12])

for i, timestep in enumerate(t_steps):
    for j, function in enumerate(function_list):

        # indexing the subplot to use
        ax = axes[i, j]

        # initialize the ODE and run integration and analytic solution
        ode = numericalFO1D(function, analytic_list[j])
        numerical = ode.runEuler(
            timestep=timestep, tmin=0, tmax=3.001, start_value=start_value[j]
        )
        analytical = ode.runAnalytical(
            timestep=timestep, tmin=0, tmax=3.001, start_value=start_value[j]
        )

        # plot both solution
        ode.plotNumerical(ax)
        ode.plotAnalytical(ax)

        ax.set_xlabel("time ($t$)")
        ax.set_ylabel("x")
        ax.legend()

        ax.set_title(
            f"{subplot_labels[i][j]}) Function {j + 1} using $\Delta t$ = {timestep}."
        )

# figure layout and show
plt.tight_layout()

# plt.suptitle("Numerical and analytical solutions for three differential equations.")
plt.savefig("Exercise3.png")
plt.show()


""" Exercise 4 """

# functions, analytical and numerical
def func4(t, x, g, k):
    return g - k * x


def analytic4(t, start_value, g, k):
    return g / k - (g / k - start_value) * np.exp(-k * t)


# 4e and 4f
growth_rates = [2, 1, 2, 1]
degrade_rates = [3, 1.5, 2, 1]

stepsize = 0.01

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[9, 5])
colors = ["b", "r", "g", "orange"]

for i, growth_rate in enumerate(growth_rates):
    ode = numericalFO1D(func4, analytic4)
    g = growth_rates[i]
    k = degrade_rates[i]
    numerical = ode.runEuler(stepsize, 0, 5, start_value=0, fargs=[g, k])
    analytical = ode.runAnalytical(stepsize, 0, 5, start_value=0, fargs=[g, k])
    ode.plotAnalytical(ax1, color=colors[i], label=f"g = {g}, k = {k}")
    ode.plotNumerical(ax2, color=colors[i], label=f"g = {g}, k = {k}")

ax1.set_xlabel("time ($t$)")
ax1.set_ylabel("x")
ax2.set_xlabel("time ($t$)")
ax2.set_ylabel("x")

ax1.legend()
ax2.legend()

ax1.set_title("a) Analytic solutions")
ax2.set_title(f"b) Numerical solutions using $\Delta t$ = {stepsize}")

plt.savefig("Exercise4_1.png")
plt.show()
