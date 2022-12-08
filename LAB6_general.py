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

        # figure layout
        ax.set_xlabel("time ($t$)")
        ax.set_ylabel("x")
        ax.legend()
        ax.set_title(
            f"{subplot_labels[i][j]}) Function {j + 1} using $\Delta t$ = {timestep}."
        )

# figure layout and show
plt.tight_layout()
plt.savefig("Exercise3.png")
plt.show()
plt.close()


""" Exercise 4 """

# functions, analytical and numerical
def func4(t, x, g, k):
    return g - k * x


def analytic4(t, start_value, g, k):
    return g / k - (g / k - start_value) * np.exp(-k * t)


# 4e and 4f

# various growth rates and degrade rates
growth_rates = [2, 1, 2, 1]
degrade_rates = [3, 1.5, 2, 1]

# stepsize used in simulations
stepsize = 0.01

# set up first figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[9, 5])

# colors for the different lines
colors = ["b", "r", "g", "orange"]

# initialize ODE
ode = numericalFO1D(func4, analytic4)

for i, growth_rate in enumerate(growth_rates):

    # run simulation with given growth and degrade
    g = growth_rates[i]
    k = degrade_rates[i]
    numerical = ode.runEuler(stepsize, 0, 5, start_value=0, fargs=[g, k])
    analytical = ode.runAnalytical(stepsize, 0, 5, start_value=0, fargs=[g, k])

    # plot the found solutions
    ode.plotAnalytical(ax1, color=colors[i], label=f"g = {g}, k = {k}")
    ode.plotNumerical(ax2, color=colors[i], label=f"g = {g}, k = {k}")

# figure layout
ax1.set_xlabel("time ($t$)")
ax1.set_ylabel("x")
ax2.set_xlabel("time ($t$)")
ax2.set_ylabel("x")

ax1.legend()
ax2.legend()

ax1.set_title("a) Analytic solutions")
ax2.set_title(f"b) Numerical solutions using $\Delta t$ = {stepsize}")

plt.savefig("Exercise4_2.png")
plt.show()
plt.close()


# 4h

# set up new figure
fig, ax = plt.subplots(1, 1, figsize=[6, 6])

for i, growth_rate in enumerate(growth_rates):

    # run simulation with given growth and degrade
    g = growth_rates[i]
    k = degrade_rates[i]
    analytical = ode.runAnalytical(stepsize, 0, 5, start_value=0, fargs=[g, k])

    # plot dx/dt for every tenth x(t) point
    x_list = analytical[1][::10]
    dx_list = func4(0, x_list, g, k)
    ax.plot(x_list, dx_list, "o-", ms=3, linewidth=1, label=f"g = {g}, k = {k}")

# figure layout
plt.xlabel("x(t)")
plt.ylabel("dx/dt")
plt.legend()
plt.savefig("Exercise4_2.png")
plt.show()
plt.close()


# 4i

# define variable function for g(t)
def g(t, g0, t1, multiplication):

    t = np.array(t)
    t1 = t1 * np.ones_like(t)

    return np.where(t < t1, g0, multiplication * g0)


# define function for dx/dt with varying g(t)
def func5(t, x, g0, t1, multiplication, k):
    return g(t, g0, t1, multiplication) - k * x


# initialize ODE
ode = numericalFO1D(func5)

# parameter values
g0 = 1
t1 = 5
multiplication = 2
k = 3

# generate g(t) data points
t = np.arange(0, 10, stepsize)
g_t = g(t, g0, t1, multiplication)

# run numerical simulation
numerical = ode.runEuler(
    stepsize, 0, 10, start_value=0, fargs=[g0, t1, multiplication, k]
)

# set up a new figure and plot g(t) and x(t)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[8, 5])
ax1.plot(t, g_t)
ode.plotNumerical(ax2)

# figure layout
ax1.set_ylim(0, 3)
ax1.set_xlabel("$t$")
ax1.set_ylabel("$g(t)$")
ax1.set_title("a) Generation rate versus time")

ax2.set_xlabel("$t$")
ax2.set_ylabel("$x(t)$")
ax2.set_title("b) Concentration versus time")

plt.savefig("Exercise4_3.png")
plt.show()
plt.close()


""" Exercise 5 """

# 5c and 5d

# define functions
def func6(t, x, r, k):
    x = np.array(x)
    zeros = np.zeros_like(x)
    return np.where(x > zeros, (r - k) * x, 0)


def analytical6(t, start_value, r, k):
    return start_value * np.exp((r - k) * t)


# initialize ODE
ode = numericalFO1D(func6, analytical6)

# define simulation parameters
r_values = [0.5, 0.5, 0.5]
k_values = [0.1, 0.5, 0.9]
stepsize = 0.01

# set up a new figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[8, 5])

for i, r in enumerate(r_values):

    k = k_values[i]

    # simulate data and plot
    analytical = ode.runAnalytical(stepsize, 0, 6, 100, [r, k])
    numerical = ode.runEuler(stepsize, 0, 6, 100, [r, k])
    ode.plotAnalytical(ax1, "C" + str(i), label=f"r = {r}, k = {k}")
    ode.plotNumerical(ax2, "C" + str(i), label=f"r = {r}, k = {k}")


# figure layout
ax1.set_xlabel("$t$")
ax1.set_ylabel("$x(t)$")

ax2.set_xlabel("$t$")
ax2.set_ylabel("$x(t)$")

ax1.set_title("a) Analytical solutions")
ax2.set_title("b) Numerical solutions")

ax1.grid()
ax2.grid()
ax1.legend()
ax2.legend()

plt.tight_layout()

plt.savefig("Exercise5_1.png")
plt.show()
plt.close()


# 5h

# define functions
def func7(t, x, r, k):
    x = np.array(x)
    zeros = np.zeros_like(x)
    nans = np.nan * np.ones_like(x)
    overflow = 1e3 * np.ones_like(x)  # so many rabbits seems sufficient

    derivative = np.where(
        x != nans and x > zeros and x < overflow, (r * x ** 2 - k * x), nans
    )

    return derivative


# initialize ODE
ode = numericalFO1D(func7)

# simulation parameters
r = 1
k = 1
starts = [0.5, 1.0, 1.5]

# set up figure
fig, ax = plt.subplots(1, 1, figsize=[5, 5])

for i, start_value in enumerate(starts):
    # simulate data and plot
    numerical = ode.runEuler(stepsize, 0, 3, start_value, [r, k])
    ode.plotNumerical(ax, "C" + str(i), label=f"$x_0$ = {start_value}")

# figure layout
ax.set_yscale("log")
ax.set_ylim(1e-2, 1e3)
ax.set_ylabel("$\log{(x(t)}$")
ax.set_xlabel("$t$")
ax.legend()

plt.tight_layout()
plt.savefig("Exercise5_2.png")
plt.show()
plt.close()


# 5i

# logistic equation
def func8(t, x, x_max):
    return x * (1 - x / x_max)


# initialize ODE
ode = numericalFO1D(func8)

# parameter values
starts = [0.8, 0.8, 1.6, 1.6]
x_max = [1, 2, 1, 2]

# set up figure
fig, ax = plt.subplots(1, 1, figsize=[5, 5])

for i, start_value in enumerate(starts):

    # run numerical simulation for the different parameters
    numerical = ode.runEuler(stepsize, 0, 10, start_value, [x_max[i]])
    ode.plotNumerical(
        ax,
        "C" + str(i),
        label=f"$x_0$ = {start_value}, " + "$x_{max}$ = " + str(x_max[i]),
    )

# figure layout
ax.set_xlabel("$t$")
ax.set_ylabel("$x(t)$")
ax.legend()

plt.savefig("Exercise5_3.png")
plt.show()
plt.close()


# 5l

# set up new figure
fig, ax = plt.subplots(1, 1, figsize=[6, 6])

# plot dx/dt for every x in a given range
xmax = 2
x_list = np.linspace(-1, xmax + 1)
dx_list = func8(0, x_list, xmax)
ax.plot(x_list, dx_list, "-", ms=3, linewidth=1, label="$x_{max} = $" + str(xmax))
ax.hlines(0, -1, xmax + 1, "black", linewidth=1)

# figure layout
plt.xlabel("x(t)")
plt.ylabel("dx/dt")
plt.xlim(-1, xmax + 1)
plt.legend()
plt.grid()

plt.savefig("Exercise5_4.png")
plt.show()
plt.close()


# 5m

# logistic equation
def func9(t, x, x_max, r):
    return x * (1 - r - x / x_max)


# initialize ODE
ode = numericalFO1D(func9)

# parameter values
starts = [0.5, 1.6, 0.5, 1.6]
x_max = 1
r_values = [0.25, 0.25, 0.75, 0.75]

# set up figure
fig, ax = plt.subplots(1, 1, figsize=[5, 5])

for i, start_value in enumerate(starts):

    # run numerical simulation for the different parameters
    numerical = ode.runEuler(stepsize, 0, 20, start_value, [x_max, r_values[i]])
    ode.plotNumerical(
        ax,
        "C" + str(i),
        label=f"$x_0$ = {start_value}, r = {r_values[i]}",
    )

# plot x_max
ax.hlines(x_max, 0, 20, label="$x_{max}$")

# figure layout
ax.set_xlabel("$t$")
ax.set_ylabel("$x(t)$")
ax.set_ylim(0, 2)
ax.set_xlim(0, 20)
ax.legend()
plt.grid()
plt.tight_layout()

plt.savefig("Exercise5_5.png")
plt.show()
plt.close()


# 5o

# parameter values
stepsizes = [0.01, 0.5, 1, 3]

# set up figure
fig, ax = plt.subplots(1, 1, figsize=[5, 5])

for i, stepsize in enumerate(stepsizes):

    # run numerical simulation for the different parameters
    numerical = ode.runEuler(stepsize, 0, 20.001, 1, [1, 2])
    ode.plotNumerical(
        ax,
        "C" + str(i),
        label=f"$\Delta t$ = {stepsize}",
    )

# figure layout
ax.set_xlabel("$t$")
ax.set_ylabel("$x(t)$")
ax.set_xlim(0, 5)
ax.set_ylim(-10, 1.5)
ax.legend()
plt.grid()
plt.tight_layout()

plt.savefig("Exercise5_6.png")
plt.show()
plt.close()
