import numpy as np
import matplotlib.pyplot as plt


class numericalFO1D:
    def __init__(self, derivative_function, analytical_function=None):
        self.function = derivative_function
        self.analytical = analytical_function

        self.times = []
        self.values = []
        self.analytic_values = []

    def runEuler(self, timestep=0.1, tmin=0.0, tmax=1.0, start_value=0.0, fargs=[]):
        self.times = np.arange(tmin, tmax, timestep)

        value = start_value

        for i, t in enumerate(self.times):
            self.values.append(value)
            value += timestep * self.function(t, value, *fargs)

        return self.times, self.values

    def runAnalytical(
        self, timestep=0.1, tmin=0.0, tmax=1.0, start_value=0.0, fargs=[]
    ):
        if self.analytical != None:

            self.times = np.arange(tmin, tmax, timestep)
            self.analytic_values = self.analytical(self.times, start_value, *fargs)

            return self.times, self.analytic_values

    def plotNumerical(self, ax, color="C0", label="numerical", ms=1):
        ax.plot(self.times, self.values, color=color, label=label, ms=1)
        return ax

    def plotAnalytical(self, ax, color="C1", label="analytical"):
        if self.analytical != None:
            ax.plot(
                self.times,
                self.analytic_values,
                color=color,
                label=label,
                ms=1,
            )
        return ax


def func3(t, x, fargs=None):
    return 2 * t


def analytical3(t, start_value, fargs=None):
    return start_value + t ** 2


if __name__ == "__main__":
    ode = numericalFO1D(func3, analytical3)
    sim = ode.runEuler(timestep=0.1, tmin=0, tmax=3, start_value=4, fargs=[])
    analytical = ode.runAnalytical(
        timestep=0.1, tmin=0, tmax=3, start_value=4, fargs=[]
    )

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ode.plotNumerical(ax)
    ode.plotAnalytical(ax)
    plt.legend()
    plt.show()
