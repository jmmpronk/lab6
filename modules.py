#
# Introduces the class of one dimensional first order ordinary differential equation.
#


import numpy as np
import matplotlib.pyplot as plt


class numericalFO1D:
    """Stores a one dimensional first order Ordinary Differential Equation with optionally an analytic solution."""

    def __init__(self, derivative_function, analytical_function=None):
        """Stores the functions given.

        Args:
            derivative_function (function): function for the time derivative of x(t) with arguments (t, x, *fargs)
            analytical_function (function, optional): function for the analytic solution x(t) with arguments (t, start_value, *fargs). Defaults to None.
        """
        self.function = derivative_function
        self.analytical = analytical_function

        self.times = []
        self.values = []
        self.analytic_values = []

    def runEuler(self, timestep=0.1, tmin=0.0, tmax=1.0, start_value=0.0, fargs=[]):
        """Runs an Euler integration of the derivative function as a numerical solution.

        Args:
            timestep (float, optional): time step used for integration. Defaults to 0.1.
            tmin (float, optional): starting point of t. Defaults to 0.0.
            tmax (float, optional): end point of t. When separated a whole number of timesteps from tmin, the integration ends before tmax. Defaults to 1.0.
            start_value (float, optional): starting value of the independent variable x. Defaults to 0.0.
            fargs (list, optional): optional extra function parameters. Defaults to [].

        Returns:
            tuple: (numpy array of time values, list of x values from numerical integration)
        """
        self.times = np.arange(tmin, tmax, timestep)
        self.values = []

        value = start_value

        for i, t in enumerate(self.times):
            self.values.append(value)
            value += timestep * self.function(t, value, *fargs)

        return self.times, self.values

    def runAnalytical(
        self, timestep=0.1, tmin=0.0, tmax=1.0, start_value=0.0, fargs=[]
    ):
        """_summary_

        Args:
            timestep (float, optional): time step used for the list of time values. Defaults to 0.1.
            tmin (float, optional): starting point of t. Defaults to 0.0.
            tmax (float, optional): end point of t. When separated a whole number of timesteps from tmin, the list of time values excludes tmax. Defaults to 1.0.
            start_value (float, optional): starting value of the independent variable x. Defaults to 0.0.
            fargs (list, optional): optional extra function parameters. Defaults to [].

        Returns:
            tuple: (numpy array of time values, list of x values from analytic solution)
        """
        self.analytic_values = []

        if self.analytical != None:

            self.times = np.arange(tmin, tmax, timestep)
            self.analytic_values = self.analytical(self.times, start_value, *fargs)

            return self.times, self.analytic_values

    def plotNumerical(self, ax, color="C0", label="numerical"):
        """Plots the numerical solution.

        Args:
            ax (matplotlib axis): axis to plot the numerical solution on.
            color (str, optional): line color. Defaults to "C0".
            label (str, optional): label for the plotted solution. Defaults to "numerical".

        Returns:
            axis: axis on which the numerical solution has been plotted.
        """
        ax.plot(self.times, self.values, color=color, label=label)
        return ax

    def plotAnalytical(self, ax, color="C1", label="analytical"):
        """Plots the analytical solution.

        Args:
            ax (matplotlib axis): axis to plot the analytical solution on.
            color (str, optional): line color. Defaults to "C1".
            label (str, optional): label for the plotted solution. Defaults to "analytical".

        Returns:
            axis: axis on which the analytical solution has been plotted.
        """
        if self.analytical != None:
            ax.plot(
                self.times,
                self.analytic_values,
                color=color,
                label=label,
                ms=1,
            )
        return ax


# example of usage
if __name__ == "__main__":

    # define functions
    def func3(t, x, fargs=None):
        return 2 * t

    def analytical3(t, start_value, fargs=None):
        return start_value + t ** 2

    # initialize ODE
    ode = numericalFO1D(func3, analytical3)

    # retrieve numerical and analytical solutions
    numerical = ode.runEuler(timestep=0.1, tmin=0, tmax=3, start_value=4, fargs=[])
    analytical = ode.runAnalytical(
        timestep=0.1, tmin=0, tmax=3, start_value=4, fargs=[]
    )

    # plot the solutions on a given axis
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ode.plotNumerical(ax)
    ode.plotAnalytical(ax)

    plt.legend()
    plt.show()
