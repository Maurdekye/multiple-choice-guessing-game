import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

def logarithmic(x, a, b, c):
    return a * np.log(x+abs(c)) + b

directory = 'data_2'

for filename in os.listdir(directory)[1:]:
    if filename.endswith('.csv'):
        # Load the data from the CSV file
        data = np.genfromtxt(os.path.join(directory, filename), delimiter=',')
        x = data[:, 0]
        y = data[:, 1]

        # Fit the logarithmic curve to the data
        popt, _ = curve_fit(logarithmic, x, y)
        a, b, c = popt

        # Generate points for the fitted curve
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = logarithmic(x_fit, a, b, c)

        # Plot the data points and the fitted curve
        plt.figure()
        plt.plot(x, y, 'bo', label='Data')
        plt.plot(x_fit, y_fit, 'r-', label='Fitted Curve')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Logarithmic Fit for {filename}')
        plt.legend()
        plt.grid(True)
        plt.savefig(filename.rsplit(".")[0] + "_pyplot.png")

        # Print the equation of the fitted curve
        print(f"Equation for {filename}: y = {a:.2f} * log(x + {c:.2f}) + {b:.2f}")
