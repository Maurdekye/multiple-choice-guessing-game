import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial
    
    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

directory = 'data_3'
plt.figure()

datas = []
names = []

for filename in os.listdir(directory):
    if filename.endswith('.csv'):
        # Load the data from the CSV file
        data = np.genfromtxt(os.path.join(directory, filename), delimiter=',')
        x = data[:, 0]
        y = data[:, 1]
        
        y_diff = np.ediff1d(y)
        y_diff = np.append(y_diff, [y_diff[-1]])
        y_diff = np.reshape(y_diff, (len(y_diff), 1))
        data = np.append(data, y_diff, axis=1)
        datas.append(data)
        names.append(filename)
        # y_diff_smooth = savitzky_golay(y_diff, 301, 3)
        
        # Plot the derivative of the fitted curve
        plt.plot(x, y_diff, label=filename)

def get_value(data, y):
    i = min(range(10_000), key=lambda i: abs(data[i, 1] - y))
    return data[i, 2]

maxes = []
for i in range(10_000):
    print(f"\r{i}", end="")
    maxes.append(max(range(len(datas)), key=lambda x:get_value(datas[x], i)))
  
with open(f"{directory}/compiled.csv", "w") as f:
    for (i, row) in enumerate(maxes):
        f.write(f"{i},{names[row]}\n")

plt.xlabel('X')
plt.ylabel('Derivative')
plt.title('Derivatives of Logarithmic Fits')
plt.legend()
plt.grid(True)
plt.savefig("derivatives_pyplot.png")
plt.show()