"""dallys_functions.py
    These functions are tested in python 3.8.10

    DESCRIPTION:

        Collection of functions that are useful when designing, simulating or
        measuring delta-sigma adcs.

    CLASSES:

        None

    FUNCTIONS:

        optimized_zeros(order)
            Zero optimization locations for delta-sigma ntf.

        create_ntf(order, h_inf, fs, fbw)
            Create discrete-time noise transfer function using Butterworth
            filter.

        create_ntf_cheby(order, h_inf, fs, fbw, ripple)
            Create discrete-time noise tranfer function using Chebyshev filter.

        plot_freq(f, h, ax, ax_lim, **kwargs)
            Plot frequency response.

        plot_pzmap(z, p, ax, continuous=False, **kwargs)
            Plot pole-zero map.

        plot_rlocus(p_re, p_im, z_re, z_im, ax, n_min, k_min)
            Plot root locus.

        poly_addn(polys)
            Add n number of polynomials together.

        poly_muln(polys)
            Multiply n number of polynomials together.

        calc_freq(tf, n_points, fmax, continuous=True)
            Calculate frequency response of a transfer function.

        tf_parrallel(tf1, tf2)
            Connect two transfer functions in parrallel.

        tf_cascade(tf1, tf2)
            Connect two transfer functions in series.

        tf_feedback(tf1, tf2, k=-1.0)
            Create feedback connection of transfer functions.

        tf_print(tf, tf_name, var)
            Print formatted output of a transfer function.

        db20(h)
            Calculate magnitude in 20dB scale.

        db10(h)
            Calculate magnitude in 10dB scale.

        angle(h)
            Calculate angle in degrees.

        fft_window(n_samples, window_name)
            Collection of FFT window functions.

        fft_rms(fftdata, bins)
            Calculate RMS value from FFT.

        calc_fft(data, window)
            Calculate FFT.

        calc_averaged_fft(data, window, n_points, n_average)
            Average multiple ffts together with 50% overlap.

        calc_rlocus(open_loop_tf, feedback_gain)
            Calculate root locus.

"""
import warnings
import scipy.signal as ss
import numpy as np
from numpy import array as arr
import matplotlib.pyplot as plt
import matplotlib.patches as plt2

__author__ = "dally"
__version__ = "0.9"
__maintainer__ = "dally"
__email__ = "valtteri.dahlbacka@kaikutek.com"
__status__ = "production"


###############################################################################
# delta-sigma functions
###############################################################################


def optimized_zeros(order : int) -> list:
    """Zero optimization locations for delta-sigma ntf
    (from book *Understanding Delta-Sigma Data Converters p. 98*).

        Parameters:
            order : order of the dsm (number of zeros)

        Returns:
            List of optimized zeros, relative to bandwidth edge
    """
    return np.array((
        [0], #1
        [-1/np.sqrt(3), 1/np.sqrt(3)], #2
        [-np.sqrt(3/5), 0, np.sqrt(3/5)], #3
        [-0.861, -0.340, 0.340, 0.861], #4
        [-0.906, -0.539, 0, 0.539, 0.906], #5
        [-0.93247, -0.66121, -0.23862, 0.23862, 0.66121, 0.93247], #6
        [-0.94911, -0.74153, -0.49585, 0, 0.49585, 0.74153, 0.94911], #7
        [-0.96029, -0.79667, -0.52553, -0.18343, 0.18343, 0.52553,\
         0.79667, 0.96029], #8
    )[order-1])


def create_ntf(order : int, h_inf : float, fs : float, fbw : float) -> list:
    """Create discrete-time noise transfer function using Butterworth filter

        Parameters:
            order : order of the dsm
            h_inf : target out-of-band gain
            fs : sampling frequency
            fbw : bandwidth

        Returns:
            List where first element holds numerator coefficients and
            second element holds denominator coefficients
    """
    # use binary search to find cutoff frequency that gives the target
    # out-of-band gain
    fcut = 0.25
    for n in range(1, 32):
        b, a = ss.butter(order, fcut, btype='high',\
                         analog=False, output='ba', fs=1)
        obg = 1/b[0]
        fcut = fcut+np.sign(h_inf-obg)*0.25*2**(-n)
    # convert to zpk
    z, p, k = ss.tf2zpk(b, a)
    # optimize zeros (if you don't want to optimize, set fbw = 0)
    fzeros = optimized_zeros(order)*fbw
    z = np.exp(1j*2*np.pi*fzeros/fs)\
        *np.exp(1j*2*np.pi/fs)
    # move poles (IQ DSM only)
    p = p*np.exp(1j*2*np.pi/fs)
    # recreate filter transfer function with new zeros
    b, a = ss.zpk2tf(z, p, k)
    return [np.real(b), np.real(a)]


def create_ntf_cheby(order : int, h_inf : float, fs : float, fbw : float,\
                     ripple : float) -> list:
    """Create discrete-time noise tranfer function using Chebyshev filter.
    Can be better for low OSR modulators.

        Parameters:
            order : order of the dsm
            h_inf : target out-of-band gain
            fs : sampling frequency
            fbw : bandwidth
            ripple : out-of-band ripple in dB

        Returns:
            List where first element holds numerator coefficients and
            second element holds denominator coefficients
    """
    # use binary search to find cutoff frequency that gives the target
    # out-of-band gain
    fcut = 0.25
    for n in range(1, 32):
        b, a = ss.cheby1(order, ripple, fcut, btype='high',\
                             analog=False, output='ba', fs=1)
        obg = 1/b[0]
        fcut = fcut+np.sign(h_inf-obg)*0.25*2**(-n)
    # convert to zpk
    z, p, k = ss.tf2zpk(b, a)
    # optimize zeros (if you don't want to optimize, set fbw = 0)
    fzeros = optimized_zeros(order)*fbw
    z = np.exp(1j*2*np.pi*fzeros/fs)\
        *np.exp(1j*2*np.pi/fs)
    # move poles (IQ DSM only)
    p = p*np.exp(1j*2*np.pi/fs)
    # recreate filter transfer function with new zeros
    b, a = ss.zpk2tf(z, p, k)
    return [b, a]


###############################################################################
# plotting
###############################################################################


def plot_freq(f : np.ndarray, h : np.ndarray, ax : plt.Axes, ax_lim : list,\
              **kwargs) -> None:
    """Plot frequency response.

        Parameters:
            f : 1D array with frequency points for the plot (x-axis)
            h : 1D array with magnitude points for the plot (y-axis)
            ax : matplotlib axis where to place the plot
            ax_lim : List of axis limits
            **kwargs : additional keyword arguments to pass to semilogx function

        Returns:
            None
    """
    ax.set_title('Frequency response')
    ax.set_ylabel('Gain [dB]')
    ax.set_xlabel('Frequency [Hz]')
    if kwargs:
        ax.semilogx(f, h, **kwargs)
    else:
        ax.semilogx(f, h)
    ax.grid()
    ax.axis(ax_lim)


def plot_pzmap(z : np.ndarray, p : np.ndarray, ax : plt.Axes,
               continuous : bool = False, **kwargs) -> None:
    """Plot pole-zero map

        Parameters:
            z : zeros
            p : poles
            ax : matplotlib axis where to place the plot
            continuous : False = discrete-time plot, True = continuous-time plot
            **kwargs : other optional plotting kwargs

        Returns:
            None
    """
    if z.size > 0 and p.size > 0:
        x_min = np.amin(np.floor([np.amin(np.real(z)),np.amin(np.real(p))]))
        x_max = np.amax(np.ceil([np.amax(np.real(z)), np.amax(np.real(p))]))
        y_min = np.amin(np.floor([np.amin(np.imag(z)),np.amin(np.imag(p))]))
        y_max = np.amax(np.ceil([np.amax(np.imag(z)), np.amax(np.imag(p))]))
    elif z.size > 0:
        x_min = np.amin(np.floor([np.amin(np.real(z))]))
        x_max = np.amax(np.ceil([np.amax(np.real(z))]))
        y_min = np.amin(np.floor([np.amin(np.imag(z))]))
        y_max = np.amax(np.ceil([np.amax(np.imag(z))]))
    elif p.size > 0:
        x_min = np.amin(np.floor([np.amin(np.real(p))]))
        x_max = np.amax(np.ceil([np.amax(np.real(p))]))
        y_min = np.amin(np.floor([np.amin(np.imag(p))]))
        y_max = np.amax(np.ceil([np.amax(np.imag(p))]))
    ax.set_title('PZMAP')
    ax.set_ylabel('Img')
    ax.set_xlabel('Real')
    if not continuous:
        circ = plt2.Circle((0,0),radius=1, fill=False, color='black',\
                           ls='solid', alpha=0.1)
        ax.add_patch(circ)
    ax.vlines(0,y_min-0.1,y_max+0.1,color='gray')
    ax.hlines(0,x_min-0.1,x_max+0.1,color='gray')
    ax.axis([x_min-0.1,x_max+0.1,y_min-0.1,y_max+0.1])
    if kwargs:
        if z.size > 0:
            ax.plot(np.real(z),np.imag(z),marker="o",linestyle="None", **kwargs)
        if p.size > 0:
            ax.plot(np.real(p),np.imag(p),marker="x",linestyle="None", **kwargs)
    else:
        if z.size > 0:
            ax.plot(np.real(z),np.imag(z),marker="o",linestyle="None")
        if p.size > 0:
            ax.plot(np.real(p),np.imag(p),marker="x",linestyle="None")
    ax.grid()


def plot_rlocus(p_re : np.ndarray, p_im : np.ndarray,\
                z_re : np.ndarray, z_im : np.ndarray,\
                ax : plt.Axes, n_min : int, k_min : float) -> None:
    """Plot root locus.

        Parameters:
            p_re : poles' real part
            p_im : poles' imaginary part
            z_re : zeros' real part
            z_im : zeros' imaginary part
            ax : matplotlib axis where to place the plot
            n_min : index of minimum stable k
            k_min : value of minimum stable k

        Returns:
            None
    """
    x_min = np.amin(np.floor([np.amin(z_re),np.amin(p_re)]))
    x_max = np.amax(np.ceil([np.amax(z_re),np.amax(p_re)]))
    y_min = np.amin(np.floor([np.amin(z_im),np.amin(p_im)]))
    y_max = np.amax(np.ceil([np.amax(z_im),np.amax(p_im)]))
    ax.hlines(0, x_min, x_max, color='k', linestyles=':', label='' )
    ax.vlines(0, y_min, y_max, color='k', linestyles=':', label='' )
    ax.plot(p_re,p_im, linewidth=3.0)
    ax.plot(p_re[n_min,:],p_im[n_min,:], marker='x', markersize=10,\
            linestyle='None', color='k')
    ax.plot(z_re[n_min,:],z_im[n_min,:], marker='o', markersize=5,\
            linestyle='None', color='k')
    ax.axis([x_min, x_max, y_min, y_max])
    ax.grid()
    ax.set_title(f'Root locus (Minimum stable k = {k_min:.2f})')
    ax.set_xlabel('real')
    ax.set_ylabel('imag')


###############################################################################
# polynomials
###############################################################################


def poly_addn(polys : list) -> np.ndarray:
    """Add n number of polynomials together.

        Parameters:
            polys : list of polynomials to add together
                    each polynomial should be given as np.ndarray

        Returns:
            total : summed polynomial
    """
    total = arr([])
    for poly in polys:
        total = np.polyadd(poly, total)
    return total


def poly_muln(polys : np.ndarray) -> np.ndarray:
    """Multiply n number of polynomials together.

        Parameters:
            polys : list of polynomials to multiply together
                    each polynomial should be given as np.ndarray

        Returns:
            total : multiplied polynomial
    """
    total = arr([1])
    for poly in polys:
        total = np.polymul(poly, total)
    return total


###############################################################################
# transfer functions
###############################################################################


def calc_freq(tf : list, n_points : int, fmax : float,\
              continuous : bool = True) -> tuple:
    """Calculate frequency response of a transfer function.

        Parameters:
            tf : transfer function given as list with 2 np.ndarray elements
                 1st element is numerator
                 2nd element is denominator
            continuous : True = continuous-time tf, False = discrete-time tf
            n_points : number of points in frequency response
            fmax : maximum frequency

        Returns:
            frequency and magnitude
    """

    if continuous:
        # calculate frequency response
        f = np.linspace(0, fmax, n_points+1)
        w, h = ss.freqs(tf[0], tf[1], worN=2*np.pi*f)
        h[np.where(h==0)] = 1e-18
    else:
        # calculate frequency response
        w, h = ss.freqz(tf[0], tf[1], worN=n_points, whole=True, plot=None,\
                        fs=2*np.pi*fmax, include_nyquist=False)
        f = np.append(w/(2*np.pi), fmax) # convert rad/s to Hz
        h = np.append(h,h[0])
        # convert zero values to small value to
        # avoid divide by zero error
        h[np.where(h==0)] = 1e-18

    return f, h


def tf_parrallel(tf1 : list, tf2 : list) -> list:
    """Connect two transfer functions in parrallel.

        Parameters:
            tf1 : transfer function given as list with 2 np.ndarray elements
                  1st element is numerator
                  2nd element is denominator
            tf2 : transfer function given as list with 2 np.ndarray elements
                  1st element is numerator
                  2nd element is denominator

        Returns:
            Combined transfer function.
    """
    tf_out = [arr([]),arr([])]
    tf_out[0] = poly_addn([poly_muln([tf1[0], tf2[1]]), poly_muln([tf1[1], tf2[0]])])
    tf_out[1] = poly_muln([tf1[1], tf2[1]])
    return tf_out


def tf_cascade(tf1 : list, tf2 : list) -> list:
    """Connect two transfer functions in series.

        Parameters:
            tf1 : transfer function given as list with 2 np.ndarray elements
                  1st element is numerator
                  2nd element is denominator
            tf2 : transfer function given as list with 2 np.ndarray elements
                  1st element is numerator
                  2nd element is denominator

        Returns:
            Combined transfer function.
    """
    tf_out = [arr([]),arr([])]
    tf_out[0] = poly_muln([tf1[0], tf2[0]])
    tf_out[1] = poly_muln([tf1[1], tf2[1]])
    return tf_out


def tf_feedback(tf1 : list, tf2 : list, k : float = -1.0) -> list:
    """Create feedback connection of transfer functions.

        Parameters:
            tf1 : direct path transfer function given as list with 2 np.ndarrays
                  1st element is numerator
                  2nd element is denominator
            tf2 : feedback transfer function given as list with 2 np.ndarrays
                  1st element is numerator
                  2nd element is denominator
            k : feedback gain

        Returns:
            Combined transfer function.
    """
    tf_out = [arr([]),arr([])]
    tf_out[1] = poly_addn([poly_muln([tf1[1], tf2[1]]), poly_muln([tf1[0], -1*k*tf2[0]])])
    tf_out[0] = poly_muln([tf1[0], tf2[1]])
    if tf_out[1][0] < 0:
        tf_out[0] = -1*tf_out[0]
        tf_out[1] = -1*tf_out[1]
    return tf_out


def tf_print(tf : list, tf_name : str, var : str) -> None:
    """Print formatted output of a transfer function.

        Parameters:
            tf : transfer function given as list with 2 np.ndarrays
                 1st element is numerator
                 2nd element is denominator

            tf_name : name of the transfer function
            var : variables of teh transfer function

        Returns:
            None
    """
    numerator = ' '.join([f'{x[1]:+.3e} {var}^{len(tf[0])-x[0]-1:d}'\
                          if (len(tf[0])-x[0]-1) > 0\
                          else f'{x[1]:+.3e}'\
                          for x in enumerate(tf[0])])
    denominator = ' '.join([f'{x[1]:+.3e} {var}^{len(tf[1])-x[0]-1:d}'\
                            if (len(tf[1])-x[0]-1) > 0\
                            else f'{x[1]:+.3e}'\
                            for x in enumerate(tf[1])])
    print((len(tf_name)+3)*' '+numerator)
    if len(numerator) > len(denominator):
        print(f'{tf_name} = '+len(numerator)*'-')
    else:
        print(f'{tf_name} = '+len(denominator)*'-')
    print((len(tf_name)+3)*' '+denominator)


###############################################################################
# fft
###############################################################################


def db20(h : np.ndarray) -> np.ndarray:
    """Calculate magnitude in 20dB scale.

        Parameters:
            h : fft data

        Returns:
            20dB magnitude of the input fft
    """
    return 20*np.log10(np.abs(h))


def db10(h : np.ndarray) -> np.ndarray:
    """Calculate magnitude in 10dB scale.

        Parameters:
            h : fft data

        Returns:
            10dB magnitude of the input fft
    """
    return 10*np.log10(np.abs(h))


def angle(h : np.ndarray) -> np.ndarray:
    """Calculate angle in degrees.

        Parameters:
            h : fft data

        Returns:
            Phase of the input fft (in degrees)
    """
    return 360*np.angle(h)/(2*np.pi)


def fft_window(n_samples : int , window_name : str) -> dict:
    """Collection of FFT window functions.

        Parameters:
            n_samples : number of points in the window
            window_name : name of the window (see below for available options)

        Returns:
            w : Dictionary containing fft window
                function : vector holding the window function
                nbins : number of nonzero signal bins
                energy : window energy (some crazy people use this to
                         normalize fft)
                offset : dc offset (normal people use this to normalize fft)
                nbw : noise bandwidth
    """
    # collection of windows
    x = np.linspace(0, n_samples-1, n_samples)
    if window_name == "hann":
        w = {
        "function" : 0.5*(1-np.cos(2*np.pi*x/n_samples)),
        "nbins" : 3
        }
    elif window_name == "hann2":
        w = {
        "function" : 0.25*(1-np.cos(2*np.pi*x/n_samples))**2,
        "nbins" : 5
        }
    elif window_name == "hamming":
        w = {
        "function" : 25/46-(1-25/46)*np.cos(2*np.pi*x/n_samples),
        "nbins" : 5
        }
    elif window_name == "flattop":
        w = {
        "function" : 0.21557895\
                    -0.416631580*np.cos(2*np.pi*x/(n_samples-1))\
                    +0.277263158*np.cos(4*np.pi*x/(n_samples-1))\
                    -0.083578947*np.cos(6*np.pi*x/(n_samples-1))\
                    +0.006947368*np.cos(8*np.pi*x/(n_samples-1)),
        "nbins" : 9
        }
    elif window_name == "blackman":
        w = {
        "function" : (7938-9240*np.cos(2*np.pi*x/n_samples)\
                      +1430*np.cos(4*np.pi*x/n_samples))/18608,
        "nbins" : 5
        }
    else:
        w = {
        "function" : np.ones([1,n_samples]),
        "nbins" : 1
        }
    w["energy"] = np.sum(np.abs(w["function"])**2)
    w["offset"] = np.abs(np.sum(w["function"]))
    w["nbw"] = w["energy"]/w["offset"]**2

    return w


def fft_rms(fftdata : np.ndarray, bins : list) -> float:
    """Calculate RMS value from FFT.

        Parameters:
            fftdata : fft data array
            bins : list of fft bins to use for calculation

        Returns:
            rms value
    """
    return np.sqrt(np.sum(np.abs(fftdata[bins])*np.abs(fftdata[bins])))


def calc_fft(data : np.ndarray, window : str) -> tuple:
    """Calculate FFT.

        Parameters:
            data : Sampled data to be fourier transformed
            window : name of window function

        Returns:
            fft and window
    """
    n_points = len(data) # number of points in fft
    w = fft_window(n_points, window) # create window function
    fft_data = np.fft.fft(data*w['function'])/w['offset']
    return fft_data, w


def calc_averaged_fft(data : np.ndarray, window : str, n_points : int,\
                      n_average : int) -> tuple:
    """Average multiple ffts together with 50% overlap.

        Parameters:
            data : Sampled data to be fourier transformed
            window : name of window function
            n_points : number of points in fft
            n_average : number of ffts to average

        Returns:
            fft and window
    """
    # create window function
    w = fft_window(n_points, window)
    # calculate ffts
    fft_datas = np.zeros((n_points, n_average))
    for n in range(n_average):
        windowed_data = data[n*int(n_points/2):n*int(n_points/2)+n_points]\
                        *w["function"]
        fft_datas[:,n] = np.abs(np.fft.fft(windowed_data)/(w["offset"]))**2
    # average ffts
    fft_data = np.sqrt(np.sum(fft_datas, axis=1)/n_average)
    return fft_data, w


###############################################################################
# other
###############################################################################


def calc_rlocus(open_loop_tf : list, feedback_gain : np.ndarray) -> tuple:
    """Calculate root locus.

        Parameters:
            open_loop_tf : Open loop transfer function
            feedback_gain : vector of different feedback gains to iterate over

        Returns:
            Locations of poles and zeros in complex plane for each feedback
            gain setting
    """
    # loop through different feedback gains
    p_re = np.zeros((len(feedback_gain), len(open_loop_tf[1])-1))
    p_im = np.zeros((len(feedback_gain), len(open_loop_tf[1])-1))
    z_re = np.zeros((len(feedback_gain), len(open_loop_tf[1])-1))
    z_im = np.zeros((len(feedback_gain), len(open_loop_tf[1])-1))
    for n, kn in enumerate(feedback_gain):
        ntf = [open_loop_tf[1], np.polyadd(open_loop_tf[1], kn*open_loop_tf[0])]
        z, p, _ = ss.tf2zpk(ntf[0], ntf[1])
        p_re[n,:] = np.real(p)
        p_im[n,:] = np.imag(p)
        z_re[n,:] = np.real(z)
        z_im[n,:] = np.imag(z)

    return p_re, p_im, z_re, z_im


###############################################################################
# measurement/simulation data processing functions
###############################################################################


def load_csv(input_file : str, columns : list, skiprows : int = 5,\
             delimiter : str = ',') -> np.ndarray:
    """Load data from csv file.

        Parameters:
            input_file : Name of the input file
            columns : list of column indexes that have relevant data
            skiprows : number of rows to skip in input_file
                       (use in case file contains header)
            delimiter : character used to separate columns in input_file

        Returns:
            Numpy array containing the data[rows,columns]
    """
    data = np.loadtxt(input_file, skiprows=skiprows, delimiter=delimiter)
    return data[:,columns]


def trim_data(data : np.ndarray, target_length : int) -> np.ndarray:
    """Trim raw data to smaller size.

        Parameters:
            data : array containing the data to be trimmed
            target_length : how many points final data set will hold

        Returns:
            Numpy array trimmed data
    """
    if data.shape[0] <= target_length:
        warnings.warn("Warning in function trim_data():\n"+\
                      "Data size is already smaller than target size!\n"+\
                      "Data trimming is skipped.")
        return data

    # first find interesting part of data (where data is toggling)
    toggles = np.where(np.any(np.diff(data, axis=0, prepend=0) != 0, axis=1))[0]
    n_start = 0 if (toggles[0] < 100) else toggles[0]
    n_stop = len(data) if (toggles[-1] > (len(data)-100)) else toggles[-1]

    # trim to size
    if target_length < (n_stop-n_start):
        return data[n_start:n_start+target_length,:]
    if target_length > n_stop:
        return data[:target_length,:]
    return data[n_stop-target_length:n_stop,:]


def resample_data(data : np.ndarray, clock : np.ndarray, clk_edge : str,\
                  data_skew : list = [0]) -> np.ndarray:
    """Resample data based on clock edge.

        Parameters:
            data : array containing the data to be resampled
            clock : array of 1s and 0s containing the clock used for sampling
            clock_edge : which clock edge to use for sampling
                         "rising", "falling" or "both"
            data_skew : add skew to data in comparison to clock
                        (list length needs to be 1 or the width of data)

        Returns:
            Numpy array of resampled data.
    """
    # add skew
    if len(data_skew) == 1:
        data = np.roll(data, data_skew[0], axis=0)
    elif len(data_skew) == data.shape[1]:
        for n, delay in enumerate(data_skew):
            data[:,n] = np.roll(data[:,n], delay, axis=0)
    else:
        warnings.warn("Warning in function resample_data():\n"+\
                      "Invalid data_skew given!\n"+\
                      "(len(data_skew)!=1 and len(data_skew)!=data_width)\n"+\
                      "Data skewing is skipped.")
    # find clock edges
    if clk_edge=='falling':
        sampling_points = (np.where(np.diff(clock)<-0.5)[0]).tolist()
    elif clk_edge=='rising':
        sampling_points = (np.where(np.diff(clock)>0.5)[0]).tolist()
    elif clk_edge=='both':
        sampling_points = (np.where(np.diff(clock)!=0)[0]).tolist()
    else:
        raise ValueError("Error in function resample_data():\n"+\
                         "clk_edge needs to be one of the following\n"+
                         "\"rising\", \"falling\" or \"both\"")
    return np.array([data[x,:] for x in sampling_points])


def combine_bits(bits : np.ndarray, bit_weights : list) -> np.ndarray:
    """Combine array of bits into single floating point vector.

        Parameters:
            bits : array containing the bits to be combined
            bit_weights : weights for each bit
                          for example, 4-bit array with first column being MSB
                          bit_weights = [8,4,2,1]

        Returns:
            Numpy array with floating point scaled between -1.0 and 1.0
    """
    if len(bit_weights) != bits.shape[1]:
        raise ValueError("Error in function combine_bits():\n"+\
                         "len(bit_weigths)!=number_of_bits")
    vout = bits*np.array(bit_weights)
    vout = np.sum(vout, axis=1)
    scale = np.sum(bit_weights)/2
    vout = (vout-scale)/scale
    return vout
