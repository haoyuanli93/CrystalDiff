import numpy as np
from CrystalDiff import util

hbar = util.hbar  # This is the reduced planck constant in keV/fs
c = util.c  # The speed of light in um / fs
pi = util.pi
two_pi = 2 * pi


class GaussianPulse3D:
    def __init__(self):
        self.x0 = np.zeros(3, dtype=np.float64)
        self.k0 = np.zeros(3, dtype=np.float64)
        self.n = np.zeros(3, dtype=np.float64)

        self.omega0 = 20000.  # PHz

        # Basically, this mean that initially, we are in the frame where
        # different components of the pulse decouple. Then we rotate
        # back in to the lab frame.
        self.sigma_x = 0.1  # fs
        self.sigma_y = 33.  # fs
        self.sigma_z = 33.  # fs

        self.sigma_mat = np.diag(np.array([self.sigma_x ** 2,
                                           self.sigma_y ** 2,
                                           self.sigma_z ** 2], dtype=np.float64))

        # Intensity. Add a coefficient for overall intensity.
        self.scaling = 1.

        # Polarization
        self.polar = np.array([1., 0., 0.], dtype=np.complex128)


# --------------------------------------------------------------
#          Get spectrum
# --------------------------------------------------------------
def get_gaussian_pulse_spectrum(k_grid, sigma_mat, scaling, k0):
    # Get the momentum difference
    dk = k0[np.newaxis, :] - k_grid

    # Get the quadratic term
    quad_term = - (dk[:, 0] * sigma_mat[0, 0] * dk[:, 0] + dk[:, 0] * sigma_mat[0, 1] * dk[:, 1] +
                   dk[:, 0] * sigma_mat[0, 2] * dk[:, 2] +
                   dk[:, 1] * sigma_mat[1, 0] * dk[:, 0] + dk[:, 1] * sigma_mat[1, 1] * dk[:, 1] +
                   dk[:, 1] * sigma_mat[1, 2] * dk[:, 2] +
                   dk[:, 2] * sigma_mat[2, 0] * dk[:, 0] + dk[:, 2] * sigma_mat[2, 1] * dk[:, 1] +
                   dk[:, 2] * sigma_mat[2, 2] * dk[:, 2]) / 2.

    # if quad_term >= -200:
    magnitude = scaling * (np.exp(quad_term) + 0.j)
    return magnitude


def get_square_pulse_spectrum(k_grid, k0, a_val, b_val, c_val, scaling):
    dk = k_grid - k0[np.newaxis, :]
    spectrum = np.multiply(np.multiply(
        np.sinc((a_val / 2. / np.pi) * dk[:, 0]),
        np.sinc((b_val / 2. / np.pi) * dk[:, 1])),
        np.sinc((c_val / 2. / np.pi) * dk[:, 2])) + 0.j
    spectrum *= scaling

    return spectrum


def get_square_pulse_spectrum_smooth(k_grid, k0, a_val, b_val, c_val, scaling, sigma):
    dk = k_grid - k0[np.newaxis, :]
    spectrum = np.multiply(np.multiply(
        np.sinc((a_val / 2. / np.pi) * dk[:, 0]),
        np.sinc((b_val / 2. / np.pi) * dk[:, 1])),
        np.sinc((c_val / 2. / np.pi) * dk[:, 2])) + 0.j

    spectrum *= scaling

    # Add the Gaussian filter
    tmp = - (dk[:, 0] ** 2 + dk[:, 1] ** 2 + dk[:, 2] ** 2) * sigma ** 2 / 2.
    gaussian = np.exp(tmp)

    return np.multiply(spectrum, gaussian)
