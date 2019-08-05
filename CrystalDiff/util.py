import datetime
import time
import numpy as np

"""
This module is the lowest-level module. It does not depend on another modules.
"""
pi = np.pi
hbar = 0.0006582119514  # This is the reduced planck constant in keV/fs

c = 299792458. * 1e-9  # The speed of light in um / fs


# --------------------------------------------------------------
#               Simple functions
# --------------------------------------------------------------
def time_stamp():
    """
    Get a time stamp
    :return: A time stamp of the form '%Y_%m_%d_%H_%M_%S'
    """
    stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')
    return stamp


def exp_stable(x):
    """
    This function calculate the exponential of a complex variable in a stable way.
    :param x:
    :return:
    """
    re = x.real
    im = x.imag

    im = np.mod(im, 2 * pi)
    phase = np.cos(im) + 1.j * np.sin(im)

    # Build a mask to find too small values
    # Assume that when re is less than -100, define the value to be 0
    magnitude = np.zeros_like(re, dtype=np.complex128)
    magnitude[re >= -100] = np.exp(re[re >= -100]) + 0.j
    return np.multiply(magnitude, phase)


def l2_norm(x):
    return np.sqrt(np.sum(np.square(x)))


def l2_square(x):
    return np.sum(np.square(x))


def l2_norm_batch(x):
    return np.sqrt(np.sum(np.square(x), axis=-1))


def l2_square_batch(x):
    return np.sum(np.square(x), axis=-1)


# --------------------------------------------------------------
#               Unit conversion
# --------------------------------------------------------------
def kev_to_petahertz_frequency(energy):
    return energy / hbar * 2 * pi


def kev_to_petahertz_angular_frequency(energy):
    return energy / hbar


def kev_to_wave_number(energy):
    return energy / hbar / c


def petahertz_frequency_to_kev(frequency):
    return hbar * 2 * pi * frequency


def petahertz_angular_frequency_to_kev(angular_frequency):
    return hbar * angular_frequency


def petahertz_angular_frequency_to_wave_number(angular_frequency):
    return angular_frequency / c


def wave_number_to_kev(wavevec):
    return wavevec * hbar * c


# --------------------------------------------------------------
#          Get wave vectors
# --------------------------------------------------------------
def get_gamma_h_3d(k0_grid, k_grid, crystal_h, z):
    """

    :param k0_grid: A array of wavevectors. The last dimension is each specific wave vectors
    :param k_grid: The length of the array of the wave vectors
    :param crystal_h: The reciprocal for the diffraction
    :param z: The normal direction of the lattice
    :return:
    """
    return np.divide(np.dot(k0_grid, z) + np.dot(crystal_h, z), k_grid)


def get_gamma_0_3d(k0_grid, k_grid, z):
    """

    :param k0_grid: A array of wavevectors. The last dimension is each specific wave vectors
    :param k_grid: The length of the array of the wave vectors
    :param z: The normal direction of the lattice
    :return:
    """
    return np.divide(np.dot(k0_grid, z), k_grid)


def get_rho_h_3d(kh_grid, k_grid, crystal_h, z):
    """
    This parameter is used to derive the incident wave vector from the reflected wave vector

    :param kh_grid:
    :param k_grid:
    :param crystal_h:
    :param z:
    :return:
    """
    return np.divide(np.dot(crystal_h, z) - np.dot(kh_grid, z), k_grid)


def get_alpha_3d(k0_grid, k_grid, crystal_h):
    """

    :param k0_grid:
    :param k_grid:
    :param crystal_h:
    :return:
    """
    return np.divide(2 * np.dot(k0_grid, crystal_h) + l2_square(crystal_h),
                     np.square(k_grid))


def get_epsilon_3d(kh_grid, k_grid, crystal_h):
    """
    This parameter is used to derive the incident wave vector from the reflected wave vector

    :param kh_grid:
    :param k_grid:
    :param crystal_h:
    :return:
    """
    return np.divide(l2_square(crystal_h) - 2 * np.dot(kh_grid, crystal_h),
                     np.square(k_grid))


def get_asymmetry_factor_batch(gamma_h_grid, gamma_0_grid):
    """

    :param gamma_h_grid:
    :param gamma_0_grid:
    :return:
    """
    return np.divide(gamma_0_grid, gamma_h_grid)


def get_momentum_transfer(k_grid, gamma_h_grid, alpha_grid):
    """

    :param k_grid:
    :param gamma_h_grid:
    :param alpha_grid:
    :return:
    """
    tmp = np.sqrt(np.square(gamma_h_grid) - alpha_grid)
    tmp_plus = np.abs(-gamma_h_grid + tmp)
    tmp_minus = np.abs(-gamma_h_grid - tmp)

    # Get the sign such that the abs is small
    sign = np.ones_like(k_grid, dtype=np.float64)
    sign[tmp_plus > tmp_minus] = -1.
    tmp = np.multiply(tmp, sign)

    return np.multiply(k_grid, -gamma_h_grid + tmp)


def get_momentum_transfer_reverse(k_grid, rho_h_grid, epsilon_grid):
    """

    :param k_grid:
    :param rho_h_grid:
    :param epsilon_grid:
    :return:
    """
    return get_momentum_transfer(k_grid, gamma_h_grid=rho_h_grid, alpha_grid=epsilon_grid)


def get_output_wave_vector(k0_grid, k_grid, crystal_h, z):
    """
    Given the input wave vector, derive the output wave vector

    :param k0_grid:
    :param k_grid:
    :param crystal_h:
    :param z:
    :return:
    """
    gamma_h_grid = get_gamma_h_3d(k0_grid=k0_grid,
                                  k_grid=k_grid,
                                  crystal_h=crystal_h,
                                  z=z)
    alpha_grid = get_alpha_3d(k0_grid=k0_grid,
                              k_grid=k_grid,
                              crystal_h=crystal_h)

    momentum_transfer = get_momentum_transfer(k_grid=k_grid,
                                              gamma_h_grid=gamma_h_grid,
                                              alpha_grid=alpha_grid)
    momentum_transfer = np.multiply(z[np.newaxis, :], momentum_transfer[:, np.newaxis])

    kh_grid = k0_grid + crystal_h[np.newaxis, :] + momentum_transfer

    rho_h_grid = get_rho_h_3d(kh_grid=kh_grid,
                              k_grid=k_grid,
                              crystal_h=crystal_h,
                              z=z)
    epsilon_grid = get_epsilon_3d(kh_grid=kh_grid,
                                  k_grid=k_grid,
                                  crystal_h=crystal_h)
    return {"kh_grid": kh_grid,
            "rho_h_grid": rho_h_grid,
            "epsilon_grid": epsilon_grid}


def get_input_wave_vector(kh_grid, k_grid, crystal_h, z):
    """
    Given the output wave vector, derive the corresponding incident wave vector.
    :param kh_grid:
    :param k_grid:
    :param crystal_h:
    :param z:
    :return:
    """
    rho_h_grid = get_rho_h_3d(kh_grid=kh_grid,
                              k_grid=k_grid,
                              crystal_h=crystal_h,
                              z=z)

    epsilon_grid = get_epsilon_3d(kh_grid=kh_grid,
                                  k_grid=k_grid,
                                  crystal_h=crystal_h)

    momentum_transfer = get_momentum_transfer_reverse(k_grid=k_grid,
                                                      rho_h_grid=rho_h_grid,
                                                      epsilon_grid=epsilon_grid)
    momentum_transfer = np.multiply(z[np.newaxis, :], momentum_transfer[:, np.newaxis])

    # get the incident wave vector
    k0_grid = kh_grid - crystal_h[np.newaxis, :] - momentum_transfer

    gamma_0_grid = get_gamma_0_3d(k0_grid=k0_grid, k_grid=k_grid, z=z)
    gamma_h_grid = get_gamma_h_3d(k0_grid=k0_grid, k_grid=k_grid, z=z,
                                  crystal_h=crystal_h)
    alpha_grid = get_alpha_3d(k0_grid=k0_grid,
                              k_grid=k_grid,
                              crystal_h=crystal_h)
    return {"k0_grid": k0_grid,
            "gamma_0_grid": gamma_0_grid,
            "gamma_h_grid": gamma_h_grid,
            "alpha_grid": alpha_grid}


def get_intersection_point(s, k, n, x0):
    """
    Assume that a line starts from point s along the direction k. It will intersect with
    the plane that passes through point x0 and has normal direction n. The function find the
    resulted intersection point.

    This function assumes that the arguments are arrays of points.

    :param s: array of shape [n,3], starting points for each array
    :param k: array of shape [n,3], the direction for each array
    :param n: array of shape [3], the normal direction of the surface
    :param x0: array of shape [3], one point on this surface
    :return:
    """
    # The intersection points for each array
    x = np.copy(s)

    # Do the math
    tmp = np.divide(np.dot(x0[np.newaxis, :] - s, n), np.dot(k, n))
    x += np.multiply(tmp[:, np.newaxis], k)
    return x


# --------------------------------------------------------------
#          Geometric operation
# --------------------------------------------------------------
def get_rot_mat_dict(kin, kout, aux=np.array([1., 0., 0.])):
    """
    Given the incident central momentum, the diffracted central momentum
    and the reciprocal lattice, decide the rotation matrices such that
    1. From the incident pulse frame to the device frame
    2. From the device frame to the diffracted pulse frame
    3. From the incident pulse frame to the diffracted pulse frame

    For the incident pulse, the kin and h span z-y plane.
    Fot the output pulse, the kout and kin span the z-y plane.

    I use the convention such that when the angle is small, the z axis
    is the propagation direction. The y is the vertical direction. The
    x is the horizontal direction.

    :param kin:
    :param kout:
    :param aux: Sometimes, when the kin and kout are exactly the same, or differ only by
                a very small a mount, then we would like to specify an aux vector to
                make the rotation operation more realistic.
    :return:
    """
    # Get a holder
    rot_mat_dict = {}

    # -------------------------------------------------
    # Device to the incident pulse
    # -------------------------------------------------
    tmp_z = kin / l2_norm(kin)

    tmp_y = np.cross(tmp_z, aux)
    tmp_y /= l2_norm(tmp_y)

    tmp_x = np.cross(tmp_y, tmp_z)
    tmp_x /= l2_norm(tmp_x)

    # Rotation matrix from the incident pulse to the device
    rot_mat_dict.update({"Device to In-Pulse": np.vstack([tmp_x, tmp_y, tmp_z])})
    rot_mat_dict.update({"In-Pulse to Device": rot_mat_dict["Device to In-Pulse"].T})

    # -------------------------------------------------
    # Device to the output pulse
    # -------------------------------------------------
    # When the input pulse and the output pulse are not along the same direction.
    new_z = kout / l2_norm(kout)

    new_y = np.cross(new_z, aux)
    new_y /= l2_norm(new_y)

    new_x = np.cross(new_y, new_z)
    new_x /= l2_norm(new_x)

    # Rotation matrix from the output pulse to the device
    rot_mat_dict.update({"Device to Out-Pulse": np.vstack([new_x, new_y, new_z])})
    rot_mat_dict.update({"Out-Pulse to Device": rot_mat_dict["Device to Out-Pulse"].T})

    # -------------------------------------------------
    # Incident pulse to the output pulse
    # -------------------------------------------------

    tmp_matrix = np.dot(rot_mat_dict["In-Pulse to Device"],
                        rot_mat_dict["Device to Out-Pulse"])

    rot_mat_dict.update({"In-Pulse to Out-Pulse": tmp_matrix})
    rot_mat_dict.update({"Out-Pulse to In-Pulse": tmp_matrix.T})

    return rot_mat_dict


def get_total_path_length(intersection_point_list):
    """
    Get the path length of a series of points

    :param intersection_point_list:
    :return:
    """
    number = len(intersection_point_list)
    total_path = 0.
    for l in range(number - 1):
        total_path += l2_norm(intersection_point_list[l + 1] -
                              intersection_point_list[l])

    return total_path


def get_phase_1d(k_grid, omega_grid, dx, dt):
    pre_phase = np.dot(k_grid, dx) - omega_grid * dt
    phase = np.cos(pre_phase) + 1.j * np.sin(pre_phase)
    return pre_phase, phase


# ---------------------------------------------------------------------------
#                     Post-process
# ---------------------------------------------------------------------------
def get_2d_sigma_matrix(density_2d, x_values, y_values):
    """

    :param density_2d:
    :param x_values:
    :param y_values:
    :return:
    """
    # Calculate the  x sigma
    density_x = np.sum(density_2d, axis=1)
    mean_x = np.sum(np.multiply(x_values, density_x))
    mean_x2 = np.sum(np.multiply(np.square(x_values), density_x))
    cov_xx = mean_x2 - mean_x ** 2

    # Calculate the  y sigma
    density_y = np.sum(density_2d, axis=0)
    mean_y = np.sum(np.multiply(y_values, density_y))
    mean_y2 = np.sum(np.multiply(np.square(y_values), density_y))
    cov_yy = mean_y2 - mean_y ** 2

    # Calculate the xy sigma
    tmp = np.multiply(density_2d, x_values[:, np.newaxis])
    tmp = np.multiply(tmp, y_values[np.newaxis, :])
    cov_xy = np.sum(tmp) - mean_x * mean_y

    # construct the Sigma matrix
    sigma_mat = np.zeros((2, 2))
    sigma_mat[0, 0] = cov_xx
    sigma_mat[0, 1] = cov_xy
    sigma_mat[1, 0] = cov_xy
    sigma_mat[1, 1] = cov_yy

    eig, eig_vals = np.linalg.eig(sigma_mat)

    return sigma_mat, eig, eig_vals
