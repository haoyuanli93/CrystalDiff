import datetime
import time

import h5py as h5
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
#          Get output wave vectors
# --------------------------------------------------------------
def get_bragg_kout(kin, h, normal, compare_length=False):
    """
    This function produce the output wave vector from a Bragg reflection.

    :param kin: (3,) numpy array. The incident wave vector
    :param h: The reciprocal lattice of the crystal
    :param normal: The normal direction of the reflection surface.
                    For a bragg reflection, n is pointing to the inside of the crystal.
    :param compare_length: Whether compare the length of the incident wave vector and the output wave vector

    :return: kout: (3,) numpy array. The diffraction wave vector.
            ratio: When compare_length=True, the second output is the ratio between the incident wave number
                                        and the output wave number.
    """

    # kout holder
    kout = kin + h

    # Incident wave number
    klen = np.sqrt(np.dot(kin, kin))

    # Get gamma and alpha
    gammah = np.dot(kin + h, normal) / klen
    alpha = (2 * np.dot(kin, h) + np.dot(h, h)) / np.square(klen)

    if np.abs(-gammah - np.sqrt(gammah ** 2 - alpha)) > np.abs(-gammah + np.sqrt(gammah ** 2 - alpha)):
        momentum = klen * (-gammah + np.sqrt(gammah ** 2 - alpha))
    else:
        momentum = klen * (-gammah - np.sqrt(gammah ** 2 - alpha))

    # Add momentum transfer
    kout += normal * momentum

    if compare_length:
        return kout, klen / l2_norm(kout)
    else:
        return kout


# --------------------------------------------------------------
#          Geometry functions
# --------------------------------------------------------------
def get_intersection_point(s, k, n, x0):
    """
    Assume that a line starts from point s along the direction k. It will intersect with
    the plane that passes through point x0 and has normal direction n. The function find the
    resulted intersection point.

    This function assumes that the arguments are arrays of points.

    :param s: array of shape [3], starting points for each array
    :param k: array of shape [3], the direction for each array
    :param n: array of shape [3], the normal direction of the surface
    :param x0: array of shape [3], one point on this surface
    :return:
    """
    # The intersection points for each array
    x = np.copy(s)

    # Do the math
    tmp = np.divide(np.dot(x0 - s, n), np.dot(k, n))
    x += tmp * k
    return x


# --------------------------------------------------------------
#          Geometric operation
# --------------------------------------------------------------
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


# ---------------------------------------------------------------------------
#                     Grating
# ---------------------------------------------------------------------------
def get_grating_output_momentum(grating_wavenum, k_vec):
    """
    Calculate output momentum of the grating with the specified wave number and
    the corresponding incident k_vec

    :param grating_wavenum:
    :param k_vec:
    :return:
    """
    wavenum_reshape = np.reshape(grating_wavenum, (1, 3))
    return k_vec + wavenum_reshape


def get_grating_wavenumber_1d(direction, period, order):
    """

    :param direction:
    :param period:
    :param order:
    :return:
    """
    return order * direction * 2. * np.pi / period


def get_grating_period(dtheta, klen_in):
    """
    Derive the grating period based on the deviation angle and the incident wave number.
    Here, one assume that the incident wave vector is perpendicular to the the grating surface.

    :param dtheta:
    :param klen_in:
    :return:
    """
    period = 2 * np.pi / klen_in / np.tan(dtheta)
    return period


# ---------------------------------------------------------------------------
#                     IO
# ---------------------------------------------------------------------------
def save_branch_result_to_h5file(file_name, io_type, branch_name,
                                 result_3d_dict, result_2d_dict, check_dict):
    with h5.File(file_name, io_type) as h5file:
        group = h5file.create_group(branch_name)
        # Save the meta data
        group_check = group.create_group('check')
        for entry in list(check_dict.keys()):
            group_check.create_dataset(entry, data=check_dict[entry])

        group_2d = group.create_group('result_2d')
        for entry in list(result_2d_dict.keys()):
            group_2d.create_dataset(entry, data=result_2d_dict[entry])

        group_3d = group.create_group('result_3d')
        for entry in list(result_3d_dict.keys()):
            group_3d.create_dataset(entry, data=result_3d_dict[entry])


def time_stamp():
    """
    Get a time stamp
    :return: A time stamp of the form '%Y_%m_%d_%H_%M_%S'
    """
    stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')
    return stamp


# ---------------------------------------------------------------------------
#                  Get k mesh
# ---------------------------------------------------------------------------

def get_k_mesh_3d(number_x, number_y, number_z, delta_e_x, delta_e_y, delta_e_z):
    # Get the corresponding energy mesh
    energy_grid_x = np.linspace(start=- delta_e_x,
                                stop=+ delta_e_x,
                                num=number_x)
    energy_grid_y = np.linspace(start=- delta_e_y,
                                stop=+ delta_e_y,
                                num=number_y)
    energy_grid_z = np.linspace(start=- delta_e_z,
                                stop=+ delta_e_z,
                                num=number_z)

    # Get the k grid
    kx_grid = np.ascontiguousarray(kev_to_wave_number(energy=energy_grid_x))
    ky_grid = np.ascontiguousarray(kev_to_wave_number(energy=energy_grid_y))
    kz_grid = np.ascontiguousarray(kev_to_wave_number(energy=energy_grid_z))

    # Get the spatial mesh along x axis
    dkx = kev_to_wave_number(energy=energy_grid_x[1] - energy_grid_x[0])
    x_range = np.pi * 2 / dkx

    x_idx = np.linspace(start=-x_range / 2., stop=x_range / 2., num=number_x)
    x_idx_tick = ["{:.2f}".format(x) for x in x_idx]

    # Get the spatial mesh along y axis
    dky = kev_to_wave_number(energy=energy_grid_y[1] - energy_grid_y[0])
    y_range = np.pi * 2 / dky

    y_idx = np.linspace(start=-y_range / 2., stop=y_range / 2., num=number_y)
    y_idx_tick = ["{:.2f}".format(x) for x in y_idx]

    # Get the spatial mesh along z axis
    dkz = kev_to_wave_number(energy=energy_grid_z[1] - energy_grid_z[0])
    z_range = np.pi * 2 / dkz

    z_idx = np.linspace(start=-z_range / 2., stop=z_range / 2., num=number_z)
    z_idx_tick = ["{:.2f}".format(x) for x in z_idx]

    # Assemble the indexes and labels
    axis_info = {"x_range": x_range,
                 "x_idx": x_idx,
                 "x_idx_tick": x_idx_tick,
                 "dkx": dkx,
                 "energy_grid_x": energy_grid_x,

                 "y_range": y_range,
                 "y_idx": y_idx,
                 "y_idx_tick": y_idx_tick,
                 "dky": dky,
                 "energy_grid_y": energy_grid_y,

                 "z_range": z_range,
                 "z_idx": z_idx,
                 "z_idx_tick": z_idx_tick,
                 "dkz": dkz,
                 "energy_grid_z": energy_grid_z,
                 "z_time_idx": np.divide(z_idx, c),
                 "z_time_tick": ["{:.2f}".format(x) for x in np.divide(z_idx, c)],

                 "de_x_in_meV": np.linspace(start=- delta_e_x * 1e6,
                                            stop=+ delta_e_x * 1e6,
                                            num=number_x)}
    return kx_grid, ky_grid, kz_grid, axis_info


# ---------------------------------------------------
#              For DuMond Diagram
# ---------------------------------------------------
def get_klen_and_angular_mesh(k_num, theta_num, phi_num, energy_range, theta_range, phi_range):
    # Get the corresponding energy mesh
    energy_grid = np.linspace(start=energy_range[0], stop=energy_range[1], num=k_num)
    # Get the k grid
    klen_grid = np.ascontiguousarray(kev_to_wave_number(energy=energy_grid))

    # Get theta grid
    theta_grid = np.linspace(start=theta_range[0], stop=theta_range[1], num=theta_num)

    # Get phi grid
    phi_grid = np.linspace(start=phi_range[0], stop=phi_range[1], num=phi_num)

    info_dict = {"energy_grid": energy_grid,
                 "klen_grid": klen_grid,
                 "theta_grid": theta_grid,
                 "phi_grid": phi_grid}
    return info_dict
