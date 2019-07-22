import datetime
import time
import numpy as np
from numba import jit, complex128, float64

pi = np.pi
hbar = 0.0006582119514  # This is the reduced planck constant in keV/fs

c = 299792458. * 1e-9  # The speed of light in um / fs


def time_stamp():
    """
    Get a time stamp
    :return: A time stamp of the form '%Y_%m_%d_%H_%M_%S'
    """
    stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')
    return stamp


"""
This module is the lowest-level module. It does not depend on another modules."""


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


def sqrt_stable(x):
    """
    Get a stable version of the sqrt function. Notice that, in this function,
    the first return always have a positive imaginary part.

    :param x:
    :return:
    """
    tmp = np.sqrt(x)
    holder = np.sign(tmp.imag)

    tmp = np.multiply(holder, tmp)

    return tmp, -tmp


def multiply_stable(x, y):
    """
    This is a stable version of complex multiplication
    :param x:
    :param y:
    :return:
    """
    x_re = x.real
    x_im = x.imag
    y_re = y.real
    y_im = y.imag

    return (np.multiply(x_re, y_re) - np.multiply(x_im, y_im) +
            1.j * (np.multiply(x_im, y_re) + np.multiply(x_re, y_im)))


def l2_norm(x):
    return np.sqrt(np.sum(np.square(x)))


def l2_square(x):
    return np.sum(np.square(x))


def l2_norm_batch(x):
    return np.sqrt(np.sum(np.square(x), axis=-1))


def l2_square_batch(x):
    return np.sum(np.square(x), axis=-1)


#########################################################################################
# For 3D diffraction
#########################################################################################

#################################
# Get geometric relation
#################################
def get_crystal_frame(lattice_vector, crystal_normal, wavevector):
    """
    Get the reference frame. The result is a dictionary

    ref = {
            z_prime, the direction of lattice vector
            y_prime, the
            x

            rot_mat, the matrix to transfer vectors from the old frame in to this new frame.
            }

    :param lattice_vector:
    :param crystal_normal:
    :param wavevector:
    :return:
    """

    # First: normalize the two vectors
    lattice_vector = np.array(lattice_vector)
    lattice_vector /= np.sqrt(np.sum(np.square(lattice_vector)))

    crystal_normal = np.array(crystal_normal)
    crystal_normal /= np.sqrt(np.sum(np.square(crystal_normal)))

    # Decide if the two vectors are almost the same
    if np.sum(np.square(np.cross(lattice_vector, crystal_normal))) <= 1e-8:
        # The surface of the lattice is parallel to the surface of the crystal.
        # In this case, the diffraction is symmetric. Therefore, we set up the coordinate so that
        # the diffraction plane is in the x-z plane.

        # Normalize the incident wavevector
        wavevector = np.array(wavevector)
        wavevector /= np.sqrt(np.sum(np.square(wavevector)))

        tmp_y = np.cross(lattice_vector, wavevector)
        tmp_y /= l2_norm(tmp_y)

        tmp_x = np.cross(tmp_y, lattice_vector)
        tmp_x /= l2_norm(tmp_x)

        tmp_rotmat = np.stack((tmp_x, tmp_y, lattice_vector), axis=0)

        ref_frame = {'x_prime': tmp_x,
                     'y_prime': tmp_y,
                     'z_prime': np.copy(lattice_vector),
                     'rot_mat': tmp_rotmat,
                     'eta': 0.,
                     'eta sin': 0.,
                     'eta cos': 1.}
    else:

        # Define the y axis according to the surface and the lattice normal direction
        tmp_y = np.cross(lattice_vector, crystal_normal)
        tmp_eta_sin = l2_norm(tmp_y)
        tmp_eta_cos = np.sqrt(1 - tmp_eta_sin ** 2)

        tmp_y /= tmp_eta_sin

        tmp_x = np.cross(tmp_y, lattice_vector)

        tmp_rotmat = np.stack((tmp_x, tmp_y, lattice_vector), axis=0)

        ref_frame = {'x_prime': tmp_x,
                     'y_prime': tmp_y,
                     'z_prime': np.copy(lattice_vector),
                     'rot_mat': tmp_rotmat,
                     'eta': np.arcsin(tmp_eta_sin),
                     'eta sin': tmp_eta_sin,
                     'eta cos': tmp_eta_cos}

    return ref_frame


#################################
# Get geometric parameters for diffraction
#################################
def get_gamma_h_3d(k0_array, k_array, reciprocal_lattice, z):
    """

    :param k0_array: A array of wavevectors. The last dimension is each specific wave vectors
    :param k_array: The length of the array of the wave vectors
    :param reciprocal_lattice: The reciprocal for the diffraction
    :param z: The normal direction of the lattice
    :return:
    """
    return np.divide(np.dot(k0_array, z) + np.dot(reciprocal_lattice, z), k_array)


def get_gamma_0_3d(k0_array, k_array, z):
    """

    :param k0_array: A array of wavevectors. The last dimension is each specific wave vectors
    :param k_array: The length of the array of the wave vectors
    :param z: The normal direction of the lattice
    :return:
    """
    return np.divide(np.dot(k0_array, z), k_array)


def get_rho_h_3d(kh_array, k_array, reciprocal_lattice, z):
    """
    This parameter is used to derive the incident wave vector from the reflected wave vector

    :param kh_array:
    :param k_array:
    :param reciprocal_lattice:
    :param z:
    :return:
    """
    return np.divide(np.dot(reciprocal_lattice, z) - np.dot(kh_array, z), k_array)


def get_alpha_3d(k0_array, k_array, reciprocal_lattice):
    """

    :param k0_array:
    :param k_array:
    :param reciprocal_lattice:
    :return:
    """
    return np.divide(2 * np.dot(k0_array, reciprocal_lattice) + l2_square(reciprocal_lattice),
                     np.square(k_array))


def get_epsilon_3d(kh_array, k_array, reciprocal_lattice):
    """
    This parameter is used to derive the incident wave vector from the reflected wave vector

    :param kh_array:
    :param k_array:
    :param reciprocal_lattice:
    :return:
    """
    return np.divide(l2_square(reciprocal_lattice) - 2 * np.dot(kh_array, reciprocal_lattice),
                     np.square(k_array))


def get_asymmetry_factor_batch(gamma_h_array, gamma_0_array):
    """

    :param gamma_h_array:
    :param gamma_0_array:
    :return:
    """
    return np.divide(gamma_0_array, gamma_h_array)


#################################
# Get momentum transfer for crystals
#################################
def get_momentum_transfer(k_array, gamma_h_array, alpha_array):
    """

    :param k_array:
    :param gamma_h_array:
    :param alpha_array:
    :return:
    """
    tmp = np.sqrt(np.square(gamma_h_array) - alpha_array)
    tmp_plus = np.abs(-gamma_h_array + tmp)
    tmp_minus = np.abs(-gamma_h_array - tmp)

    # Get the sign such that the abs is small
    sign = np.ones_like(k_array, dtype=np.float64)
    sign[tmp_plus > tmp_minus] = -1.
    tmp = np.multiply(tmp, sign)

    return np.multiply(k_array, -gamma_h_array + tmp)


def get_momentum_transfer_reverse(k_array, rho_h_array, epsilon_array):
    """

    :param k_array:
    :param rho_h_array:
    :param epsilon_array:
    :return:
    """
    return get_momentum_transfer(k_array, gamma_h_array=rho_h_array, alpha_array=epsilon_array)


def derive_reflect_wavevec_and_rhoh_epsilon(k0_array, k_array, reciprocal_lattice, z):
    gamma_h_array = get_gamma_h_3d(k0_array=k0_array,
                                   k_array=k_array,
                                   reciprocal_lattice=reciprocal_lattice,
                                   z=z)
    alpha_array = get_alpha_3d(k0_array=k0_array,
                               k_array=k_array,
                               reciprocal_lattice=reciprocal_lattice)

    momentum_transfer = get_momentum_transfer(k_array=k_array,
                                              gamma_h_array=gamma_h_array,
                                              alpha_array=alpha_array)
    momentum_transfer = np.multiply(z[np.newaxis, :], momentum_transfer[:, np.newaxis])

    kh_array = k0_array + reciprocal_lattice[np.newaxis, :] + momentum_transfer

    rho_h_array = get_rho_h_3d(kh_array=kh_array,
                               k_array=k_array,
                               reciprocal_lattice=reciprocal_lattice,
                               z=z)
    epsilon_array = get_epsilon_3d(kh_array=kh_array,
                                   k_array=k_array,
                                   reciprocal_lattice=reciprocal_lattice)
    return {"kh_array": kh_array,
            "rho_h_array": rho_h_array,
            "epsilon_array": epsilon_array}


def derive_reflect_wavevec_and_gamma0_gammah_alpha(kh_array, k_array, reciprocal_lattice, z):
    rho_h_array = get_rho_h_3d(kh_array=kh_array,
                               k_array=k_array,
                               reciprocal_lattice=reciprocal_lattice,
                               z=z)

    epsilon_array = get_epsilon_3d(kh_array=kh_array,
                                   k_array=k_array,
                                   reciprocal_lattice=reciprocal_lattice)

    momentum_transfer = get_momentum_transfer_reverse(k_array=k_array,
                                                      rho_h_array=rho_h_array,
                                                      epsilon_array=epsilon_array)
    momentum_transfer = np.multiply(z[np.newaxis, :], momentum_transfer[:, np.newaxis])

    # get the incident wave vector
    k0_array = kh_array - reciprocal_lattice[np.newaxis, :] - momentum_transfer

    gamma_0_array = get_gamma_0_3d(k0_array=k0_array, k_array=k_array, z=z)
    gamma_h_array = get_gamma_h_3d(k0_array=k0_array, k_array=k_array, z=z,
                                   reciprocal_lattice=reciprocal_lattice)
    alpha_array = get_alpha_3d(k0_array=k0_array,
                               k_array=k_array,
                               reciprocal_lattice=reciprocal_lattice)
    return {"k0_array": k0_array,
            "gamma_0_array": gamma_0_array,
            "gamma_h_array": gamma_h_array,
            "alpha_array": alpha_array}


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


def get_phase(distance, k):
    """

    :param distance: array of shape [n]
    :param k: array of shape [
    :return:
    """
    return np.multiply(distance, k)


#################################
# Get bragg complex reflectivity
#################################
def generic_function_for_reflectivity(alpha_tidle, k_array, polar_factor,
                                      gamma_0_array, gamma_h_array, alpha_array,
                                      crystal_thickness, chih, chihbar):
    b_array = get_asymmetry_factor_batch(gamma_0_array=gamma_0_array,
                                         gamma_h_array=gamma_h_array)

    # Get sqrt(alpha**2 + beta**2) value
    sqrt_a2_b2, _ = sqrt_stable(np.square(alpha_tidle) +
                                np.multiply(np.square(polar_factor),
                                            b_array) * chih * chihbar)
    tmp_length = np.abs(sqrt_a2_b2)

    # There are three masks in total. Two masks are for phases
    # One mask is for the zero approximation
    mask_approx = np.zeros_like(alpha_array, dtype=np.bool)  # For zero approximation

    threshold = np.multiply(polar_factor, np.sqrt(np.abs(b_array) * np.abs(chih * chihbar)))
    threshold /= np.multiply(np.sqrt(k_array), np.sqrt(crystal_thickness))
    threshold *= 1e-6
    mask_approx[tmp_length <= threshold] = True
    if np.any(mask_approx):
        print("Approximation is used.")

    mask_analytic = np.logical_not(mask_approx)  # For analytical calculation

    # Set some auxiliary variables
    exp_phase = np.divide(np.multiply(k_array + 0.j, sqrt_a2_b2) * crystal_thickness, gamma_0_array)
    r0h = np.zeros_like(k_array, dtype=np.complex128)

    # For positive imaginary values
    numerator = 1. - exp_stable(1.j * exp_phase[mask_analytic])
    denominator = np.multiply(alpha_tidle[mask_analytic],
                              numerator) + np.multiply(sqrt_a2_b2[mask_analytic],
                                                       2 - numerator)

    r0h[mask_analytic] = np.multiply(np.multiply(b_array[mask_analytic], polar_factor[mask_analytic]) * chih,
                                     np.divide(numerator, denominator))

    # Calculate where one can only use approximation
    tmp5 = 1. / (alpha_tidle[mask_approx] +
                 (2.j / crystal_thickness) * np.divide(gamma_0_array[mask_approx], k_array[mask_approx]))

    holder = tmp5 * (1 + tmp5 * exp_phase[mask_approx] * sqrt_a2_b2[mask_approx] / 6.)
    r0h[mask_approx] = holder[:]
    return r0h


def get_bragg_reflection_finite_thickness(k_array, polar_factor,
                                          gamma_0_array, gamma_h_array, alpha_array,
                                          crystal_thickness, chi0, chih, chihbar):
    b_array = get_asymmetry_factor_batch(gamma_0_array=gamma_0_array,
                                         gamma_h_array=gamma_h_array)
    # Get alpha tidle
    alpha_tidle = (np.multiply(alpha_array, b_array) + chi0 * (1 - b_array)) / 2.

    ############################################
    # Step 1: Get reflection information for sigma polarization
    # Here, polar_factor=1
    ############################################

    # Get sqrt(alpha**2 + beta**2) value
    test = np.sqrt(np.square(alpha_tidle) +
                   np.multiply(np.square(polar_factor),
                               b_array) * chih * chihbar)

    holder = np.sign(test.imag)

    sqrt_a2_b2 = np.multiply(holder, test)

    tmp_length = np.abs(sqrt_a2_b2)

    # There are three masks in total. Two masks are for phases
    # One mask is for the zero approximation
    mask_approx = np.zeros_like(alpha_array, dtype=np.bool)  # For zero approximation

    threshold = np.multiply(polar_factor, np.sqrt(np.abs(b_array) * np.abs(chih * chihbar)))
    threshold /= np.multiply(np.sqrt(k_array), np.sqrt(crystal_thickness))
    threshold *= 1e-6
    mask_approx[tmp_length <= threshold] = True
    if np.any(mask_approx):
        print("Approximation is used.")

    mask_analytic = np.logical_not(mask_approx)  # For analytical calculation

    # Set some auxiliary variables
    exp_phase = np.divide(np.multiply(k_array + 0.j, sqrt_a2_b2) * crystal_thickness, gamma_0_array)
    r0h = np.zeros_like(k_array, dtype=np.complex128)

    # For positive imaginary values
    numerator = 1. - exp_stable(1.j * exp_phase[mask_analytic])
    denominator = np.multiply(alpha_tidle[mask_analytic],
                              numerator) + np.multiply(sqrt_a2_b2[mask_analytic],
                                                       2 - numerator)

    r0h[mask_analytic] = np.multiply(np.multiply(b_array[mask_analytic], polar_factor[mask_analytic]) * chih,
                                     np.divide(numerator, denominator))

    # Calculate where one can only use approximation
    tmp5 = 1. / (alpha_tidle[mask_approx] +
                 (2.j / crystal_thickness) * np.divide(gamma_0_array[mask_approx], k_array[mask_approx]))

    holder = tmp5 * (1 + tmp5 * exp_phase[mask_approx] * sqrt_a2_b2[mask_approx] / 6.)
    r0h[mask_approx] = holder[:]

    return r0h


#################################
# Get polarization info
#################################
def get_wavevector_in_and_out_crystal(k0_array, m_transfer_array, varkappa_array, z, reciprocal_lattice):
    k0_crystal = k0_array + np.outer(varkappa_array, z)
    kh_crystal = k0_array + np.reshape(reciprocal_lattice, (1, 3))
    kh_array = k0_array + m_transfer_array

    # Get the orientation of the polarization
    sigma_ori = np.cross(kh_crystal, k0_crystal)
    norm = l2_norm_batch(sigma_ori)
    sigma_ori /= norm[:, np.newaxis]

    pi_ori = np.cross(kh_crystal, sigma_ori)
    norm = l2_norm_batch(pi_ori)
    pi_ori /= norm[:, np.newaxis]

    return k0_crystal, kh_crystal, kh_array, sigma_ori, pi_ori


def get_epsilon_plus(alpha_array, b_array, polar_factor, chi0, chih, chihbar):
    alpha_tidle = (np.multiply(alpha_array, b_array) + chi0 * (1 - b_array)) / 2.
    return np.sqrt(np.square(alpha_tidle) + polar_factor ** 2 * chih * chihbar * b_array) - alpha_tidle


def get_epsilon_minus(alpha_array, b_array, polar_factor, chi0, chih, chihbar):
    alpha_tidle = (np.multiply(alpha_array, b_array) + chi0 * (1 - b_array)) / 2.
    return -np.sqrt(np.square(alpha_tidle) + polar_factor ** 2 * chih * chihbar * b_array) - alpha_tidle


def get_varkappa(epsilon_array, k_array, gamma_0_array):
    return np.divide(np.multiply(epsilon_array, k_array), gamma_0_array) / 2.


#########################################################################################
# For 2D diffraction
#########################################################################################

###########################################################
#       For Bragg Geometry
###########################################################


def phi0(theta, eta):
    """
    This is the phi0 variable defined in the paper. This is the angle between the direction of
    the incident wave vector and the inward normal direction.

    :param theta:
    :param eta:
    :return:
    """
    return theta + eta - pi / 2.


def phih(theta, eta):
    """
    The angle between the direction of the diffracted wave vector and the
    inward normal direction.
    :param theta:
    :param eta:
    :return:
    """
    return pi / 2. + theta - eta


def gamma0(theta, eta):
    """
    This is cos(phi0)

    :param theta:
    :param eta:
    :return:
    """
    return np.sin(eta + theta)


def gammah(theta, eta):
    """
    This is cos(phiH)

    :param theta:
    :param eta:
    :return:
    """
    return np.sin(eta - theta)


def asymmetry_factor(theta, eta):
    """
    This is the ratio between gamma0 and gammah.

    :param theta:
    :param eta:
    :return:
    """
    return np.divide(np.sin(eta + theta), np.sin(eta - theta))


# TODO: Here, the chih and chihbar is not treated properly.
def extinct_length(theta, eta, photon_wn, polar_factor, chih, chihbar):
    """
    The extinction length of this crystal at this geometry configuration

    :param theta:
    :param eta:
    :param photon_wn: The photon wave_number in fs^-1
    :param polar_factor: The polarization factor  |P|
    :param chih
    :param chihbar
    :return:
    """
    return (np.sqrt(np.sin(theta + eta) * np.abs(np.sin(eta - theta)) / (chihbar * chih))
            / (photon_wn * np.abs(polar_factor)))


def bragg_corr(theta, eta, chi0):
    """dffraction

    :param theta:
    :param eta:
    :param chi0:
    :return:
    """
    return 0.25 * (np.sin(eta - theta) / np.sin(eta + theta) - 1.) * chi0 / (np.sin(theta) ** 2)


def char_time(theta, eta, ext_length):
    """

    :param theta:
    :param eta:
    :param ext_length:
    :return:
    """
    return 2 * ext_length * (np.sin(theta) ** 2) / c / np.abs(gammah(theta=theta, eta=eta))


def capital_a(d, ext_length):
    """
    This is some intermediate result in the paper I have no idea how to interpret.

    :param d:
    :param ext_length: extinction length
    :return:
    """
    return d / ext_length


@jit([complex128[:](float64[:], float64[:], float64, complex128)])
def capital_c(photon_wn, gamma_0_val, thickness, chi0):
    """
    This is the value of R00 when y goes to infinity
    :param photon_wn:
    :param gamma_0_val:
    :param thickness:
    :param chi0:
    :return:
    """
    tmp = 0.5 * thickness * np.divide(photon_wn, gamma_0_val)
    holder = np.exp(- chi0.imag * tmp)
    holder = np.multiply(holder,
                         np.cos(chi0.real * tmp) + 1.j * np.sin(chi0.real * tmp))
    return holder


def capital_g(theta, eta, chih, chihb):
    """

    :param theta:
    :param eta:
    :param chih:
    :param chihb:
    :return:
    """
    return np.sqrt(np.abs(asymmetry_factor(theta=theta, eta=eta)) * chih * chihb) / chihb


def get_alpha(lattice_wave_number, photon_wave_number, theta):
    """

    :param lattice_wave_number:
    :param photon_wave_number:
    :param theta:
    :return:
    """
    return (-2 * lattice_wave_number * photon_wave_number * np.sin(theta) +
            lattice_wave_number ** 2) / photon_wave_number ** 2


@jit([complex128(float64, float64, complex128, float64, float64, complex128)], nopython=True)
def get_y(photon_wn, theta, extinction_length, lattice_wn, eta, chi0):
    """

    :param photon_wn:
    :param lattice_wn:
    :param theta:
    :param eta:
    :param chi0:
    :param extinction_length:
    :return:
    """
    alpha_val = complex((-2 * lattice_wn * photon_wn * np.sin(theta) +
                         lattice_wn ** 2) / photon_wn ** 2)

    b_val = complex(np.sin(eta + theta) / np.sin(eta - theta))
    gamma0_val = complex(np.sin(eta + theta))

    return photon_wn * extinction_length / 2. / gamma0_val * (
            b_val * alpha_val + chi0 * (1 - b_val))


@jit([complex128[:](float64[:],
                    float64[:],
                    complex128[:],
                    float64,
                    float64,
                    complex128,
                    )], nopython=True)
def get_y_array(photon_wn, theta, extinction_length, lattice_wn, eta, chi0):
    """
    Get a list of y values.

    Here, one assumes that lattice wave_number, eta, chi0 are the same for the list.

    :param photon_wn: The shape has to be (n,)
    :param lattice_wn:  The shape has to be (n,)
    :param theta: The shape has to be (n,)
    :param eta:
    :param chi0:
    :param extinction_length:
    :return:
    """

    # Step 1: Create a holder for the value
    y_holder = np.ones(photon_wn.shape[0],
                       dtype=np.complex128)

    # Step 2
    alpha_val = np.divide(-2 * lattice_wn *
                          np.multiply(photon_wn,
                                      np.sin(theta)) +
                          np.square(lattice_wn),
                          np.square(photon_wn)
                          )

    gamma0_val = np.sin(eta + theta)

    b_val = np.divide(gamma0_val, np.sin(eta - theta))

    # Step 3: Use the definition
    y_holder[:] = np.divide(np.multiply(photon_wn, extinction_length),
                            2. * gamma0_val)
    y_holder[:] = np.multiply(y_holder,
                              np.multiply(b_val, alpha_val) + chi0 * (1 - b_val))

    return y_holder


def get_bragg_r0h(y_val, cap_g_val, cap_a_val):
    """

    :param y_val:
    :param cap_g_val:
    :param cap_a_val:
    :return:
    """
    val_1 = np.sqrt(y_val ** 2 - 1)

    # Get the exponential minus 1 term
    val_2 = (np.exp(-cap_a_val.real * val_1.imag - cap_a_val.imag * val_1.real) *
             (np.cos(cap_a_val.real * val_1.real - cap_a_val.imag * val_1.imag) +
              1.j * np.sin(cap_a_val.real * val_1.real - cap_a_val.imag * val_1.imag))) - 1

    val = np.divide(np.multiply(cap_g_val, val_2),
                    (val_1 * (val_2 + 2) - y_val * val_2))

    return val


def get_bragg_r00(y_val, cap_a_val, photon_wn, gamma_0, d, chi0):
    """

    :param y_val:
    :param cap_a_val:
    :param d:
    :param photon_wn:
    :param chi0:
    :param gamma_0:
    :return:
    """

    val_1 = np.sqrt(y_val ** 2 - 1)

    # Get the exponential minus 1 term
    val_2 = (np.exp(-cap_a_val.real * val_1.imag - cap_a_val.imag * val_1.real) *
             (np.cos(cap_a_val.real * val_1.real - cap_a_val.imag * val_1.imag) +
              1.j * np.sin(cap_a_val.real * val_1.real - cap_a_val.imag * val_1.imag))) - 1

    # Get the remainding phase factor
    val_3 = np.exp(-0.5 * (d * photon_wn / gamma_0 * chi0.imag -
                           cap_a_val.real * (y_val.imag - val_1.imag) -
                           cap_a_val.imag * (y_val.real - val_1.real)))

    tmp = 0.5 * (d * photon_wn / gamma_0 * chi0.real -
                 cap_a_val.real * (y_val.real - val_1.real) +
                 cap_a_val.imag * (y_val.imag - val_1.imag))

    val_3 = np.multiply(val_3, np.cos(tmp) + 1.j * np.sin(tmp))

    # Assemble everything
    val = 2 * val_3 * val_1 / (val_1 * (val_2 + 2) - y_val * val_2)

    return val


def get_laue_r00(y_val, cap_a_val, photon_wn, gamma_0, d, chi0):
    """

    :param y_val:
    :param cap_a_val:
    :param photon_wn:
    :param gamma_0:
    :param d:
    :param chi0:
    :return:
    """

    # The square root term
    val_1 = np.sqrt(y_val ** 2 + 1)

    # Get the exponential minus 1 term
    val_2 = (np.exp(-cap_a_val.real * val_1.imag - cap_a_val.imag * val_1.real) *
             (np.cos(cap_a_val.real * val_1.real - cap_a_val.imag * val_1.imag) +
              1.j * np.sin(cap_a_val.real * val_1.real - cap_a_val.imag * val_1.imag))) - 1

    # Get the remainding phase factor
    val_3 = np.exp(-0.5 * (d * photon_wn / gamma_0 * chi0.imag -
                           cap_a_val.real * (y_val.imag + val_1.imag) -
                           cap_a_val.imag * (y_val.real + val_1.real)))

    tmp = 0.5 * (d * photon_wn / gamma_0 * chi0.real -
                 cap_a_val.real * (y_val.real + val_1.real) +
                 cap_a_val.imag * (y_val.imag + val_1.imag))

    val_3 = np.multiply(val_3, np.cos(tmp) + 1.j * np.sin(tmp))

    # Assemble every thing
    val = np.divide(np.multiply(val_3,
                                val_1 * (val_2 + 2) + y_val * val_2),
                    2 * val_1)

    return val


def get_laue_r0h(y_val, cap_g_val, cap_a_val, photon_wn, gamma_0, d, chi0):
    """

    :param y_val:
    :param cap_g_val:
    :param cap_a_val:
    :param d:
    :param photon_wn:
    :param chi0:
    :param gamma_0:
    :return:
    """

    # The square root term
    val_1 = np.sqrt(y_val ** 2 + 1)

    # Get the exponential minus 1 term times g
    val_2 = np.exp(-cap_a_val * val_1.imag) * (np.cos(cap_a_val * val_1.real) +
                                               1.j * np.sin(cap_a_val * val_1.real)) - 1
    val_2 = np.multiply(val_2, cap_g_val)

    # Get the remainding phase factor
    val_3 = np.exp(-0.5 * (d * photon_wn / gamma_0 * chi0.imag -
                           cap_a_val.real * (y_val.imag + val_1.imag) -
                           cap_a_val.imag * (y_val.real + val_1.real)))

    tmp = 0.5 * (d * photon_wn / gamma_0 * chi0.real -
                 cap_a_val.real * (y_val.real + val_1.real) +
                 cap_a_val.imag * (y_val.imag + val_1.imag))

    val_3 = np.multiply(val_3, np.cos(tmp) + 1.j * np.sin(tmp))

    # Assemble every thing
    val = np.divide(np.multiply(val_3, val_2), 2 * val_1)

    return val


############################
# Quantities conversion
############################
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


############################
# Time response related
############################
def get_time_response(spectral_value_array, spectrum_array, verbose=False):
    """
    Convert the response in frequency to response in time
    :param spectral_value_array:
    :param spectrum_array:
    :param verbose:
    :return:
    """
    # Convert the corresponding frequency index to time index
    number = spectral_value_array.shape[0]
    domega = spectrum_array[1] - spectrum_array[0]
    t_range = pi * 2 / domega
    time_idx = np.linspace(start=0, stop=t_range, num=number)

    if verbose:
        print("The total time range is {:.2f} fs. ".format(t_range) +
              "Each pixel represent {:.2e} fs".format(t_range / number))

    # Get the response function by applying the fft and some scaling factor.
    response = np.fft.fft(spectral_value_array) * domega * 2 * pi

    return {"Response": response,
            "Time idx": time_idx}


def capital_d(theta, eta):
    """
    This is the normalized angular dispersion rate

    :param theta:
    :param eta:
    :return:
    """

    return - np.multiply(1 + asymmetry_factor(theta=theta, eta=eta), np.tan(theta))


def get_script_cap_t_d(d, theta, eta):
    """
    This is a characteristic measure of time in Bragg diffraction associated withthe crystal
    thickness d. In the Laue-case geometry, T d is equal to the total duration of forward
    Bragg diffraction

    :param d:
    :param theta:
    :param eta:
    :return:
    """
    return 2. * d / c * np.divide(np.square(np.sin(theta)), gammah(theta, eta))


def get_xi_0(time_val, r_vals, u_0):
    """

    :param time_val:
    :param r_vals:
    :param u_0:
    :return:
    """
    return time_val - np.dot(r_vals, u_0) / c


def get_xi_h(tau_h_vals, theta, eta, normal_distance):
    """
    This is the phase term of the reflected pulse
    :param tau_h_vals:

    :param theta:
    :param eta:
    :param normal_distance:
    :return:
    """
    val = tau_h_vals + 2. / c * np.divide(np.multiply(np.square(np.sin(theta)),
                                                      normal_distance),
                                          gammah(theta=theta, eta=eta))
    return val


def get_tau(time_val, r_vals, u):
    """

    :param time_val:
    :param r_vals:
    :param u:
    :return:
    """
    return time_val - np.dot(r_vals, u) / c


def get_lognormal(mean=0.0, sigma=1.0, size=None):
    """
    This function return an array with the size specified in the arguments.
    This array obeys lognormal distribtion. The mean value is the mean value specified here.
    Notice that here, the mean value is the mean value of the lognormal distribution rather
    than the mean value of the corresponding normal distribution.

    The same holds for the sigma.

    :param mean:
    :param sigma:
    :param size:
    :return:
    """
    # First get the correct mean and sigma
    tmp_mean = np.log(mean ** 2 / np.sqrt(mean ** 2 + sigma ** 2))
    tmp_sigma = np.sqrt(np.log(1 + np.square(sigma / mean)))
    return np.random.lognormal(mean=tmp_mean, sigma=tmp_sigma, size=size)


def get_rot_mat_dict_lcls_2(kin, kout, h_vec, aux=np.array([1., 0., 0.])):
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
    :param h_vec:
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


def get_rot_mat_dict(kin, kout, h_vec):
    """
    Given the incident central momentum, the diffracted central momentum
    and the reciprocal lattice, decide the rotation matrices such that
    1. From the incident pulse frame to the device frame
    2. From the device frame to the diffracted pulse frame
    3. From the incident pulse frame to the diffracted pulse frame

    For different frames, kin and h span x-y plane with kin being the x axis
    and h with positive y component.
    kout and h span x-y plane with kout being the x axis and b with positive
    y component.

    :param kin:
    :param kout:
    :param h_vec:
    :return:
    """
    # Get a holder
    rot_mat_dict = {}

    # -------------------------------------------------
    # Device to the incident pulse
    # -------------------------------------------------
    tmp_x = kin / l2_norm(kin)

    tmp_z = np.cross(tmp_x, h_vec)
    tmp_z /= l2_norm(tmp_z)

    tmp_y = np.cross(tmp_z, tmp_x)
    tmp_y /= l2_norm(tmp_y)

    # Rotation matrix from the incident pulse to the device
    rot_mat_dict.update({"Device to In-Pulse": np.vstack([tmp_x, tmp_y, tmp_z])})
    rot_mat_dict.update({"In-Pulse to Device": rot_mat_dict["Device to In-Pulse"].T})

    # -------------------------------------------------
    # Device to the output pulse
    # -------------------------------------------------
    tmp_x = kout / l2_norm(kout)

    tmp_z = np.cross(kin, tmp_x)
    tmp_z /= l2_norm(tmp_z)

    tmp_y = np.cross(tmp_z, tmp_x)
    tmp_y /= l2_norm(tmp_y)

    # Rotation matrix from the incident pulse to the device
    rot_mat_dict.update({"Device to Out-Pulse": np.vstack([tmp_x, tmp_y, tmp_z])})
    rot_mat_dict.update({"Out-Pulse to Device": rot_mat_dict["Device to Out-Pulse"].T})

    # -------------------------------------------------
    # Incident pulse to the output pulse
    # -------------------------------------------------

    tmp = np.dot(rot_mat_dict["In-Pulse to Device"],
                 rot_mat_dict["Device to Out-Pulse"])

    rot_mat_dict.update({"In-Pulse to Out-Pulse": tmp})
    rot_mat_dict.update({"Out-Pulse to In-Pulse": tmp.T})

    return rot_mat_dict


def get_rot_mat_dict_bk_2(kin, kout, h_vec):
    """
    Given the incident central momentum, the diffracted central momentum
    and the reciprocal lattice, decide the rotation matrices such that
    1. From the incident pulse frame to the device frame
    2. From the device frame to the diffracted pulse frame
    3. From the incident pulse frame to the diffracted pulse frame

    For different frames, kin and h span x-y plane with kin being the x axis
    and h with positive y component.
    kout and h span x-y plane with kout being the x axis and b with positive
    y component.

    :param kin:
    :param kout:
    :param h_vec:
    :return:
    """
    # Get a holder
    rot_mat_dict = {}

    # -------------------------------------------------
    # Device to the incident pulse
    # -------------------------------------------------
    tmp_x = kin / l2_norm(kin)

    tmp_z = np.cross(tmp_x, h_vec)
    tmp_z /= l2_norm(tmp_z)

    tmp_y = np.cross(tmp_z, tmp_x)
    tmp_y /= l2_norm(tmp_y)

    # Rotation matrix from the incident pulse to the device
    rot_mat_dict.update({"Device to In-Pulse": np.vstack([tmp_x, tmp_y, tmp_z])})
    rot_mat_dict.update({"In-Pulse to Device": rot_mat_dict["Device to In-Pulse"].T})

    # -------------------------------------------------
    # Device to the output pulse
    # -------------------------------------------------
    tmp_x = kout / l2_norm(kout)

    tmp_z = np.cross(tmp_x, h_vec)
    tmp_z /= l2_norm(tmp_z)

    tmp_y = np.cross(tmp_z, tmp_x)
    tmp_y /= l2_norm(tmp_y)

    # Rotation matrix from the incident pulse to the device
    rot_mat_dict.update({"Device to Out-Pulse": np.vstack([tmp_x, tmp_y, tmp_z])})
    rot_mat_dict.update({"Out-Pulse to Device": rot_mat_dict["Device to Out-Pulse"].T})

    # -------------------------------------------------
    # Incident pulse to the output pulse
    # -------------------------------------------------

    tmp = np.dot(rot_mat_dict["In-Pulse to Device"],
                 rot_mat_dict["Device to Out-Pulse"])

    rot_mat_dict.update({"In-Pulse to Out-Pulse": tmp})
    rot_mat_dict.update({"Out-Pulse to In-Pulse": tmp.T})

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


########################################################################################
#     Verify the simulation with numpy calculation
########################################################################################
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


def get_square_pulse_spectrum_1d(k_grid, k0, a_val, b_val, c_val, scaling):
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


def get_phase_1d(k_grid, omega_grid, dx, dt):
    pre_phase = np.dot(k_grid, dx) - omega_grid * dt
    phase = np.cos(pre_phase) + 1.j * np.sin(pre_phase)
    return pre_phase, phase


# ---------------------------------------------------------------------------
#                     Post-process
# ---------------------------------------------------------------------------
def get_2d_sigma_matrix(density_2d, x_values, y_values):
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
