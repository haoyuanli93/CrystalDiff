import cmath
import numpy as np
import math
from numba import cuda

"""
                Format requirement
    For all the cuda functions, here is the format of the argument
    1. All the functions have void output
    2. The first arguments are the variables holding the output
    3. Following the output variables are the input variables. For the input variables, 
        1. First specify the input k vector or k length
        2. Then specify the k components 
        3. Then specify the crystal related properties
    4. The last argument is the number of scalars or vectors to calculate 
    
"""

c = 299792458. * 1e-9  # The speed of light in um / fs
two_pi = 2 * math.pi
eps = np.finfo(np.float64).eps


def l2_norm(x):
    return np.sqrt(np.sum(np.square(x)))


###################################################################################################
#
#          Initialization
#
###################################################################################################
@cuda.jit("void(float64[:,:], float64[:], float64[:], float64[:], float64, float64, float64, int64)")
def initialize_kvec_grid(kvec_grid, klen_grid, kx_grid, kx_square_grid, ky, kz, square_ky_kz, num):
    """

    :param kvec_grid: The grid of wave vectors
    :param klen_grid: The grid of the length of the wave vectors
    :param kx_grid:
    :param kx_square_grid:
    :param kz:
    :param ky:
    :param square_ky_kz:
    :param num:
    :return:
    """
    idx = cuda.grid(1)
    if idx < num:
        kvec_grid[idx, 0] = kx_grid[idx]
        kvec_grid[idx, 1] = ky
        kvec_grid[idx, 2] = kz

        klen_grid[idx] = math.sqrt(kx_square_grid[idx] + square_ky_kz)


@cuda.jit("void(float64[:,:], float64[:], float64[:], float64[:], float64, float64, float64, int64)")
def initialize_kvec_grid_lcls2(kvec_grid, klen_grid, kz_grid, kz_square_grid, kx, ky, square_ky_kz, num):
    """

    :param kvec_grid: The grid of wave vectors
    :param klen_grid: The grid of the length of the wave vectors
    :param kz_grid:
    :param kz_square_grid:
    :param kx:
    :param ky:
    :param square_ky_kz:
    :param num:
    :return:
    """
    idx = cuda.grid(1)
    if idx < num:
        kvec_grid[idx, 0] = kx
        kvec_grid[idx, 1] = ky
        kvec_grid[idx, 2] = kz_grid[idx]

        klen_grid[idx] = math.sqrt(kz_square_grid[idx] + square_ky_kz)


@cuda.jit("void(float64[:,:], float64[:], float64[:], float64[:], float64[:], int64)")
def initialize_kvec_grid_dumond_bk(kvec_grid, klen_grid, x_coef, y_coef, z_coef, num):
    """

    :param kvec_grid:
    :param klen_grid:
    :param x_coef:
    :param y_coef:
    :param z_coef:
    :param num:
    :return:
    """
    idx = cuda.grid(1)
    if idx < num:
        kvec_grid[idx, 0] = klen_grid[idx] * x_coef[idx]
        kvec_grid[idx, 1] = klen_grid[idx] * y_coef[idx]
        kvec_grid[idx, 2] = klen_grid[idx] * z_coef[idx]


@cuda.jit("void(float64[:,:], float64[:], float64, float64, float64, int64)")
def initialize_kvec_grid_dumond(kvec_grid, klen_grid, x_coef, y_coef, z_coef, num):
    """

    :param kvec_grid:
    :param klen_grid:
    :param x_coef:
    :param y_coef:
    :param z_coef:
    :param num:
    :return:
    """
    idx = cuda.grid(1)
    if idx < num:
        kvec_grid[idx, 0] = klen_grid[idx] * x_coef
        kvec_grid[idx, 1] = klen_grid[idx] * y_coef
        kvec_grid[idx, 2] = klen_grid[idx] * z_coef


@cuda.jit("void(float64[:], int64)")
def initialize_jacobian_grid(jacobian, num):
    """

    :param jacobian:
    :param num:
    :return:
    """

    idx = cuda.grid(1)
    if idx < num:
        jacobian[idx] = 1.


@cuda.jit("void(complex128[:], int64)")
def initialize_phase_holder(phase, num):
    idx = cuda.grid(1)
    if idx < num:
        phase[idx] = complex(1., 0.)


@cuda.jit("void(float64[:], float64, int64)")
def initialize_path_len(path_len, length, num):
    idx = cuda.grid(1)
    if idx < num:
        path_len[idx] = length


###################################################################################################
#
#          Elementwise operation
#
###################################################################################################
@cuda.jit("void(complex128[:], complex128[:], complex128[:], int64)")
def element_wise_multiply_complex(a, b, out, num):
    idx = cuda.grid(1)
    if idx < num:
        out[idx] = a[idx] * b[idx]


@cuda.jit("void(complex128[:], complex128[:], complex128[:,:], int64)")
def expand_scalar_grid_to_vector_grid(scalar_grid, vec, vec_grid, num):
    """

    :param scalar_grid:
    :param vec:
    :param vec_grid:
    :param num:
    :return:
    """
    idx = cuda.grid(1)
    if idx < num:
        vec_grid[idx, 0] = scalar_grid[idx] * vec[0]
        vec_grid[idx, 1] = scalar_grid[idx] * vec[1]
        vec_grid[idx, 2] = scalar_grid[idx] * vec[2]


@cuda.jit("void(complex128[:,:], complex128[:], complex128[:], complex128[:], int64)")
def vector_decomposition(vector, x, y, z, num):
    """

    :param vector:
    :param x:
    :param y:
    :param z:
    :param num:
    :return:
    """
    idx = cuda.grid(1)
    if idx < num:
        x[idx] = vector[idx, 0]
        y[idx] = vector[idx, 1]
        z[idx] = vector[idx, 2]


###################################################################################################
#
#          Data transfer
#
###################################################################################################
@cuda.jit("void(complex128[:,:], complex128[:], int64, int64, int64, int64, int64)")
def fill_column_complex_fftshift(holder, source, row_idx, idx_start1, num1, idx_start2, num2):
    """

    :param holder:
    :param source:
    :param row_idx:
    :param idx_start1:
    :param num1:
    :param idx_start2:
    :param num2:
    :return:
    """
    col_idx = cuda.grid(1)
    if col_idx < num1:
        holder[row_idx, col_idx] = source[idx_start1 + col_idx]
    if col_idx < num2:
        holder[row_idx, num1 + col_idx] = source[idx_start2 + col_idx]


@cuda.jit("void(complex128[:,:], complex128[:], int64, int64, int64)")
def fill_column_complex(holder, source, idx_start, row_idx, num):
    """
    This is a auxiliary function.

    Holder is the final function I would like to use to save the
    spatial-temporal function. I would like to fill in this variable
    with values from the source which would be a 1d response function.

    :param source: The source response function from which the data will be copied
    :param holder: The holder array to store the copied data
    :param idx_start: The first idx from which the data will be copied
    :param row_idx: The row idx of the holder variable to store the information
    :param num: The total number of data to store for each line.
    :return:
    """
    column = cuda.grid(1)
    if column < num:
        holder[row_idx, column] = source[column + idx_start]


@cuda.jit("void(complex128[:,:], complex128[:], int64, int64, int64, int64)")
def fill_column_complex_ds(holder, source, idx_start, ds_size, row_idx, num):
    """
    This is a auxiliary function.

    Holder is the final function I would like to use to save the
    spatial-temporal function. I would like to fill in this variable
    with values from the source which would be a 1d response function.

    :param source: The source response function from which the data will be copied
    :param holder: The holder array to store the copied data
    :param idx_start: The first idx from which the data will be copied
    :param ds_size: The number of values to average.
    :param row_idx: The row idx of the holder variable to store the information
    :param num: The total number of data to store for each line.
    :return:
    """
    column = cuda.grid(1)
    if column < num:
        init = column * ds_size + idx_start
        for l in range(ds_size):
            holder[row_idx, column] += source[init + l]


@cuda.jit("void(float64[:,:], float64[:], int64, int64, int64)")
def fill_column_float(holder, source, idx_start, row_idx, num):
    """
    This is a auxiliary function.

    Holder is the final function I would like to use to save the
    spatial-temporal function. I would like to fill in this variable
    with values from the source which would be a 1d response function.

    :param source: The source response function from which the data will be copied
    :param holder: The holder array to store the copied data
    :param idx_start: The first idx from which the data will be copied
    :param row_idx: The row idx of the holder variable to store the information
    :param num: The total number of data to store for each line.
    :return:
    """
    column = cuda.grid(1)
    if column < num:
        holder[row_idx, column] = source[column + idx_start]


@cuda.jit("void(float64[:,:], float64[:], int64, int64, int64, int64)")
def fill_column_float_ds(holder, source, idx_start, ds_size, row_idx, num):
    """
    This is a auxiliary function.

    Holder is the final function I would like to use to save the
    spatial-temporal function. I would like to fill in this variable
    with values from the source which would be a 1d response function.

    :param source: The source response function from which the data will be copied
    :param holder: The holder array to store the copied data
    :param idx_start: The first idx from which the data will be copied
    :param ds_size
    :param row_idx: The row idx of the holder variable to store the information
    :param num: The total number of data to store for each line.
    :return:
    """
    column = cuda.grid(1)
    if column < num:
        init = column * ds_size + idx_start
        for l in range(ds_size):
            holder[row_idx, column] += source[init + l]


###################################################################################################
#
#          Positive direction
# Below are functions that are used to calculate the reflectivity when we know the
# incident pulse infomation.
#
###################################################################################################
@cuda.jit('void(complex128[:], complex128[:], float64[:,:], complex128[:,:],'
          ' float64[:], float64[:,:],'
          ' float64, float64[:], float64[:],'
          ' float64, float64, float64,'
          ' complex128, complex128, complex128, complex128, complex128, int64)')
def get_bragg_field_natural_direction(reflectivity_sigma, reflectivity_pi, kout_grid, efield_grid,
                                      klen_grid, kin_grid,
                                      d, h, n,
                                      dot_hn, h_square, h_len,
                                      chi0, chih_sigma, chihbar_sigma,
                                      chih_pi, chihbar_pi,
                                      num):
    """
    Given the crystal info, the input electric field, this function returns the
    reflectivity for the sigma polarization and pi polarization and the
    diffracted electric field.

    :param reflectivity_sigma:
    :param reflectivity_pi:
    :param kout_grid:
    :param efield_grid:
    :param klen_grid:
    :param kin_grid:
    :param d:
    :param h:
    :param n:
    :param dot_hn:
    :param h_square:
    :param h_len:
    :param chi0:
    :param chih_sigma:
    :param chihbar_sigma:
    :param chih_pi:
    :param chihbar_pi:
    :param num:
    :return:
    """
    idx = cuda.grid(1)
    if idx < num:

        #####################################################################################################
        # Step 1: Get parameters for reflectivity and decompose input field
        #####################################################################################################
        # Get k components
        kin_x = kin_grid[idx, 0]
        kin_y = kin_grid[idx, 1]
        kin_z = kin_grid[idx, 2]

        k = klen_grid[idx]
        k_square = k ** 2

        # Get gamma and alpha and b
        dot_kn = kin_x * n[0] + kin_y * n[1] + kin_z * n[2]
        dot_kh = kin_x * h[0] + kin_y * h[1] + kin_z * h[2]

        gamma_0 = dot_kn / k
        gamma_h = (dot_kn + dot_hn) / k
        b = gamma_0 / gamma_h
        alpha = (2 * dot_kh + h_square) / k_square

        # Get momentum tranfer
        sqrt_gamma_alpha = math.sqrt(gamma_h ** 2 - alpha)
        tmp_pos = abs(-gamma_h + sqrt_gamma_alpha)
        tmp_neg = abs(-gamma_h - sqrt_gamma_alpha)
        if tmp_pos > tmp_neg:
            m_trans = k * (-gamma_h - sqrt_gamma_alpha)
        else:
            m_trans = k * (-gamma_h + sqrt_gamma_alpha)

        # Get output wave vector
        kout_x = kin_x + h[0] + m_trans * n[0]
        kout_y = kin_y + h[1] + m_trans * n[1]
        kout_z = kin_z + h[2] + m_trans * n[2]

        # Update the kout_grid
        kout_grid[idx, 0] = kout_x
        kout_grid[idx, 1] = kout_y
        kout_grid[idx, 2] = kout_z

        # ----------------------------------------
        #          Get polarization direction
        # ----------------------------------------
        """
        sigma_in = kH cross k0     sigma_out = sigma_in  
        pi_in = k0 cross sigma0    pi_out = k_out cross sigma_out        
        """
        sigma_in_x = kout_y * kin_z - kout_z * kin_y
        sigma_in_y = kout_z * kin_x - kout_x * kin_z
        sigma_in_z = kout_x * kin_y - kout_y * kin_x
        tmp_len = math.sqrt(sigma_in_x ** 2 + sigma_in_y ** 2 + sigma_in_z ** 2)
        sigma_in_x /= tmp_len
        sigma_in_y /= tmp_len
        sigma_in_z /= tmp_len

        pi_in_x = kin_y * sigma_in_z - kin_z * sigma_in_y
        pi_in_y = kin_z * sigma_in_x - kin_x * sigma_in_z
        pi_in_z = kin_x * sigma_in_y - kin_y * sigma_in_x
        tmp_len = math.sqrt(pi_in_x ** 2 + pi_in_y ** 2 + pi_in_z ** 2)
        pi_in_x /= tmp_len
        pi_in_y /= tmp_len
        pi_in_z /= tmp_len

        pi_out_x = kout_y * sigma_in_z - kout_z * sigma_in_y
        pi_out_y = kout_z * sigma_in_x - kout_x * sigma_in_z
        pi_out_z = kout_x * sigma_in_y - kout_y * sigma_in_x
        tmp_len = math.sqrt(pi_out_x ** 2 + pi_out_y ** 2 + pi_out_z ** 2)
        pi_out_x /= tmp_len
        pi_out_y /= tmp_len
        pi_out_z /= tmp_len

        # Decompose the input electric field
        efield_sigma = (efield_grid[idx, 0] * complex(sigma_in_x, 0.) +
                        efield_grid[idx, 1] * complex(sigma_in_y, 0.) +
                        efield_grid[idx, 2] * complex(sigma_in_z, 0.))
        efield_pi = (efield_grid[idx, 0] * complex(pi_in_x, 0.) +
                     efield_grid[idx, 1] * complex(pi_in_y, 0.) +
                     efield_grid[idx, 2] * complex(pi_in_z, 0.))

        #####################################################################################################
        # Step 2: Get the reflectivity for input sigma polarization
        #####################################################################################################
        # ----------------------------------------
        #    Get reflectivity
        # ----------------------------------------
        # Get alpha tidle
        alpha_tidle = complex((alpha * b + chi0.real * (1. - b)) / 2., chi0.imag * (1. - b) / 2.)

        # Get sqrt(alpha**2 + beta**2) value
        sqrt_a2_b2 = cmath.sqrt(alpha_tidle ** 2 +
                                complex(b, 0.) * chih_sigma * chihbar_sigma)

        if sqrt_a2_b2.imag < 0:
            sqrt_a2_b2 = - sqrt_a2_b2

        # Calculate the phase term
        re = k * d / gamma_0 * sqrt_a2_b2.real
        im = k * d / gamma_0 * sqrt_a2_b2.imag

        # Take care of the exponential
        if im <= 100.:
            magnitude = complex(math.exp(-im), 0.)

            phase = complex(math.cos(re), math.sin(re))
            # Calculate some intermediate part
            numerator = 1. - magnitude * phase
            denominator = alpha_tidle * numerator + sqrt_a2_b2 * (2. - numerator)

            # Assemble everything
            reflectivity_sigma[idx] = complex(b, 0.) * chih_sigma * numerator / denominator

        else:
            # When the crystal is super thick, the numerator becomes 1 The exponential term becomes 0.
            # Calculate some intermediate part
            denominator = alpha_tidle + sqrt_a2_b2

            # Assemble everything
            reflectivity_sigma[idx] = complex(b, 0.) * chih_sigma / denominator

        # Get the output electric field due to this component
        efield_out_sigma_x = reflectivity_sigma[idx] * efield_sigma * complex(sigma_in_x, 0.)
        efield_out_sigma_y = reflectivity_sigma[idx] * efield_sigma * complex(sigma_in_y, 0.)
        efield_out_sigma_z = reflectivity_sigma[idx] * efield_sigma * complex(sigma_in_z, 0.)

        #####################################################################################################
        # Step 3: Get the reflectivity for input pi polarization
        #####################################################################################################
        # ----------------------------------------
        #    Get reflectivity
        # ----------------------------------------
        sin_theta = (dot_kh - h_square - m_trans * dot_hn) / k / h_len
        polar_factor = (1 - 2 * sin_theta ** 2) ** 2

        # Get sqrt(alpha**2 + beta**2) value
        sqrt_a2_b2 = cmath.sqrt(alpha_tidle ** 2 +
                                complex(polar_factor * b, 0.) * chih_pi * chihbar_pi)

        # Because this is a thick crystal, only one mode will be activated.
        if sqrt_a2_b2.imag < 0:
            # Because only one mode is activated,
            sqrt_a2_b2 = - sqrt_a2_b2

        # Calculate the phase term
        re = k * d / gamma_0 * sqrt_a2_b2.real
        im = k * d / gamma_0 * sqrt_a2_b2.imag

        # Take care of the exponential
        if im <= 100.:
            magnitude = complex(math.exp(-im), 0.)
            phase = complex(math.cos(re), math.sin(re))

            # Calculate some intermediate part
            numerator = 1. - magnitude * phase
            denominator = alpha_tidle * numerator + sqrt_a2_b2 * (2. - numerator)
            # Assemble everything
            reflectivity_pi[idx] = complex(b * polar_factor, 0.) * chih_pi * numerator / denominator

        else:
            # Calculate some intermediate part
            denominator = alpha_tidle + sqrt_a2_b2
            # Assemble everything
            reflectivity_pi[idx] = complex(b * polar_factor, 0.) * chih_pi / denominator

        # Get the output electric field due to this component
        efield_out_pi_x = reflectivity_pi[idx] * efield_pi * complex(pi_out_x, 0.)
        efield_out_pi_y = reflectivity_pi[idx] * efield_pi * complex(pi_out_y, 0.)
        efield_out_pi_z = reflectivity_pi[idx] * efield_pi * complex(pi_out_z, 0.)

        #####################################################################################################
        # Step 4: Assemble to get the output electric field
        #####################################################################################################
        efield_grid[idx, 0] = efield_out_sigma_x + efield_out_pi_x
        efield_grid[idx, 1] = efield_out_sigma_y + efield_out_pi_y
        efield_grid[idx, 2] = efield_out_sigma_z + efield_out_pi_z


###################################################################################################
#
#          Negative direction
# Below are functions that are used to calculate the reflectivity when we know the
# reflected pulse infomation.
#
###################################################################################################
@cuda.jit('void'
          '(float64[:,:],'
          'float64[:], float64[:,:], '
          'float64[:], float64[:], float64, float64, '
          'int64)')
def get_kin_grid(kin_grid,
                 klen_grid, kout_grid,
                 h, n, dot_hn, h_square,
                 num):
    """
    Given kout info, this function derives the corresponding kin info.

    :param kin_grid: This function derive the input wave vectors
    :param klen_grid: The wave vector length
    :param kout_grid:
    :param h: The crystal h vector
    :param n: The crystal normal direction
    :param dot_hn: The inner product between h and n
    :param h_square: The length of the h vector
    :param num:
    :return:
    """
    # Step 0: Get the cuda grid idx
    idx = cuda.grid(1)
    if idx < num:

        ##################################################################
        # Step 1: Get the corresponding parameters to get the reflectivity
        ##################################################################
        # Get k
        ku = kout_grid[idx, 0]
        kv = kout_grid[idx, 2]
        kw = kout_grid[idx, 1]
        kvwn = kv * n[2] + kw * n[1]
        kvwh = kv * h[2] + kw * h[1]

        k = klen_grid[idx]
        k_square = k ** 2

        # Get rho and epsilon
        dot_kn = kvwn + ku * n[0]
        dot_kh = kvwh + ku * h[0]
        rho = (dot_hn - dot_kn) / k
        epsilon = (h_square - 2 * dot_kh) / k_square

        # Decide the sign
        sqrt_rho_epsilon = math.sqrt(rho ** 2 - epsilon)
        tmp_pos = abs(-rho + sqrt_rho_epsilon)
        tmp_neg = abs(-rho - sqrt_rho_epsilon)
        if tmp_pos > tmp_neg:
            m_trans = k * (-rho - sqrt_rho_epsilon)
        else:
            m_trans = k * (-rho + sqrt_rho_epsilon)

        # Get the incident wave vector
        kin_grid[idx, 0] = ku - h[0] - m_trans * n[0]
        kin_grid[idx, 1] = kw - h[1] - m_trans * n[1]
        kin_grid[idx, 2] = kv - h[2] - m_trans * n[2]


@cuda.jit('void'
          '(float64[:,:], float64[:],'
          'float64[:], float64[:,:], '
          'float64[:], float64[:], float64, float64, '
          'int64)')
def get_kin_grid_and_jacobian(kin_grid, jacobian_grid,
                              klen_grid, kout_grid,
                              h, n, dot_hn, h_square,
                              num):
    """
    Given kout info, this function derives the corresponding kin info.

    :param kin_grid: This function derive the input wave vectors
    :param jacobian_grid:
    :param klen_grid: The wave vector length
    :param kout_grid:
    :param h: The crystal h vector
    :param n: The crystal normal direction
    :param dot_hn: The inner product between h and n
    :param h_square: The length of the h vector
    :param num:
    :return:
    """
    # Step 0: Get the cuda grid idx
    idx = cuda.grid(1)
    if idx < num:

        ##################################################################
        # Step 1: Get the corresponding parameters to get the reflectivity
        ##################################################################
        # Get k
        ku = kout_grid[idx, 0]
        kv = kout_grid[idx, 2]
        kw = kout_grid[idx, 1]
        kvwn = kv * n[2] + kw * n[1]
        kvwh = kv * h[2] + kw * h[1]

        k = klen_grid[idx]
        k_square = k ** 2

        # Get rho and epsilon
        dot_kn = kvwn + ku * n[0]
        dot_kh = kvwh + ku * h[0]
        rho = (dot_hn - dot_kn) / k
        epsilon = (h_square - 2 * dot_kh) / k_square

        # Decide the sign
        sqrt_rho_epsilon = math.sqrt(rho ** 2 - epsilon)
        tmp_pos = abs(-rho + sqrt_rho_epsilon)
        tmp_neg = abs(-rho - sqrt_rho_epsilon)
        if tmp_pos > tmp_neg:
            m_trans = k * (-rho - sqrt_rho_epsilon)
        else:
            m_trans = k * (-rho + sqrt_rho_epsilon)

        # Get the incident wave vector
        kin_grid[idx, 0] = ku - h[0] - m_trans * n[0]
        kin_grid[idx, 1] = kw - h[1] - m_trans * n[1]
        kin_grid[idx, 2] = kv - h[2] - m_trans * n[2]

        # Get the jacobian grid
        jacobian_grid[idx] *= math.fabs(dot_kn / (dot_kn - dot_hn - m_trans))


@cuda.jit('void'
          '(complex128[:], float64[:,:],'
          'float64[:], float64[:], float64, float64,'
          'float64, float64[:], float64[:], '
          'float64, float64, float64, float64, '
          'complex128, complex128, complex128, '
          'int64)')
def get_coplane_bragg_sigma_reflectivity_scan(reflectivity, kin_grid,
                                              klen_grid, kx_grid, ky, kz,
                                              d, h, n,
                                              dot_hn, h_square, kyzn, kyzh,
                                              chi0, chih_sigma, chihbar_sigma,
                                              num):
    """
    Assuming that one specify the kout info, this function gives the reflectivity info.
    Notice that, this function assumes that the polarization is sigma polarization and
    the diffraction is coplane. For general polarization and general 2-beam
    diffraction, please use function get_bragg_field_natural_direction

    :param reflectivity: The output reflectivity
    :param kin_grid: This function derive the input wave vectors
    :param klen_grid: The wave vector length
    :param kx_grid:  The ku grid. This is for x axis
    :param kz: The kv value. This is the z axis.
    :param ky: The kw value. This is the y axis
    :param d: The crystal thickness
    :param h: The crystal h vector
    :param n: The crystal normal direction
    :param dot_hn: The inner product between h and n
    :param h_square: The length of the h vector
    :param kyzn: kz * n[2] + ky * n[1]
    :param kyzh: kz * h[2] + ky * h[1]
    :param chi0:
    :param chih_sigma:
    :param chihbar_sigma:
    :param num:
    :return:
    """

    # This function only handle the sigma polarization
    polar_factor = 1.

    # Step 0: Get the cuda grid idx
    idx = cuda.grid(1)
    if idx < num:

        ##################################################################
        # Step 1: Get the corresponding parameters to get the reflectivity
        ##################################################################
        # Get k
        kx = kx_grid[idx]
        k = klen_grid[idx]
        k_square = k ** 2

        # Get rho and epsilon
        dot_kn = kyzn + kx * n[0]
        dot_kh = kyzh + kx * h[0]
        rho = (dot_hn - dot_kn) / k
        epsilon = (h_square - 2 * dot_kh) / k_square

        # Get momentum : Notice that this might not be the most efficient methods
        # however, considering that later, I need to consider multiple reflection
        # I would need this anyway.

        sqrt_rho_epsilon = math.sqrt(rho ** 2 - epsilon)
        tmp_pos = abs(-rho + sqrt_rho_epsilon)
        tmp_neg = abs(-rho - sqrt_rho_epsilon)
        if tmp_pos > tmp_neg:
            m_trans = k * (-rho - sqrt_rho_epsilon)
        else:
            m_trans = k * (-rho + sqrt_rho_epsilon)

        # get the incident wave vector
        kin_grid[idx, 0] = kx - h[0] - m_trans * n[0]
        kin_grid[idx, 1] = ky - h[1] - m_trans * n[1]
        kin_grid[idx, 2] = kz - h[2] - m_trans * n[2]

        # Get gamma 0,h and alpha
        gamma_h = (dot_kn - m_trans) / k
        gamma_0 = gamma_h - dot_hn / k
        b = gamma_0 / gamma_h
        alpha = (2 * (dot_kh - m_trans * dot_hn) - h_square) / k_square

        ##################################################################
        # Step 2: Get the reflectivity
        ##################################################################
        # Get alpha tidle
        alpha_tidle = complex((alpha * b + chi0.real * (1. - b)) / 2., chi0.imag * (1. - b) / 2.)

        # Get sqrt(alpha**2 + beta**2) value
        sqrt_a2_b2 = cmath.sqrt(alpha_tidle ** 2 +
                                complex(polar_factor * b, 0.) * chih_sigma * chihbar_sigma)

        if sqrt_a2_b2.imag < 0:
            sqrt_a2_b2 = - sqrt_a2_b2

        # Calculate the phase term
        re = k * d / gamma_0 * sqrt_a2_b2.real
        im = k * d / gamma_0 * sqrt_a2_b2.imag

        # Take care of the exponential
        if im <= 100.:
            magnitude = complex(math.exp(-im), 0.)

            phase = complex(math.cos(re), math.sin(re))
            # Calculate some intermediate part
            numerator = 1. - magnitude * phase
            denominator = alpha_tidle * numerator + sqrt_a2_b2 * (2. - numerator)

            # Assemble everything
            reflectivity[idx] = complex(b * polar_factor, 0.) * chih_sigma * numerator / denominator

        else:
            # When the crystal is super thick, the numerator in the previous branch becomes 1.
            # The exponential term becomes 0.

            # Calculate some intermediate part
            denominator = alpha_tidle + sqrt_a2_b2

            # Assemble everything
            reflectivity[idx] = complex(b * polar_factor, 0.) * chih_sigma / denominator


@cuda.jit('void'
          '(complex128[:], float64[:,:],'
          'float64[:], float64[:,:], '
          'float64, float64[:], float64[:], '
          'float64, float64, '
          'complex128, complex128, complex128, '
          'int64)')
def get_coplane_bragg_sigma_reflectivity(reflectivity, kin_grid,
                                         klen_grid, kout_grid,
                                         d, h, n,
                                         dot_hn, h_square,
                                         chi0, chih_sigma, chihbar_sigma,
                                         num):
    """
    Assuming that one specify the kout info, this function gives the reflectivity info.
    Notice that, this function assumes that the polarization is sigma polarization and
    the diffraction is coplane. For general polarization and general 2-beam
    diffraction, please use function get_bragg_field_natural_direction

    :param reflectivity: The output reflectivity
    :param kin_grid: This function derive the input wave vectors
    :param klen_grid: The wave vector length
    :param kout_grid:
    :param d: The crystal thickness
    :param h: The crystal h vector
    :param n: The crystal normal direction
    :param dot_hn: The inner product between h and n
    :param h_square: The length of the h vector
    :param chi0:
    :param chih_sigma:
    :param chihbar_sigma:
    :param num:
    :return:
    """

    # This function only handle the sigma polarization
    polar_factor = 1.

    # Step 0: Get the cuda grid idx
    idx = cuda.grid(1)
    if idx < num:

        ##################################################################
        # Step 1: Get the corresponding parameters to get the reflectivity
        ##################################################################
        # Get k
        kx = kout_grid[idx, 0]
        ky = kout_grid[idx, 1]
        kz = kout_grid[idx, 2]

        kyzn = kz * n[2] + ky * n[1]
        kyzh = kz * h[2] + ky * h[1]

        k = klen_grid[idx]
        k_square = k ** 2

        # Get rho and epsilon
        dot_kn = kyzn + kx * n[0]
        dot_kh = kyzh + kx * h[0]
        rho = (dot_hn - dot_kn) / k
        epsilon = (h_square - 2 * dot_kh) / k_square

        # Get momentum : Notice that this might not be the most efficient methods
        # however, considering that later, I need to consider multiple reflection
        # I would need this anyway.

        sqrt_rho_epsilon = math.sqrt(rho ** 2 - epsilon)
        tmp_pos = abs(-rho + sqrt_rho_epsilon)
        tmp_neg = abs(-rho - sqrt_rho_epsilon)
        if tmp_pos > tmp_neg:
            m_trans = k * (-rho - sqrt_rho_epsilon)
        else:
            m_trans = k * (-rho + sqrt_rho_epsilon)

        # get the incident wave vector
        kin_grid[idx, 0] = kx - h[0] - m_trans * n[0]
        kin_grid[idx, 1] = ky - h[1] - m_trans * n[1]
        kin_grid[idx, 2] = kz - h[2] - m_trans * n[2]

        # Get gamma 0,h and alpha
        gamma_h = (dot_kn - m_trans) / k
        gamma_0 = gamma_h - dot_hn / k
        b = gamma_0 / gamma_h
        alpha = (2 * (dot_kh - m_trans * dot_hn) - h_square) / k_square

        ##################################################################
        # Step 2: Get the reflectivity
        ##################################################################
        # Get alpha tidle
        alpha_tidle = complex((alpha * b + chi0.real * (1. - b)) / 2., chi0.imag * (1. - b) / 2.)

        # Get sqrt(alpha**2 + beta**2) value
        sqrt_a2_b2 = cmath.sqrt(alpha_tidle ** 2 +
                                complex(polar_factor * b, 0.) * chih_sigma * chihbar_sigma)

        if sqrt_a2_b2.imag < 0:
            sqrt_a2_b2 = - sqrt_a2_b2

        # Calculate the phase term
        re = k * d / gamma_0 * sqrt_a2_b2.real
        im = k * d / gamma_0 * sqrt_a2_b2.imag

        # Take care of the exponential
        if im <= 100.:
            magnitude = complex(math.exp(-im), 0.)

            phase = complex(math.cos(re), math.sin(re))
            # Calculate some intermediate part
            numerator = 1. - magnitude * phase
            denominator = alpha_tidle * numerator + sqrt_a2_b2 * (2. - numerator)

            # Assemble everything
            reflectivity[idx] = complex(b * polar_factor, 0.) * chih_sigma * numerator / denominator

        else:
            # When the crystal is super thick, the numerator in the previous branch becomes 1.
            # The exponential term becomes 0.

            # Calculate some intermediate part
            denominator = alpha_tidle + sqrt_a2_b2

            # Assemble everything
            reflectivity[idx] = complex(b * polar_factor, 0.) * chih_sigma / denominator


@cuda.jit("void(float64[:], float64[:,:],"
          "float64[:,:], float64[:], float64[:],"
          "float64[:,:], float64[:], float64[:], int64)")
def get_intersection_point(path_length_remain,
                           intersect_point,
                           kvec_grid,
                           klen_gird,
                           path_length,
                           source_point,
                           surface_position,
                           surface_normal,
                           num):
    """
    This function trace down the intersection point of the previous reflection plane.
    Then calculate the distance and then calculate the remaining distance to go
    to get to the initial point of this k component.

    :param path_length_remain:
    :param intersect_point:
    :param kvec_grid: The incident k vector. Notice that I need this for all the
                    reflections. Therefore, I can not pre define kv ** 2 + ku ** 2
                    to reduce calculation
    :param klen_gird: Notice that all the reflections does not change the length
                        of the wave vectors. Therefore, I do not need to calculate
                        this value again and again.
    :param path_length:
    :param source_point:
    :param surface_position:
    :param surface_normal:
    :param num:
    :return:
    """
    idx = cuda.grid(1)
    if idx < num:
        # Get the coefficient before K
        coef_k = (surface_normal[0] * (surface_position[0] - source_point[idx, 0]) +
                  surface_normal[1] * (surface_position[1] - source_point[idx, 1]) +
                  surface_normal[2] * (surface_position[2] - source_point[idx, 2]))
        coef_k /= (surface_normal[0] * kvec_grid[idx, 0] +
                   surface_normal[1] * kvec_grid[idx, 1] +
                   surface_normal[2] * kvec_grid[idx, 2])

        # Assign the value
        intersect_point[idx, 0] = source_point[idx, 0] + coef_k * kvec_grid[idx, 0]
        intersect_point[idx, 1] = source_point[idx, 1] + coef_k * kvec_grid[idx, 1]
        intersect_point[idx, 2] = source_point[idx, 2] + coef_k * kvec_grid[idx, 2]

        # Get the distance change
        distance = math.fabs(coef_k * klen_gird[idx])
        path_length_remain[idx] = path_length[idx] - distance


@cuda.jit("void(float64[:], float64[:,:],"
          "float64[:,:], float64[:], float64,"
          "float64[:], float64[:], float64[:], int64)")
def get_intersection_point_final_reflection(path_length_remain,
                                            intersect_point,
                                            kvec_grid,
                                            klen_gird,
                                            path_length,
                                            source_point,
                                            surface_position,
                                            surface_normal,
                                            num):
    """
    This function tries to handle the problem that our observation point is a
    single point. Therefore, to get the reflection point for the last reflection,
    the source point has to be the observation point.

    :param path_length_remain:
    :param intersect_point:
    :param kvec_grid: The incident k vector. Notice that I need this for all the
                    reflections. Therefore, I can not pre define kv ** 2 + ku ** 2
                    to reduce calculation
    :param klen_gird: Notice that all the reflections does not change the length
                        of the wave vectors. Therefore, I do not need to calculate
                        this value again and again.
    :param path_length:
    :param source_point:
    :param surface_position:
    :param surface_normal:
    :param num:
    :return:
    """
    idx = cuda.grid(1)
    if idx < num:
        # Get the coefficient before K
        coef_k = (surface_normal[0] * (surface_position[0] - source_point[0]) +
                  surface_normal[1] * (surface_position[1] - source_point[1]) +
                  surface_normal[2] * (surface_position[2] - source_point[2]))
        coef_k /= (surface_normal[0] * kvec_grid[idx, 0] +
                   surface_normal[1] * kvec_grid[idx, 1] +
                   surface_normal[2] * kvec_grid[idx, 2])

        # Assign the value
        intersect_point[idx, 0] = source_point[0] + coef_k * kvec_grid[idx, 0]
        intersect_point[idx, 1] = source_point[1] + coef_k * kvec_grid[idx, 1]
        intersect_point[idx, 2] = source_point[2] + coef_k * kvec_grid[idx, 2]

        # Get the distance change
        distance = math.fabs(coef_k * klen_gird[idx])
        path_length_remain[idx] = path_length - distance


@cuda.jit("void(float64[:,:], float64[:,:], float64[:,:], float64[:], float64[:], int64)")
def find_source_point(source_point, end_point, kvec_grid, klen_grid, path_length, num):
    """
    Find the source point of this wave vector component at time 0.

    :param source_point:
    :param end_point:
    :param kvec_grid:
    :param klen_grid
    :param path_length:
    :param num:
    :return:
    """
    idx = cuda.grid(1)
    if idx < num:
        # Get the length of the wave vector
        # k_len = klen_grid[idx]

        coef = path_length[idx] / klen_grid[idx]
        # coef = path_length[idx] / k_len

        source_point[idx, 0] = end_point[idx, 0] - coef * kvec_grid[idx, 0]
        source_point[idx, 1] = end_point[idx, 1] - coef * kvec_grid[idx, 1]
        source_point[idx, 2] = end_point[idx, 2] - coef * kvec_grid[idx, 2]


@cuda.jit("void(complex128[:], float64[:,:], float64[:], float64[:,:], int64)")
def get_phase_from_space(phase, source_point, reference_point, k_vec, num):
    """
    At present, I assume that the reference point is the same for all the components
    Then this function, calculate the relative phase of this wave component at the
    source point with respect to the reference point.

    The phase at the reference point is the phase that we have obtained when we
    take the fourier transformation of the gaussian field at the x0 position at time t=0

    :param phase:
    :param source_point:
    :param reference_point:
    :param k_vec:
    :param num:
    :return:
    """
    idx = cuda.grid(1)
    if idx < num:
        tmp = (k_vec[idx, 0] * (source_point[idx, 0] - reference_point[0]) +
               k_vec[idx, 1] * (source_point[idx, 1] - reference_point[1]) +
               k_vec[idx, 2] * (source_point[idx, 2] - reference_point[2]))

        phase[idx] = complex(math.cos(tmp), math.sin(tmp))


@cuda.jit("void(complex128[:], float64[:,:], float64[:], float64[:,:], float64[:], int64)")
def get_relative_phase_from_space(phase, source_point, reference_point, k_vec, k_ref, num):
    """
    At present, I assume that the reference point is the same for all the components
    Then this function, calculate the relative phase of this wave component at the
    source point with respect to the reference point.

    The phase at the reference point is the phase that we have obtained when we
    take the fourier transformation of the gaussian field at the x0 position at time t=0

    :param phase:
    :param source_point:
    :param reference_point:
    :param k_vec:
    :param k_ref:
    :param num:
    :return:
    """
    idx = cuda.grid(1)
    if idx < num:
        tmp = ((k_vec[idx, 0] - k_ref[0]) * (source_point[idx, 0] - reference_point[0]) +
               (k_vec[idx, 1] - k_ref[1]) * (source_point[idx, 1] - reference_point[1]) +
               (k_vec[idx, 2] - k_ref[2]) * (source_point[idx, 2] - reference_point[2]))

        phase[idx] = complex(math.cos(tmp), math.sin(tmp))


@cuda.jit("void(complex128[:], float64[:,:], float64[:], float64[:], "
          "float64[:], float64[:,:], float64[:], int64)")
def get_phase_and_distance(phase, source_point, distance, spatial_phase,
                           reference_point, k_vec, k_ref, num):
    """
    At present, I assume that the reference point is the same for all the components
    Then this function, calculate the relative phase of this wave component at the
    source point with respect to the reference point.

    The phase at the reference point is the phase that we have obtained when we
    take the fourier transformation of the gaussian field at the x0 position at time t=0

    :param phase:
    :param source_point:
    :param distance:
    :param spatial_phase:
    :param reference_point:
    :param k_vec:
    :param k_ref:
    :param num:
    :return:
    """
    idx = cuda.grid(1)
    if idx < num:
        dx = source_point[idx, 0] - reference_point[0]
        dy = source_point[idx, 1] - reference_point[1]
        dz = source_point[idx, 2] - reference_point[2]

        tmp = ((k_vec[idx, 0] - k_ref[0]) * dx +
               (k_vec[idx, 1] - k_ref[1]) * dy +
               (k_vec[idx, 2] - k_ref[2]) * dz)
        spatial_phase[idx] = tmp

        phase[idx] = complex(math.cos(tmp), math.sin(tmp))
        # Get the distance
        distance[idx] = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)


################################################################################
#  Get pulse spectrum
################################################################################
@cuda.jit('void'
          '(complex128[:], '
          'float64[:,:], float64, '
          'float64[:,:], complex128,'
          'float64[:], float64[:], float64, float64[:], int64)')
def get_gaussian_pulse_spectrum(coef,
                                k_vec, t,
                                sigma_mat, scaling,
                                x0, k0, omega0, n,
                                num):
    """
    Calculate the corresponding coefficient in the incident gaussian pulse
    for each wave vectors to investigate.

    :param coef: The coefficent to be calculated
    :param k_vec: The wave vectors to be calculated
    :param t: The time for the snapshot. This value will usually be 0.
    :param num: The number of wave vectors to calculate
    :param sigma_mat: The sigma matrix of the Gaussian pulse. Notice that here,
                        the elements in this matrix should have unit of um.
    :param scaling: A linear scaling coefficient to take the intensity
                    and some other factors in to consideration.
    :param x0: This is the displacement vector with respect to the origin of the incident pulse frame.
    :param k0:
    :param omega0:
    :param n: The direction of k0
    :return:
    """
    row = cuda.grid(1)
    if row < num:
        # Get the momentum difference
        dk_x = k0[0] - k_vec[row, 0]
        dk_y = k0[1] - k_vec[row, 1]
        dk_z = k0[2] - k_vec[row, 2]

        phase = -t * omega0

        # Notice that here, we are using the fourier component rather than the characteristic function
        # If it's characteristic function, then it should be +=
        phase -= ((x0[0] + c * t * n[0]) * dk_x +
                  (x0[1] + c * t * n[1]) * dk_y +
                  (x0[2] + c * t * n[2]) * dk_z)

        phase_term = complex(math.cos(phase), math.sin(phase))

        # Get the quadratic term
        quad_term = - (dk_x * sigma_mat[0, 0] * dk_x + dk_x * sigma_mat[0, 1] * dk_y +
                       dk_x * sigma_mat[0, 2] * dk_z +
                       dk_y * sigma_mat[1, 0] * dk_x + dk_y * sigma_mat[1, 1] * dk_y +
                       dk_y * sigma_mat[1, 2] * dk_z +
                       dk_z * sigma_mat[2, 0] * dk_x + dk_z * sigma_mat[2, 1] * dk_y +
                       dk_z * sigma_mat[2, 2] * dk_z
                       ) / 2.

        # if quad_term >= -200:
        magnitude = scaling * complex(math.exp(quad_term), 0)
        coef[row] = magnitude * phase_term


@cuda.jit('void'
          '(complex128[:], float64[:,:],'
          'float64[:], float64, float64, float64, complex128,'
          'int64)')
def get_square_pulse_spectrum(coef,
                              k_vec,
                              k0, a_val, b_val, c_val, scaling,
                              num):
    """
    Calculate the spectrum of a square pulse. 
    
    :param coef: 
    :param k_vec: 
    :param k0: 
    :param a_val: 
    :param b_val: 
    :param c_val: 
    :param scaling: 
    :param num: 
    :return: 
    """
    row = cuda.grid(1)
    if row < num:
        # Get the momentum difference
        dk_x = a_val * (k_vec[row, 0] - k0[0]) / 2.
        dk_y = b_val * (k_vec[row, 1] - k0[1]) / 2.
        dk_z = c_val * (k_vec[row, 2] - k0[2]) / 2.

        holder = 1.

        # Get the contribution from the x component
        if math.fabs(dk_x) <= eps:
            holder *= (0.05 * dk_x ** 2 - 1.) * (dk_x ** 2) / 6. + 1.
        else:
            holder *= math.sin(dk_x) / dk_x

        # Get the contribution from the y component
        if math.fabs(dk_y) <= eps:
            holder *= (0.05 * dk_y ** 2 - 1.) * (dk_y ** 2) / 6. + 1.
        else:
            holder *= math.sin(dk_y) / dk_y

        # Get the contribution from the z component
        if math.fabs(dk_z) <= eps:
            holder *= (0.05 * dk_z ** 2 - 1.) * (dk_z ** 2) / 6. + 1.
        else:
            holder *= math.sin(dk_z) / dk_z

        coef[row] = scaling * complex(holder, 0.)


@cuda.jit('void'
          '(complex128[:], float64[:,:],'
          'float64[:], float64, float64, float64, complex128, float64, '
          'int64)')
def get_square_pulse_spectrum_smooth(coef,
                                     k_vec,
                                     k0, a_val, b_val, c_val, scaling, sigma,
                                     num):
    """
    Calculate the spectrum of a square pulse.

    :param coef:
    :param k_vec:
    :param k0:
    :param a_val:
    :param b_val:
    :param c_val:
    :param scaling:
    :param sigma: The sigma value of the gaussian filter.
    :param num:
    :return:
    """
    row = cuda.grid(1)
    if row < num:
        # Get the momentum difference
        dk_x = a_val * (k_vec[row, 0] - k0[0]) / 2.
        dk_y = b_val * (k_vec[row, 1] - k0[1]) / 2.
        dk_z = c_val * (k_vec[row, 2] - k0[2]) / 2.

        holder = 1.

        # Get the contribution from the x component
        if math.fabs(dk_x) <= eps:
            holder *= (0.05 * dk_x ** 2 - 1.) * (dk_x ** 2) / 6. + 1.
        else:
            holder *= math.sin(dk_x) / dk_x

        # Get the contribution from the y component
        if math.fabs(dk_y) <= eps:
            holder *= (0.05 * dk_y ** 2 - 1.) * (dk_y ** 2) / 6. + 1.
        else:
            holder *= math.sin(dk_y) / dk_y

        # Get the contribution from the z component
        if math.fabs(dk_z) <= eps:
            holder *= (0.05 * dk_z ** 2 - 1.) * (dk_z ** 2) / 6. + 1.
        else:
            holder *= math.sin(dk_z) / dk_z

        gaussian = math.exp(-(dk_x ** 2 + dk_y ** 2 + dk_z ** 2) * (sigma ** 2) / 2.)
        coef[row] = scaling * complex(holder * gaussian, 0.)
