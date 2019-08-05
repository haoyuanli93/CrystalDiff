import numpy as np
import sys
import time
import h5py as h5
import pickle

sys.path.append("/home/haoyuan/my_repos/CrystalDiffraction")

from numba import cuda
from pyculib import fft as cufft
from CrystalDiff import util, pulse, lclsutil, gutil_bk_2019_8_5

"""
                Calculation Plan
    For this simulation. I would like to have a 3D result.
    For the 3D result, the 2D slices are calculated one by one. By this, I mean, 
    all the 2D information are calculated directed in the GPU without further
    interference from the CPU.
"""
# ---------------------------------
# Specify the output file name
# ---------------------------------
pre_tag = "inclined_part_{}".format(5)

# Change the STDOUT stream to the file to save the log.
stdout = sys.stdout
sys.stdout = open('./{}_log.txt'.format(pre_tag), 'w')

############################################################################################################
# ----------------------------------------------------------------------------------------------------------
#
#                       Step 1: Prepare the devices and pulses
#
# ----------------------------------------------------------------------------------------------------------
############################################################################################################
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                               Crystal
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set up bragg angle
h_length = 2. * np.pi / (1.9201 * 1e-4)
hlen_vals = np.array([h_length, ] * 4)

bragg_angle = np.radians(18.836)

# Set up the angles
theta = np.pi / 2. + bragg_angle
theta_vals = [theta,
              np.pi + theta,
              - theta,
              np.pi - theta]
rho = bragg_angle - np.pi / 2.
rho_vals = [rho,
            np.pi + rho,
            - rho,
            np.pi - rho]

tau_vals = [0.,
            np.radians(10.),
            - np.radians(10.),
            0.,
            ]

# Set up the surface position
surface_points = [np.zeros(3, dtype=np.float64) for x in range(4)]

# Initialize the crystals
crystal_list = lclsutil.get_crystal_list(num=4,
                                         hlen_vals=hlen_vals,
                                         rho_vals=rho_vals,
                                         theta_vals=theta_vals,
                                         tau_vals=tau_vals,
                                         surface_points=surface_points,
                                         chi0=complex(-0.97631E-05, 0.14871E-06),
                                         chih_sigma=complex(0.59310E-05, -0.14320E-06),
                                         chihbar_sigma=complex(0.59310E-05, -0.14320E-06),
                                         chih_pi=complex(0.46945E-05, -0.11201E-06),
                                         chihbar_pi=complex(0.46945E-05, -0.11201E-06)
                                         )

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                               Pulse
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# From FWHM to sigma
# fwhm = 500.  # um
# sigma = fwhm / 2 / np.sqrt(2 * np.log(2))
# sigma_t = sigma / util.c
# print("The corresponding sigma in the definition of the pulse should be {:.2f} fs.".format(sigma_t))

# Set up the pulse
my_pulse = pulse.GaussianPulse3D()

energy_center = 10.
k_len = util.kev_to_wave_number(energy=energy_center)

my_pulse.polar = np.array([0., 1., 0.], dtype=np.complex128)

my_pulse.k0 = np.array([0., 0., k_len])
my_pulse.n = my_pulse.k0 / util.l2_norm(my_pulse.k0)
my_pulse.omega0 = k_len * util.c

my_pulse.sigma_x = 100.  # sigma_t
my_pulse.sigma_y = 100.  # sigma_t  # fs
my_pulse.sigma_z = 1.  # fs
my_pulse.sigma_mat = np.diag(np.array([my_pulse.sigma_x ** 2,
                                       my_pulse.sigma_y ** 2,
                                       my_pulse.sigma_z ** 2], dtype=np.float64))
my_pulse.sigma_mat *= util.c ** 2

magnitude_peak = 1.
my_pulse.scaling = complex(my_pulse.sigma_x * my_pulse.sigma_y *
                           my_pulse.sigma_z * (util.c ** 3), 0.) * magnitude_peak

pre_length = 3e4
my_pulse.x0 = np.array([0., 0., -pre_length])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Arrange the crystal separation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
path_sections = [0., 2. * 1e4, 12. * 1e4, 2. * 1e4]

intersection_points, kout_vec_list = lclsutil.get_intersection_point(kin_vec=my_pulse.k0,
                                                                     path_sections=path_sections,
                                                                     crystal_list=crystal_list)

crystal_list = lclsutil.get_crystal_list(num=4,
                                         hlen_vals=hlen_vals,
                                         rho_vals=rho_vals,
                                         theta_vals=theta_vals,
                                         tau_vals=tau_vals,
                                         surface_points=np.copy(intersection_points),
                                         chi0=complex(-0.97631E-05, 0.14871E-06),
                                         chih_sigma=complex(0.59310E-05, -0.14320E-06),
                                         chihbar_sigma=complex(0.59310E-05, -0.14320E-06),
                                         chih_pi=complex(0.46945E-05, -0.11201E-06),
                                         chihbar_pi=complex(0.46945E-05, -0.11201E-06)
                                         )
"""
Here is the starting point where different processes are distinguished
"""
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get the observation point
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
post_length = path_sections[2]
observation = (intersection_points[1] +
               post_length * kout_vec_list[1] / util.l2_norm(kout_vec_list[1]))

total_path = (pre_length + post_length +
              util.get_total_path_length(intersection_point_list=intersection_points[:2]))
print("The total propagation length is {:.2f}um.".format(total_path))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                  Change frame
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(my_pulse,
 crystal_list,
 observation,
 rot_mat_dict) = lclsutil.get_to_kout_frame_lcls2(kin=my_pulse.k0,
                                                  kout=kout_vec_list[1],
                                                  h=crystal_list[1].h,
                                                  displacement=-intersection_points[1],
                                                  obvservation=observation,
                                                  pulse=my_pulse,
                                                  crystal_list=crystal_list)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                  Get the momentum mesh
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
number_x = 500
number_y = 500
number_z = 10 ** 5
kx_grid, ky_grid, kz_grid, axis_info = lclsutil.get_k_mesh_3d(number_x=number_x,
                                                              number_y=number_y,
                                                              number_z=number_z,
                                                              delta_e_x=1e-3,
                                                              delta_e_y=1e-3,
                                                              delta_e_z=1e-3 / util.c)
kz_grid += k_len

# Apply fft shift
kx_grid = np.ascontiguousarray(np.fft.fftshift(kx_grid))
ky_grid = np.ascontiguousarray(np.fft.fftshift(ky_grid))
kz_grid = np.ascontiguousarray(np.fft.fftshift(kz_grid))

# Save the pickle:
with open("./{}_axis_info.pkl".format(pre_tag), 'wb') as axis_file:
    pickle.dump(axis_info, axis_file)

############################################################################################################
# ----------------------------------------------------------------------------------------------------------
#
#                       Step 2: Prepare the cuda array
#
# ----------------------------------------------------------------------------------------------------------
############################################################################################################
tic = time.time()

# Set the range of the index to save
z_idx_range = 500
num1 = 450
num2 = 50

central_start = int(z_idx_range // 2 - z_idx_range // 5)
idx_start_1 = number_z - num1
idx_start_2 = 0

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  [3D Blocks] For visualization
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
efield_3d = np.zeros((number_x, number_y, z_idx_range, 3), dtype=np.complex128)
efield_spectrum_3d = np.zeros((number_x, number_y, z_idx_range, 3), dtype=np.complex128)
reflect_pi_3d = np.zeros((number_x, number_y, z_idx_range), dtype=np.complex128)
reflect_sigma_3d = np.zeros((number_x, number_y, z_idx_range), dtype=np.complex128)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  [2D slices] Final field in the simulation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Original field
x_field_holder_2d = np.ascontiguousarray(np.zeros((number_y, z_idx_range), dtype=np.complex128))
y_field_holder_2d = np.ascontiguousarray(np.zeros((number_y, z_idx_range), dtype=np.complex128))
z_field_holder_2d = np.ascontiguousarray(np.zeros((number_y, z_idx_range), dtype=np.complex128))

cuda_x_field_2d = cuda.to_device(x_field_holder_2d)
cuda_y_field_2d = cuda.to_device(y_field_holder_2d)
cuda_z_field_2d = cuda.to_device(z_field_holder_2d)

# Original spectrum
x_spectrum_holder_2d = np.ascontiguousarray(np.zeros((number_y, z_idx_range), dtype=np.complex128))
y_spectrum_holder_2d = np.ascontiguousarray(np.zeros((number_y, z_idx_range), dtype=np.complex128))
z_spectrum_holder_2d = np.ascontiguousarray(np.zeros((number_y, z_idx_range), dtype=np.complex128))

cuda_x_spectrum_2d = cuda.to_device(x_spectrum_holder_2d)
cuda_y_spectrum_2d = cuda.to_device(y_spectrum_holder_2d)
cuda_z_spectrum_2d = cuda.to_device(z_spectrum_holder_2d)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  [2D slices] Reflectivity
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Holder for the reflectivity
reflect_pi_2d = np.ascontiguousarray(np.zeros((number_y, z_idx_range), dtype=np.complex128))
cuda_reflect_pi_2d = cuda.to_device(reflect_pi_2d)

reflect_sigma_2d = np.ascontiguousarray(np.zeros((number_y, z_idx_range), dtype=np.complex128))
cuda_reflect_sigma_2d = cuda.to_device(reflect_sigma_2d)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  [1D slices] Various intersection points, path length and phase
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The holder for the intersection point
intersection_point = np.ascontiguousarray(np.zeros((number_z, 3), dtype=np.float64))
cuda_intersect = cuda.to_device(intersection_point)

# This holder for the initial point for each wave component
source_point = np.ascontiguousarray(np.zeros((number_z, 3), dtype=np.float64))
cuda_source_point = cuda.to_device(source_point)

# This is the holder for the remaining path length
remaining_length = np.ascontiguousarray(np.zeros(number_z, dtype=np.float64))
cuda_remain_path = cuda.to_device(remaining_length)

# Holder for the propagation phase for the wave front distortion
phase_grid = np.ascontiguousarray(np.zeros(number_z, dtype=np.complex128))
cuda_phase = cuda.to_device(phase_grid)

spatial_phase_grid = np.ascontiguousarray(np.zeros(number_z, dtype=np.float64))
cuda_spatial_phase = cuda.to_device(spatial_phase_grid)

distance_grid = np.ascontiguousarray(np.zeros(number_z, dtype=np.float64))
cuda_distance_grid = cuda.to_device(distance_grid)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# [1D slices] Reflectivity and time response
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Holder for the reflectivity
reflectivity_pi = np.ascontiguousarray(np.zeros(number_z, dtype=np.complex128))
cuda_reflect_pi = cuda.to_device(reflectivity_pi)

reflectivity_sigma = np.ascontiguousarray(np.zeros(number_z, dtype=np.complex128))
cuda_reflect_sigma = cuda.to_device(reflectivity_sigma)

# Holder for the total reflectivity
reflectivity_total_sigma = np.ascontiguousarray(np.ones(number_z, dtype=np.complex128))
cuda_reflect_total_sigma = cuda.to_device(reflectivity_total_sigma)

reflectivity_total_pi = np.ascontiguousarray(np.ones(number_z, dtype=np.complex128))
cuda_reflect_total_pi = cuda.to_device(reflectivity_total_pi)

# Holder for the time response function: Usually not useful
time_response = np.ascontiguousarray(np.zeros(number_z, dtype=np.complex128))
cuda_time_response = cuda.to_device(time_response)

# Holder for the time response function: Usually not useful
time_response_total = np.ascontiguousarray(np.zeros(number_z, dtype=np.complex128))
cuda_time_response_total = cuda.to_device(time_response_total)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# [1D slices] Vector field
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The reciprocal space
vector_spectrum_holder = np.ascontiguousarray(np.zeros((number_z, 3), dtype=np.complex128))
cuda_spectrum_vec = cuda.to_device(vector_spectrum_holder)

x_spectrum_holder = np.ascontiguousarray(np.zeros(number_z, dtype=np.complex128))
cuda_spectrum_x = cuda.to_device(x_spectrum_holder)

y_spectrum_holder = np.ascontiguousarray(np.zeros(number_z, dtype=np.complex128))
cuda_spectrum_y = cuda.to_device(y_spectrum_holder)

z_spectrum_holder = np.ascontiguousarray(np.zeros(number_z, dtype=np.complex128))
cuda_spectrum_z = cuda.to_device(z_spectrum_holder)

# Real space
x_field_holder = np.ascontiguousarray(np.zeros(number_z, dtype=np.complex128))
cuda_x_field = cuda.to_device(x_field_holder)

y_field_holder = np.ascontiguousarray(np.zeros(number_z, dtype=np.complex128))
cuda_y_field = cuda.to_device(y_field_holder)

z_field_holder = np.ascontiguousarray(np.zeros(number_z, dtype=np.complex128))
cuda_z_field = cuda.to_device(z_field_holder)

# The coefficient of different wave component in the diffracted field
spectrum_holder = np.ascontiguousarray(np.ones(number_z, dtype=np.complex128))
cuda_spectrum = cuda.to_device(spectrum_holder)

# The coefficient of different wave component in the gaussian field
coefficient = np.ascontiguousarray(np.zeros(number_z, dtype=np.complex128))
cuda_coef = cuda.to_device(coefficient)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  [1D slices] k grid and jacobian grid
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
kz_grid = np.ascontiguousarray(kz_grid)
cuda_kz_grid = cuda.to_device(kz_grid)

# Holder for ku square. This is purely for convenience
kz_square = np.ascontiguousarray(np.square(kz_grid))
cuda_kz_square = cuda.to_device(kz_square)

# Holder for all the wave vectors during this process except the final wave vector grid.
kin_grid = np.ascontiguousarray(np.zeros((number_z, 3), dtype=np.float64))
cuda_kin_grid = cuda.to_device(kin_grid)

# Holder for the scanned wave vectors
kout_grid = np.zeros((number_z, 3), dtype=np.float64)
kout_grid[:, 2] = kz_grid[:]
kout_grid = np.ascontiguousarray(kout_grid)
cuda_kout_grid = cuda.to_device(kout_grid)

# Holder for k length
klen_grid = np.ascontiguousarray(np.zeros(number_z, dtype=np.float64))
cuda_klen_grid = cuda.to_device(klen_grid)

# Jacobian grid
jacob_grid = np.ascontiguousarray(np.zeros(number_z, dtype=np.float64))
cuda_jacob = cuda.to_device(jacob_grid)

# The end of the preparation
toc = time.time()
print("It takes {} seconds to create cuda arrays for the calculation".format(toc - tic))

############################################################################################################
# ----------------------------------------------------------------------------------------------------------
#
#                           Step 3: Calculate the field and save the data
#
# ----------------------------------------------------------------------------------------------------------
############################################################################################################
d_num = 512
b_num = (number_z + d_num - 1) // d_num

tic_tot = time.time()
for x_idx in range(number_x):
    #    # Get the time
    tic = time.time()

    kx = kx_grid[x_idx]
    # kx = 0.
    # x_idx = 0

    for y_idx in range(number_y):
        # Initialize output k vector mesh

        ky = ky_grid[y_idx]
        # ky= 0.

        # --------------------------------------------------------------------
        #  Step 1. Get k_out mesh
        # --------------------------------------------------------------------
        gutil_bk_2019_8_5.initialize_kvec_grid_lcls2[b_num, d_num](cuda_kout_grid,
                                                                   cuda_klen_grid,
                                                                   cuda_kz_grid,
                                                                   cuda_kz_square,
                                                                   kx,
                                                                   ky,
                                                                   ky ** 2 + kx ** 2,
                                                                   number_z)

        gutil_bk_2019_8_5.initialize_jacobian_grid[b_num, d_num](cuda_jacob,
                                                                 number_z)
        # --------------------------------------------------------------------
        #  Step 2. Get k_in mesh and the jacobian
        # --------------------------------------------------------------------

        # --------------
        # Crystal 2
        # --------------
        # Get the intersection point from the final point
        gutil_bk_2019_8_5.get_intersection_point_final_reflection[b_num, d_num](cuda_remain_path,
                                                                                cuda_intersect,
                                                                                cuda_kout_grid,
                                                                                cuda_klen_grid,
                                                                                total_path,
                                                                                observation,
                                                                                crystal_list[1].surface_point,
                                                                                crystal_list[1].normal,
                                                                                number_z)

        # Calculate the incident wave vector
        gutil_bk_2019_8_5.get_kin_grid_and_jacobian[b_num, d_num](cuda_kin_grid,
                                                                  cuda_jacob,
                                                                  cuda_klen_grid,
                                                                  cuda_kout_grid,
                                                                  crystal_list[1].h,
                                                                  crystal_list[1].normal,
                                                                  crystal_list[1].dot_hn,
                                                                  crystal_list[1].h_square,
                                                                  number_z)

        # --------------
        # Crystal 1
        # --------------
        # Get the intersection point from the final point
        gutil_bk_2019_8_5.get_intersection_point[b_num, d_num](cuda_remain_path,
                                                               cuda_intersect,
                                                               cuda_kin_grid,
                                                               cuda_klen_grid,
                                                               cuda_remain_path,
                                                               cuda_intersect,
                                                               crystal_list[0].surface_point,
                                                               crystal_list[0].normal,
                                                               number_z)

        # Calculate the incident wave vector
        gutil_bk_2019_8_5.get_kin_grid_and_jacobian[b_num, d_num](cuda_kin_grid,
                                                                  cuda_jacob,
                                                                  cuda_klen_grid,
                                                                  cuda_kin_grid,
                                                                  crystal_list[0].h,
                                                                  crystal_list[0].normal,
                                                                  crystal_list[0].dot_hn,
                                                                  crystal_list[0].h_square,
                                                                  number_z)
        # """
        # --------------------------------------------------------------------
        #  Step 3. Get the Fourier coefficients
        # --------------------------------------------------------------------
        # Calculate the corresponding coefficient in the incident pulse
        gutil_bk_2019_8_5.get_gaussian_pulse_spectrum[b_num, d_num](cuda_coef,
                                                                    cuda_kin_grid,
                                                                    0.,
                                                                    my_pulse.sigma_mat,
                                                                    my_pulse.scaling,
                                                                    np.zeros(3, dtype=np.float64),
                                                                    my_pulse.k0,
                                                                    my_pulse.omega0,
                                                                    my_pulse.n,
                                                                    number_z)

        # --------------------------------------------------------------------
        #  Step 4. Find the initial source position and phase
        # --------------------------------------------------------------------
        gutil_bk_2019_8_5.find_source_point[b_num, d_num](cuda_source_point,
                                                          cuda_intersect,
                                                          cuda_kin_grid,
                                                          cuda_klen_grid,
                                                          cuda_remain_path,
                                                          number_z)

        # Get the propagational phase from the inital phase.
        # This is the reason for the phase front distortion.
        gutil_bk_2019_8_5.get_relative_phase_from_space[b_num, d_num](cuda_phase,
                                                                      cuda_source_point,
                                                                      my_pulse.x0,
                                                                      cuda_kin_grid,
                                                                      my_pulse.k0,
                                                                      number_z)

        # Add the phase
        gutil_bk_2019_8_5.element_wise_multiply_complex[b_num, d_num](cuda_coef,
                                                                      cuda_phase,
                                                                      cuda_spectrum,
                                                                      number_z
                                                                      )

        # Add Jacobian
        gutil_bk_2019_8_5.element_wise_multiply_complex[b_num, d_num](cuda_spectrum,
                                                                      cuda_jacob,
                                                                      cuda_spectrum,
                                                                      number_z
                                                                      )

        # Get the vector field
        gutil_bk_2019_8_5.expand_scalar_grid_to_vector_grid[b_num, d_num](cuda_spectrum,
                                                                          my_pulse.polar,
                                                                          cuda_spectrum_vec,
                                                                          number_z)

        # --------------------------------------------------------------------
        #  Step 5. Forward propagation
        # --------------------------------------------------------------------
        # """
        # --------------
        # Crystal 1
        # --------------
        gutil_bk_2019_8_5.get_bragg_field_natural_direction[b_num, d_num](cuda_reflect_total_sigma,
                                                                          cuda_reflect_total_pi,
                                                                          cuda_kin_grid,
                                                                          cuda_spectrum_vec,
                                                                          cuda_klen_grid,
                                                                          cuda_kin_grid,
                                                                          crystal_list[0].d,
                                                                          crystal_list[0].h,
                                                                          crystal_list[0].normal,
                                                                          crystal_list[0].dot_hn,
                                                                          crystal_list[0].h_square,
                                                                          crystal_list[0].h_len,
                                                                          crystal_list[0].chi0,
                                                                          crystal_list[0].chih_sigma,
                                                                          crystal_list[0].chihbar_sigma,
                                                                          crystal_list[0].chih_pi,
                                                                          crystal_list[0].chihbar_pi,
                                                                          number_z)
        # --------------
        # Crystal 2
        # --------------
        gutil_bk_2019_8_5.get_bragg_field_natural_direction[b_num, d_num](cuda_reflect_sigma,
                                                                          cuda_reflect_pi,
                                                                          cuda_kin_grid,
                                                                          cuda_spectrum_vec,
                                                                          cuda_klen_grid,
                                                                          cuda_kin_grid,
                                                                          crystal_list[1].d,
                                                                          crystal_list[1].h,
                                                                          crystal_list[1].normal,
                                                                          crystal_list[1].dot_hn,
                                                                          crystal_list[1].h_square,
                                                                          crystal_list[1].h_len,
                                                                          crystal_list[1].chi0,
                                                                          crystal_list[1].chih_sigma,
                                                                          crystal_list[1].chihbar_sigma,
                                                                          crystal_list[1].chih_pi,
                                                                          crystal_list[1].chihbar_pi,
                                                                          number_z)
        gutil_bk_2019_8_5.element_wise_multiply_complex[b_num, d_num](cuda_reflect_sigma,
                                                                      cuda_reflect_total_sigma,
                                                                      cuda_reflect_total_sigma,
                                                                      number_z)
        gutil_bk_2019_8_5.element_wise_multiply_complex[b_num, d_num](cuda_reflect_pi,
                                                                      cuda_reflect_total_pi,
                                                                      cuda_reflect_total_pi,
                                                                      number_z)
        # """
        """
        This is the end point where one needs to modify for different inspection point
        """
        # --------------
        # Save the reflectivity
        # --------------
        gutil_bk_2019_8_5.fill_column_complex[b_num, d_num](cuda_reflect_sigma_2d,
                                                            cuda_reflect_total_sigma,
                                                            central_start,
                                                            y_idx,
                                                            z_idx_range)
        gutil_bk_2019_8_5.fill_column_complex[b_num, d_num](cuda_reflect_pi_2d,
                                                            cuda_reflect_total_pi,
                                                            central_start,
                                                            y_idx,
                                                            z_idx_range)

        # --------------------------------------------------------------------
        #  Step 6. Goes from the reciprocal space to the real space
        # --------------------------------------------------------------------
        # Decompose the electric field
        # Save the result to the total reflectivity
        gutil_bk_2019_8_5.vector_decomposition[b_num, d_num](cuda_spectrum_vec,
                                                             cuda_spectrum_x,
                                                             cuda_spectrum_y,
                                                             cuda_spectrum_z,
                                                             number_z)
        # Save the spectrum of the field
        gutil_bk_2019_8_5.fill_column_complex[b_num, d_num](cuda_x_spectrum_2d,
                                                            cuda_spectrum_x,
                                                            central_start,
                                                            y_idx,
                                                            z_idx_range
                                                            )
        gutil_bk_2019_8_5.fill_column_complex[b_num, d_num](cuda_y_spectrum_2d,
                                                            cuda_spectrum_y,
                                                            central_start,
                                                            y_idx,
                                                            z_idx_range
                                                            )
        gutil_bk_2019_8_5.fill_column_complex[b_num, d_num](cuda_z_spectrum_2d,
                                                            cuda_spectrum_z,
                                                            central_start,
                                                            y_idx,
                                                            z_idx_range
                                                            )

        # Take the fourier transformation
        cufft.ifft(cuda_spectrum_x, cuda_x_field)
        cufft.ifft(cuda_spectrum_y, cuda_y_field)
        cufft.ifft(cuda_spectrum_z, cuda_z_field)

        # Update the data holder
        gutil_bk_2019_8_5.fill_column_complex_fftshift[b_num, d_num](cuda_x_field_2d,
                                                                     cuda_x_field,
                                                                     y_idx,
                                                                     idx_start_1,
                                                                     num1,
                                                                     idx_start_2,
                                                                     num2)

        gutil_bk_2019_8_5.fill_column_complex_fftshift[b_num, d_num](cuda_y_field_2d,
                                                                     cuda_y_field,
                                                                     y_idx,
                                                                     idx_start_1,
                                                                     num1,
                                                                     idx_start_2,
                                                                     num2)

        gutil_bk_2019_8_5.fill_column_complex_fftshift[b_num, d_num](cuda_z_field_2d,
                                                                     cuda_z_field,
                                                                     y_idx,
                                                                     idx_start_1,
                                                                     num1,
                                                                     idx_start_2,
                                                                     num2)
        # """
    ###################################################################################################
    #                                  Finish
    ###################################################################################################

    # Move the 2D slices back to the host and then save them to the variables
    cuda_x_field_2d.to_host()
    cuda_y_field_2d.to_host()
    cuda_z_field_2d.to_host()

    cuda_x_spectrum_2d.to_host()
    cuda_y_spectrum_2d.to_host()
    cuda_z_spectrum_2d.to_host()

    cuda_reflect_sigma_2d.to_host()
    cuda_reflect_pi_2d.to_host()

    # Update the 3D variables.
    efield_3d[x_idx, :, :, 0] = x_field_holder_2d
    efield_3d[x_idx, :, :, 1] = y_field_holder_2d
    efield_3d[x_idx, :, :, 2] = z_field_holder_2d

    efield_spectrum_3d[x_idx, :, :, 0] = x_spectrum_holder_2d
    efield_spectrum_3d[x_idx, :, :, 1] = y_spectrum_holder_2d
    efield_spectrum_3d[x_idx, :, :, 2] = z_spectrum_holder_2d

    reflect_pi_3d[x_idx, :, :] = reflect_pi_2d
    reflect_sigma_3d[x_idx, :, :] = reflect_sigma_2d

    # Move the variables back to the GPU
    cuda_x_field_2d = cuda.to_device(x_field_holder_2d)
    cuda_y_field_2d = cuda.to_device(y_field_holder_2d)
    cuda_z_field_2d = cuda.to_device(z_field_holder_2d)

    cuda_x_spectrum_2d = cuda.to_device(x_spectrum_holder_2d)
    cuda_y_spectrum_2d = cuda.to_device(y_spectrum_holder_2d)
    cuda_z_spectrum_2d = cuda.to_device(z_spectrum_holder_2d)

    cuda_reflect_sigma_2d = cuda.to_device(reflect_sigma_2d)
    cuda_reflect_pi_2d = cuda.to_device(reflect_pi_2d)

    # Get the time
    toc = time.time()
    print("It takes {:.2f} seconds to finish one 2D slice.".format(toc - tic))

# Move the arrays back to the device for debugging.
cuda_source_point.to_host()
cuda_coef.to_host()

cuda_remain_path.to_host()
cuda_spectrum.to_host()
cuda_intersect.to_host()
cuda_phase.to_host()
cuda_time_response_total.to_host()

cuda_kin_grid.to_host()
cuda_kout_grid.to_host()
cuda_klen_grid.to_host()

cuda_reflect_sigma.to_host()
cuda_reflect_total_sigma.to_host()
cuda_time_response.to_host()
cuda_time_response_total.to_host()

cuda_spectrum_x.to_host()
cuda_spectrum_y.to_host()
cuda_spectrum_z.to_host()
cuda_spectrum_vec.to_host()

cuda_x_field.to_host()
cuda_y_field.to_host()
cuda_z_field.to_host()
toc_tot = time.time()
print("It takes {:.2f} seconds to finish one 3D simulation.".format(toc_tot - tic_tot))
############################################################################################################
# ----------------------------------------------------------------------------------------------------------
#
#                           Step 4: Save the calculation data to h5 files
#
# ----------------------------------------------------------------------------------------------------------
############################################################################################################


# -------------------------------------------------------------------------------
#               Save the data
# -------------------------------------------------------------------------------
# Post process
real_space_efield_vec = np.fft.ifft(np.fft.ifft(efield_3d, axis=0), axis=1)
real_space_efield_vec = np.fft.ifftshift(real_space_efield_vec, axes=(0, 1))

tic = time.time()
with h5.File("./{}_result.h5".format(pre_tag), 'w') as h5file:
    # Save the raw data
    h5file.create_dataset("electric field", data=real_space_efield_vec,
                          chunks=True, compression="gzip")
    h5file.create_dataset("electric field spectrum", data=efield_spectrum_3d,
                          chunks=True, compression="gzip")
    h5file.create_dataset("reflectivity pi", data=reflect_pi_3d,
                          chunks=True, compression="gzip")
    h5file.create_dataset("reflectivity sigma", data=reflect_sigma_3d,
                          chunks=True, compression="gzip")

with h5.File("./{}_y_field.h5".format(pre_tag), 'w') as h5file:
    # Save the raw data
    h5file.create_dataset("field", data=real_space_efield_vec[:, :, :, 1],
                          chunks=True, compression="gzip")

toc = time.time()
print("It takes {:.2f} seconds to save the data to the h5 file.".format(toc - tic))

# Change the stdout back
sys.stdout = stdout
