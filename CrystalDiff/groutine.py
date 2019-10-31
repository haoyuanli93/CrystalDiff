import time

import numpy as np
from numba import cuda
from pyculib import fft as cufft

from CrystalDiff import gfun


def get_split_delay_single_branch_field_shear_fix(grating_pair, channel_cuts, shear_fix_crystals,
                                                  total_path, observation,
                                                  my_pulse, pulse_delay_time, pulse_k0_final,
                                                  grating_orders,
                                                  kx_grid, ky_grid, kz_grid,
                                                  number_x, number_y, number_z,
                                                  z_idx_range, num1, num2, d_num=512):
    """

    :param grating_pair:
    :param channel_cuts:
    :param shear_fix_crystals: The crystals used to fix the shearing.
    :param total_path:
    :param observation:
    :param my_pulse:
    :param pulse_delay_time:
    :param pulse_k0_final: The output wave vector of the central wave length of the incident pulse.
    :param grating_orders:
    :param kx_grid:
    :param ky_grid:
    :param kz_grid:
    :param number_x:
    :param number_y:
    :param number_z:
    :param z_idx_range:
    :param num1:
    :param num2:
    :param d_num:
    :return:
    """
    # Get the inital points for the fft data collection
    idx_start_1 = number_z - num1
    idx_start_2 = 0

    tic = time.time()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #  [3D Blocks]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    efield_3d = np.zeros((number_x, number_y, z_idx_range, 3), dtype=np.complex128)
    efield_spec_3d = np.zeros((number_x, number_y, z_idx_range, 3), dtype=np.complex128)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #  [2D slices]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    x_field_2d = np.ascontiguousarray(np.zeros((number_y, z_idx_range), dtype=np.complex128))
    y_field_2d = np.ascontiguousarray(np.zeros((number_y, z_idx_range), dtype=np.complex128))
    z_field_2d = np.ascontiguousarray(np.zeros((number_y, z_idx_range), dtype=np.complex128))
    x_spec_2d = np.ascontiguousarray(np.zeros((number_y, z_idx_range), dtype=np.complex128))
    y_spec_2d = np.ascontiguousarray(np.zeros((number_y, z_idx_range), dtype=np.complex128))
    z_spec_2d = np.ascontiguousarray(np.zeros((number_y, z_idx_range), dtype=np.complex128))

    cuda_x_field_2d = cuda.to_device(x_field_2d)
    cuda_y_field_2d = cuda.to_device(y_field_2d)
    cuda_z_field_2d = cuda.to_device(z_field_2d)
    cuda_x_spec_2d = cuda.to_device(x_spec_2d)
    cuda_y_spec_2d = cuda.to_device(y_spec_2d)
    cuda_z_spec_2d = cuda.to_device(z_spec_2d)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #  [1D slices] Various intersection points, path length and phase
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    intersect_points = np.ascontiguousarray(np.zeros((number_z, 3), dtype=np.float64))
    component_final_points = np.ascontiguousarray(np.zeros((number_z, 3), dtype=np.float64))
    remaining_length = np.ascontiguousarray(np.zeros(number_z, dtype=np.float64))
    phase_grid = np.ascontiguousarray(np.ones(number_z, dtype=np.complex128))
    jacob_grid = np.ascontiguousarray(np.ones(number_z, dtype=np.float64))

    cuda_intersect = cuda.to_device(intersect_points)
    cuda_final_points = cuda.to_device(component_final_points)
    cuda_remain_path = cuda.to_device(remaining_length)
    cuda_phase = cuda.to_device(phase_grid)
    cuda_jacob = cuda.to_device(jacob_grid)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # [1D slices] reflect and time response
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    reflect_pi = np.ascontiguousarray(np.ones(number_z, dtype=np.complex128))
    reflect_total_pi = np.ascontiguousarray(np.ones(number_z, dtype=np.complex128))
    reflect_sigma = np.ascontiguousarray(np.ones(number_z, dtype=np.complex128))
    reflect_total_sigma = np.ascontiguousarray(np.ones(number_z, dtype=np.complex128))

    cuda_reflect_pi = cuda.to_device(reflect_pi)
    cuda_reflect_total_pi = cuda.to_device(reflect_total_pi)
    cuda_reflect_sigma = cuda.to_device(reflect_sigma)
    cuda_reflect_total_sigma = cuda.to_device(reflect_total_sigma)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # [1D slices] Vector field
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # The reciprocal space
    coef_grid = np.ascontiguousarray(np.zeros(number_z, dtype=np.complex128))  # Input spectrum without Jacobian
    scalar_spec_holder = np.ascontiguousarray(np.ones(number_z, dtype=np.complex128))  # With Jacobian
    vector_spec_holder = np.ascontiguousarray(np.zeros((number_z, 3), dtype=np.complex128))
    x_spec_holder = np.ascontiguousarray(np.zeros(number_z, dtype=np.complex128))
    y_spec_holder = np.ascontiguousarray(np.zeros(number_z, dtype=np.complex128))
    z_spec_holder = np.ascontiguousarray(np.zeros(number_z, dtype=np.complex128))
    x_field_holder = np.ascontiguousarray(np.zeros(number_z, dtype=np.complex128))
    y_field_holder = np.ascontiguousarray(np.zeros(number_z, dtype=np.complex128))
    z_field_holder = np.ascontiguousarray(np.zeros(number_z, dtype=np.complex128))

    cuda_coef = cuda.to_device(coef_grid)
    cuda_spec_scalar = cuda.to_device(scalar_spec_holder)
    cuda_spec_vec = cuda.to_device(vector_spec_holder)
    cuda_spec_x = cuda.to_device(x_spec_holder)
    cuda_spec_y = cuda.to_device(y_spec_holder)
    cuda_spec_z = cuda.to_device(z_spec_holder)
    cuda_x_field = cuda.to_device(x_field_holder)
    cuda_y_field = cuda.to_device(y_field_holder)
    cuda_z_field = cuda.to_device(z_field_holder)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #  [1D slices] k grid
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    kin_grid = np.ascontiguousarray(np.zeros((number_z, 3), dtype=np.float64))
    klen_grid = np.ascontiguousarray(np.zeros(number_z, dtype=np.float64))

    kz_grid = np.ascontiguousarray(kz_grid)
    kz_square = np.ascontiguousarray(np.square(kz_grid))

    cuda_kin_grid = cuda.to_device(kin_grid)
    cuda_klen_grid = cuda.to_device(klen_grid)
    cuda_kz_grid = cuda.to_device(kz_grid)
    cuda_kz_square = cuda.to_device(kz_square)

    toc = time.time()
    print("It takes {:.2f} seconds to prepare the variables.".format(toc - tic))
    ############################################################################################################
    # ----------------------------------------------------------------------------------------------------------
    #
    #                           Step 3: Calculate the field and save the data
    #
    # ----------------------------------------------------------------------------------------------------------
    ############################################################################################################
    # d_num = 512
    b_num = (number_z + d_num - 1) // d_num

    for x_idx in range(number_x):
        kx = kx_grid[x_idx]

        for y_idx in range(number_y):
            ky = ky_grid[y_idx]

            # --------------------------------------------------------------------
            #  Step 1. Get k_out mesh
            # --------------------------------------------------------------------
            gfun.init_kvec[b_num, d_num](cuda_kin_grid,
                                         cuda_klen_grid,
                                         cuda_kz_grid,
                                         cuda_kz_square,
                                         kx,
                                         ky,
                                         ky ** 2 + kx ** 2,
                                         number_z)

            gfun.init_jacobian[b_num, d_num](cuda_jacob, number_z)
            gfun.init_scalar_grid[b_num, d_num](cuda_remain_path, total_path, number_z)
            gfun.init_vector_grid[b_num, d_num](cuda_intersect, my_pulse.x0, 3, number_z)

            # --------------------------------------------------------------------
            #  Step 2. Back propagate from the last grating.
            #          Notice that here, the momentum mesh is not deformed.
            # --------------------------------------------------------------------
            # Here, no deformation of the momentum lattice.
            gfun.add_vector[b_num, d_num](cuda_kin_grid,
                                          cuda_kin_grid,
                                          - grating_orders[1] * grating_pair[1].base_wave_vector,
                                          number_z)

            # Update the wave number
            gfun.get_vector_length[b_num, d_num](cuda_klen_grid,
                                                 cuda_kin_grid,
                                                 3,
                                                 number_z)

            # --------------------------------------------------------------------
            #  Step 3. Get k_in mesh and the jacobian
            # --------------------------------------------------------------------
            for crystal_idx in [3, 2, 1, 0]:
                # Calculate the incident wave vector
                gfun.get_kin_and_jacobian[b_num, d_num](cuda_kin_grid,
                                                        cuda_jacob,
                                                        cuda_klen_grid,
                                                        cuda_kin_grid,
                                                        channel_cuts[crystal_idx].h,
                                                        channel_cuts[crystal_idx].normal,
                                                        channel_cuts[crystal_idx].dot_hn,
                                                        channel_cuts[crystal_idx].h_square,
                                                        number_z)
            # """
            # --------------------------------------------------------------------
            #  Step 4. Back propagate from the First grating
            # --------------------------------------------------------------------
            gfun.add_vector[b_num, d_num](cuda_kin_grid,
                                          cuda_kin_grid,
                                          - grating_orders[0] * grating_pair[0].base_wave_vector,
                                          number_z)

            # Update the wave number
            gfun.get_vector_length[b_num, d_num](cuda_klen_grid,
                                                 cuda_kin_grid,
                                                 3,
                                                 number_z)

            # --------------------------------------------------------------------
            #  Step 5. Back propagate through the shearing fix crystals
            # --------------------------------------------------------------------
            for crystal_idx in [3, 2, 1, 0]:
                # Calculate the incident wave vector
                gfun.get_kin_and_jacobian[b_num, d_num](cuda_kin_grid,
                                                        cuda_jacob,
                                                        cuda_klen_grid,
                                                        cuda_kin_grid,
                                                        shear_fix_crystals[crystal_idx].h,
                                                        shear_fix_crystals[crystal_idx].normal,
                                                        shear_fix_crystals[crystal_idx].dot_hn,
                                                        shear_fix_crystals[crystal_idx].h_square,
                                                        number_z)

            # --------------------------------------------------------------------
            #  Step 6. Get the coefficient of each monochromatic component
            # --------------------------------------------------------------------
            # Calculate the corresponding coefficient in the incident pulse
            gfun.get_gaussian_pulse_spectrum[b_num, d_num](cuda_coef,
                                                           cuda_kin_grid,
                                                           float(pulse_delay_time),
                                                           my_pulse.sigma_mat,
                                                           my_pulse.scaling,
                                                           np.zeros(3, dtype=np.float64),
                                                           my_pulse.k0,
                                                           my_pulse.omega0,
                                                           my_pulse.n,
                                                           number_z)

            # --------------------------------------------------------------------
            #  Step 7. Calculate the Jacobian weighted vector spectrum
            # --------------------------------------------------------------------
            # Add Jacobian
            gfun.scalar_scalar_multiply_complex[b_num, d_num](cuda_coef,
                                                              cuda_jacob,
                                                              cuda_spec_scalar,
                                                              number_z
                                                              )

            # Get the vector field
            gfun.scalar_vector_multiply_complex[b_num, d_num](cuda_spec_scalar,
                                                              my_pulse.polar,
                                                              cuda_spec_vec,
                                                              number_z)

            # --------------------------------------------------------------------
            #  Step 8. Forward propagation
            # --------------------------------------------------------------------

            # Diffracted by the shearing fixed crystals
            for crystal_idx in range(4):
                # Get the intersection point from the previous intersection point
                gfun.get_intersection_point[b_num, d_num](cuda_remain_path,
                                                          cuda_intersect,
                                                          cuda_kin_grid,
                                                          cuda_klen_grid,
                                                          cuda_remain_path,
                                                          cuda_intersect,
                                                          shear_fix_crystals[crystal_idx].surface_point,
                                                          shear_fix_crystals[crystal_idx].normal,
                                                          number_z)

                # Get the reflectivity
                gfun.get_bragg_reflection[b_num, d_num](cuda_reflect_sigma,
                                                        cuda_reflect_pi,
                                                        cuda_kin_grid,
                                                        cuda_spec_vec,
                                                        cuda_klen_grid,
                                                        cuda_kin_grid,
                                                        shear_fix_crystals[crystal_idx].d,
                                                        shear_fix_crystals[crystal_idx].h,
                                                        shear_fix_crystals[crystal_idx].normal,
                                                        shear_fix_crystals[crystal_idx].dot_hn,
                                                        shear_fix_crystals[crystal_idx].h_square,
                                                        shear_fix_crystals[crystal_idx].h_len,
                                                        shear_fix_crystals[crystal_idx].chi0,
                                                        shear_fix_crystals[crystal_idx].chih_sigma,
                                                        shear_fix_crystals[crystal_idx].chihbar_sigma,
                                                        shear_fix_crystals[crystal_idx].chih_pi,
                                                        shear_fix_crystals[crystal_idx].chihbar_pi,
                                                        number_z)
                gfun.scalar_scalar_multiply_complex[b_num, d_num](cuda_reflect_sigma,
                                                                  cuda_reflect_total_sigma,
                                                                  cuda_reflect_total_sigma,
                                                                  number_z)
                gfun.scalar_scalar_multiply_complex[b_num, d_num](cuda_reflect_pi,
                                                                  cuda_reflect_total_pi,
                                                                  cuda_reflect_total_pi,
                                                                  number_z)

            # Get the intersection point on the first grating from the initial point
            gfun.get_intersection_point[b_num, d_num](cuda_remain_path,
                                                      cuda_intersect,
                                                      cuda_kin_grid,
                                                      cuda_klen_grid,
                                                      cuda_remain_path,
                                                      cuda_intersect,
                                                      grating_pair[0].surface_point,
                                                      grating_pair[0].normal,
                                                      number_z)

            # Diffracted by the first grating
            gfun.get_square_grating_effect_non_zero[b_num, d_num](cuda_kin_grid,
                                                                  cuda_spec_vec,
                                                                  cuda_klen_grid,
                                                                  cuda_kin_grid,
                                                                  grating_pair[0].h,
                                                                  grating_pair[0].n,
                                                                  grating_pair[0].ab_ratio,
                                                                  grating_orders[0],
                                                                  grating_pair[0].base_wave_vector,
                                                                  number_z)

            # Diffracted by the delay lines
            for crystal_idx in range(4):
                # Get the intersection point from the previous intersection point
                gfun.get_intersection_point[b_num, d_num](cuda_remain_path,
                                                          cuda_intersect,
                                                          cuda_kin_grid,
                                                          cuda_klen_grid,
                                                          cuda_remain_path,
                                                          cuda_intersect,
                                                          channel_cuts[crystal_idx].surface_point,
                                                          channel_cuts[crystal_idx].normal,
                                                          number_z)

                # Get the reflectivity
                gfun.get_bragg_reflection[b_num, d_num](cuda_reflect_sigma,
                                                        cuda_reflect_pi,
                                                        cuda_kin_grid,
                                                        cuda_spec_vec,
                                                        cuda_klen_grid,
                                                        cuda_kin_grid,
                                                        channel_cuts[crystal_idx].d,
                                                        channel_cuts[crystal_idx].h,
                                                        channel_cuts[crystal_idx].normal,
                                                        channel_cuts[crystal_idx].dot_hn,
                                                        channel_cuts[crystal_idx].h_square,
                                                        channel_cuts[crystal_idx].h_len,
                                                        channel_cuts[crystal_idx].chi0,
                                                        channel_cuts[crystal_idx].chih_sigma,
                                                        channel_cuts[crystal_idx].chihbar_sigma,
                                                        channel_cuts[crystal_idx].chih_pi,
                                                        channel_cuts[crystal_idx].chihbar_pi,
                                                        number_z)
                gfun.scalar_scalar_multiply_complex[b_num, d_num](cuda_reflect_sigma,
                                                                  cuda_reflect_total_sigma,
                                                                  cuda_reflect_total_sigma,
                                                                  number_z)
                gfun.scalar_scalar_multiply_complex[b_num, d_num](cuda_reflect_pi,
                                                                  cuda_reflect_total_pi,
                                                                  cuda_reflect_total_pi,
                                                                  number_z)

            # Get the intersection point on the second grating from the previous intersection point
            gfun.get_intersection_point[b_num, d_num](cuda_remain_path,
                                                      cuda_intersect,
                                                      cuda_kin_grid,
                                                      cuda_klen_grid,
                                                      cuda_remain_path,
                                                      cuda_intersect,
                                                      grating_pair[1].surface_point,
                                                      grating_pair[1].normal,
                                                      number_z)

            # Diffracted by the second grating
            gfun.get_square_grating_effect_non_zero[b_num, d_num](cuda_kin_grid,
                                                                  cuda_spec_vec,
                                                                  cuda_klen_grid,
                                                                  cuda_kin_grid,
                                                                  grating_pair[1].h,
                                                                  grating_pair[1].n,
                                                                  grating_pair[1].ab_ratio,
                                                                  grating_orders[1],
                                                                  grating_pair[1].base_wave_vector,
                                                                  number_z)
            # --------------------------------------------------------------------
            #  Step 8. Get the propagation phase
            # --------------------------------------------------------------------
            gfun.get_final_point[b_num, d_num](cuda_final_points,
                                               cuda_intersect,
                                               cuda_kin_grid,
                                               cuda_klen_grid,
                                               cuda_remain_path,
                                               number_z)

            # Get the propagational phase from the inital phase.
            gfun.get_relative_spatial_phase[b_num, d_num](cuda_phase,
                                                          cuda_final_points,
                                                          observation,
                                                          cuda_kin_grid,
                                                          pulse_k0_final,
                                                          number_z)

            # Add the phase
            gfun.scalar_vector_elementwise_multiply_complex[b_num, d_num](cuda_phase,
                                                                          cuda_spec_vec,
                                                                          cuda_spec_vec,
                                                                          number_z)

            # --------------------------------------------------------------------
            #  Step 9. Goes from the reciprocal space to the real space
            # --------------------------------------------------------------------
            # Save the result to the total reflect
            gfun.vector_expansion[b_num, d_num](cuda_spec_vec,
                                                cuda_spec_x,
                                                cuda_spec_y,
                                                cuda_spec_z,
                                                number_z)
            # Save the spec of the field
            gfun.fill_column_complex_fftshift[b_num, d_num](cuda_x_spec_2d,
                                                            cuda_spec_x,
                                                            y_idx,
                                                            idx_start_1,
                                                            num1,
                                                            idx_start_2,
                                                            num2)
            gfun.fill_column_complex_fftshift[b_num, d_num](cuda_y_spec_2d,
                                                            cuda_spec_y,
                                                            y_idx,
                                                            idx_start_1,
                                                            num1,
                                                            idx_start_2,
                                                            num2)
            gfun.fill_column_complex_fftshift[b_num, d_num](cuda_z_spec_2d,
                                                            cuda_spec_z,
                                                            y_idx,
                                                            idx_start_1,
                                                            num1,
                                                            idx_start_2,
                                                            num2)

            # Take the fourier transformation
            cufft.ifft(cuda_spec_x, cuda_x_field)
            cufft.ifft(cuda_spec_y, cuda_y_field)
            cufft.ifft(cuda_spec_z, cuda_z_field)

            # Update the data holder
            gfun.fill_column_complex_fftshift[b_num, d_num](cuda_x_field_2d,
                                                            cuda_x_field,
                                                            y_idx,
                                                            idx_start_1,
                                                            num1,
                                                            idx_start_2,
                                                            num2)

            gfun.fill_column_complex_fftshift[b_num, d_num](cuda_y_field_2d,
                                                            cuda_y_field,
                                                            y_idx,
                                                            idx_start_1,
                                                            num1,
                                                            idx_start_2,
                                                            num2)

            gfun.fill_column_complex_fftshift[b_num, d_num](cuda_z_field_2d,
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
        cuda_x_spec_2d.to_host()
        cuda_y_spec_2d.to_host()
        cuda_z_spec_2d.to_host()

        # Update the 3D variables.
        efield_3d[x_idx, :, :, 0] = x_field_2d
        efield_3d[x_idx, :, :, 1] = y_field_2d
        efield_3d[x_idx, :, :, 2] = z_field_2d
        efield_spec_3d[x_idx, :, :, 0] = x_spec_2d
        efield_spec_3d[x_idx, :, :, 1] = y_spec_2d
        efield_spec_3d[x_idx, :, :, 2] = z_spec_2d

        # Move the variables back to the GPU
        cuda_x_field_2d = cuda.to_device(x_field_2d)
        cuda_y_field_2d = cuda.to_device(y_field_2d)
        cuda_z_field_2d = cuda.to_device(z_field_2d)
        cuda_x_spec_2d = cuda.to_device(x_spec_2d)
        cuda_y_spec_2d = cuda.to_device(y_spec_2d)
        cuda_z_spec_2d = cuda.to_device(z_spec_2d)

    # Move the arrays back to the device for debugging.
    cuda_final_points.to_host()
    cuda_remain_path.to_host()
    cuda_spec_scalar.to_host()
    cuda_intersect.to_host()
    cuda_phase.to_host()

    cuda_kin_grid.to_host()
    cuda_klen_grid.to_host()

    cuda_reflect_pi.to_host()
    cuda_reflect_sigma.to_host()
    cuda_reflect_total_pi.to_host()
    cuda_reflect_total_sigma.to_host()

    cuda_coef.to_host()
    cuda_spec_x.to_host()
    cuda_spec_y.to_host()
    cuda_spec_z.to_host()
    cuda_spec_vec.to_host()

    cuda_x_field.to_host()
    cuda_y_field.to_host()
    cuda_z_field.to_host()

    # Create result dictionary

    check_dict = {"intersect_points": intersect_points,
                  "component_final_points": component_final_points,
                  "remaining_length": remaining_length,
                  "phase_grid": phase_grid,
                  "jacob_grid": jacob_grid,
                  "reflectivity_pi": reflect_pi,
                  "reflectivity_sigma": reflect_sigma,
                  "reflectivity_pi_tot": reflect_total_pi,
                  "reflectivity_sigma_tot": reflect_total_sigma
                  }

    result_3d_dict = {"efield_3d": efield_3d,
                      "efield_spec_3d": efield_spec_3d}

    result_2d_dict = {"x_field_2d": x_field_2d,
                      "y_field_2d": y_field_2d,
                      "z_field_2d": z_field_2d,
                      "x_spec_2d": x_spec_2d,
                      "y_spec_2d": y_spec_2d,
                      "z_spec_2d": z_spec_2d}

    return result_3d_dict, result_2d_dict, check_dict


def get_split_delay_single_branch_field(grating_pair, channel_cuts,
                                        total_path, observation,
                                        my_pulse, pulse_delay_time, pulse_k0_final,
                                        grating_orders,
                                        kx_grid, ky_grid, kz_grid,
                                        number_x, number_y, number_z,
                                        z_idx_range, num1, num2, d_num=512):
    """

    :param grating_pair:
    :param channel_cuts:
    :param total_path:
    :param observation:
    :param my_pulse:
    :param pulse_delay_time:
    :param pulse_k0_final: The output wave vector of the central wave length of the incident pulse.
    :param grating_orders:
    :param kx_grid:
    :param ky_grid:
    :param kz_grid:
    :param number_x:
    :param number_y:
    :param number_z:
    :param z_idx_range:
    :param num1:
    :param num2:
    :param d_num:
    :return:
    """
    # Get the inital points for the fft data collection
    idx_start_1 = number_z - num1
    idx_start_2 = 0

    tic = time.time()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #  [3D Blocks]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    efield_3d = np.zeros((number_x, number_y, z_idx_range, 3), dtype=np.complex128)
    efield_spec_3d = np.zeros((number_x, number_y, z_idx_range, 3), dtype=np.complex128)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #  [2D slices]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    x_field_2d = np.ascontiguousarray(np.zeros((number_y, z_idx_range), dtype=np.complex128))
    y_field_2d = np.ascontiguousarray(np.zeros((number_y, z_idx_range), dtype=np.complex128))
    z_field_2d = np.ascontiguousarray(np.zeros((number_y, z_idx_range), dtype=np.complex128))
    x_spec_2d = np.ascontiguousarray(np.zeros((number_y, z_idx_range), dtype=np.complex128))
    y_spec_2d = np.ascontiguousarray(np.zeros((number_y, z_idx_range), dtype=np.complex128))
    z_spec_2d = np.ascontiguousarray(np.zeros((number_y, z_idx_range), dtype=np.complex128))

    cuda_x_field_2d = cuda.to_device(x_field_2d)
    cuda_y_field_2d = cuda.to_device(y_field_2d)
    cuda_z_field_2d = cuda.to_device(z_field_2d)
    cuda_x_spec_2d = cuda.to_device(x_spec_2d)
    cuda_y_spec_2d = cuda.to_device(y_spec_2d)
    cuda_z_spec_2d = cuda.to_device(z_spec_2d)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #  [1D slices] Various intersection points, path length and phase
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    intersect_points = np.ascontiguousarray(np.zeros((number_z, 3), dtype=np.float64))
    component_final_points = np.ascontiguousarray(np.zeros((number_z, 3), dtype=np.float64))
    remaining_length = np.ascontiguousarray(np.zeros(number_z, dtype=np.float64))
    phase_grid = np.ascontiguousarray(np.ones(number_z, dtype=np.complex128))
    jacob_grid = np.ascontiguousarray(np.ones(number_z, dtype=np.float64))

    cuda_intersect = cuda.to_device(intersect_points)
    cuda_final_points = cuda.to_device(component_final_points)
    cuda_remain_path = cuda.to_device(remaining_length)
    cuda_phase = cuda.to_device(phase_grid)
    cuda_jacob = cuda.to_device(jacob_grid)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # [1D slices] reflect and time response
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    reflect_pi = np.ascontiguousarray(np.ones(number_z, dtype=np.complex128))
    reflect_total_pi = np.ascontiguousarray(np.ones(number_z, dtype=np.complex128))
    reflect_sigma = np.ascontiguousarray(np.ones(number_z, dtype=np.complex128))
    reflect_total_sigma = np.ascontiguousarray(np.ones(number_z, dtype=np.complex128))

    cuda_reflect_pi = cuda.to_device(reflect_pi)
    cuda_reflect_total_pi = cuda.to_device(reflect_total_pi)
    cuda_reflect_sigma = cuda.to_device(reflect_sigma)
    cuda_reflect_total_sigma = cuda.to_device(reflect_total_sigma)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # [1D slices] Vector field
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # The reciprocal space
    coef_grid = np.ascontiguousarray(np.zeros(number_z, dtype=np.complex128))  # Input spectrum without Jacobian
    scalar_spec_holder = np.ascontiguousarray(np.ones(number_z, dtype=np.complex128))  # With Jacobian
    vector_spec_holder = np.ascontiguousarray(np.zeros((number_z, 3), dtype=np.complex128))
    x_spec_holder = np.ascontiguousarray(np.zeros(number_z, dtype=np.complex128))
    y_spec_holder = np.ascontiguousarray(np.zeros(number_z, dtype=np.complex128))
    z_spec_holder = np.ascontiguousarray(np.zeros(number_z, dtype=np.complex128))
    x_field_holder = np.ascontiguousarray(np.zeros(number_z, dtype=np.complex128))
    y_field_holder = np.ascontiguousarray(np.zeros(number_z, dtype=np.complex128))
    z_field_holder = np.ascontiguousarray(np.zeros(number_z, dtype=np.complex128))

    cuda_coef = cuda.to_device(coef_grid)
    cuda_spec_scalar = cuda.to_device(scalar_spec_holder)
    cuda_spec_vec = cuda.to_device(vector_spec_holder)
    cuda_spec_x = cuda.to_device(x_spec_holder)
    cuda_spec_y = cuda.to_device(y_spec_holder)
    cuda_spec_z = cuda.to_device(z_spec_holder)
    cuda_x_field = cuda.to_device(x_field_holder)
    cuda_y_field = cuda.to_device(y_field_holder)
    cuda_z_field = cuda.to_device(z_field_holder)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #  [1D slices] k grid
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    kin_grid = np.ascontiguousarray(np.zeros((number_z, 3), dtype=np.float64))
    klen_grid = np.ascontiguousarray(np.zeros(number_z, dtype=np.float64))

    kz_grid = np.ascontiguousarray(kz_grid)
    kz_square = np.ascontiguousarray(np.square(kz_grid))

    cuda_kin_grid = cuda.to_device(kin_grid)
    cuda_klen_grid = cuda.to_device(klen_grid)
    cuda_kz_grid = cuda.to_device(kz_grid)
    cuda_kz_square = cuda.to_device(kz_square)

    toc = time.time()
    print("It takes {:.2f} seconds to prepare the variables.".format(toc - tic))
    ############################################################################################################
    # ----------------------------------------------------------------------------------------------------------
    #
    #                           Step 3: Calculate the field and save the data
    #
    # ----------------------------------------------------------------------------------------------------------
    ############################################################################################################
    # d_num = 512
    b_num = (number_z + d_num - 1) // d_num

    for x_idx in range(number_x):
        kx = kx_grid[x_idx]

        for y_idx in range(number_y):
            ky = ky_grid[y_idx]

            # --------------------------------------------------------------------
            #  Step 1. Get k_out mesh
            # --------------------------------------------------------------------
            gfun.init_kvec[b_num, d_num](cuda_kin_grid,
                                         cuda_klen_grid,
                                         cuda_kz_grid,
                                         cuda_kz_square,
                                         kx,
                                         ky,
                                         ky ** 2 + kx ** 2,
                                         number_z)

            gfun.init_jacobian[b_num, d_num](cuda_jacob, number_z)
            gfun.init_scalar_grid[b_num, d_num](cuda_remain_path, total_path, number_z)
            gfun.init_vector_grid[b_num, d_num](cuda_intersect, my_pulse.x0, 3, number_z)

            # --------------------------------------------------------------------
            #  Step 2. Back propagate from the last grating.
            #          Notice that here, the momentum mesh is not deformed.
            # --------------------------------------------------------------------
            # Here, no deformation of the momentum lattice.
            gfun.add_vector[b_num, d_num](cuda_kin_grid,
                                          cuda_kin_grid,
                                          - grating_orders[1] * grating_pair[1].base_wave_vector,
                                          number_z)

            # Update the wave number
            gfun.get_vector_length[b_num, d_num](cuda_klen_grid,
                                                 cuda_kin_grid,
                                                 3,
                                                 number_z)

            # --------------------------------------------------------------------
            #  Step 3. Get k_in mesh and the jacobian
            # --------------------------------------------------------------------
            for crystal_idx in [3, 2, 1, 0]:
                # Calculate the incident wave vector
                gfun.get_kin_and_jacobian[b_num, d_num](cuda_kin_grid,
                                                        cuda_jacob,
                                                        cuda_klen_grid,
                                                        cuda_kin_grid,
                                                        channel_cuts[crystal_idx].h,
                                                        channel_cuts[crystal_idx].normal,
                                                        channel_cuts[crystal_idx].dot_hn,
                                                        channel_cuts[crystal_idx].h_square,
                                                        number_z)
            # """
            # --------------------------------------------------------------------
            #  Step 4. Back propagate from the First grating
            # --------------------------------------------------------------------
            gfun.add_vector[b_num, d_num](cuda_kin_grid,
                                          cuda_kin_grid,
                                          - grating_orders[0] * grating_pair[0].base_wave_vector,
                                          number_z)

            # Update the wave number
            gfun.get_vector_length[b_num, d_num](cuda_klen_grid,
                                                 cuda_kin_grid,
                                                 3,
                                                 number_z)
            # --------------------------------------------------------------------
            #  Step 5. Get the coefficient of each monochromatic component
            # --------------------------------------------------------------------
            # Calculate the corresponding coefficient in the incident pulse
            gfun.get_gaussian_pulse_spectrum[b_num, d_num](cuda_coef,
                                                           cuda_kin_grid,
                                                           float(pulse_delay_time),
                                                           my_pulse.sigma_mat,
                                                           my_pulse.scaling,
                                                           np.zeros(3, dtype=np.float64),
                                                           my_pulse.k0,
                                                           my_pulse.omega0,
                                                           my_pulse.n,
                                                           number_z)

            # --------------------------------------------------------------------
            #  Step 6. Calculate the Jacobian weighted vector spectrum
            # --------------------------------------------------------------------
            # Add Jacobian
            gfun.scalar_scalar_multiply_complex[b_num, d_num](cuda_coef,
                                                              cuda_jacob,
                                                              cuda_spec_scalar,
                                                              number_z
                                                              )

            # Get the vector field
            gfun.scalar_vector_multiply_complex[b_num, d_num](cuda_spec_scalar,
                                                              my_pulse.polar,
                                                              cuda_spec_vec,
                                                              number_z)

            # --------------------------------------------------------------------
            #  Step 7. Forward propagation
            # --------------------------------------------------------------------
            # Get the intersection point on the first grating from the initial point
            gfun.get_intersection_point[b_num, d_num](cuda_remain_path,
                                                      cuda_intersect,
                                                      cuda_kin_grid,
                                                      cuda_klen_grid,
                                                      cuda_remain_path,
                                                      cuda_intersect,
                                                      grating_pair[0].surface_point,
                                                      grating_pair[0].normal,
                                                      number_z)

            # Diffracted by the first grating
            gfun.get_square_grating_effect_non_zero[b_num, d_num](cuda_kin_grid,
                                                                  cuda_spec_vec,
                                                                  cuda_klen_grid,
                                                                  cuda_kin_grid,
                                                                  grating_pair[0].h,
                                                                  grating_pair[0].n,
                                                                  grating_pair[0].ab_ratio,
                                                                  grating_orders[0],
                                                                  grating_pair[0].base_wave_vector,
                                                                  number_z)

            # Diffracted by the delay lines
            for crystal_idx in range(4):
                # Get the intersection point from the previous intersection point
                gfun.get_intersection_point[b_num, d_num](cuda_remain_path,
                                                          cuda_intersect,
                                                          cuda_kin_grid,
                                                          cuda_klen_grid,
                                                          cuda_remain_path,
                                                          cuda_intersect,
                                                          channel_cuts[crystal_idx].surface_point,
                                                          channel_cuts[crystal_idx].normal,
                                                          number_z)

                # Get the reflectivity
                gfun.get_bragg_reflection[b_num, d_num](cuda_reflect_sigma,
                                                        cuda_reflect_pi,
                                                        cuda_kin_grid,
                                                        cuda_spec_vec,
                                                        cuda_klen_grid,
                                                        cuda_kin_grid,
                                                        channel_cuts[crystal_idx].d,
                                                        channel_cuts[crystal_idx].h,
                                                        channel_cuts[crystal_idx].normal,
                                                        channel_cuts[crystal_idx].dot_hn,
                                                        channel_cuts[crystal_idx].h_square,
                                                        channel_cuts[crystal_idx].h_len,
                                                        channel_cuts[crystal_idx].chi0,
                                                        channel_cuts[crystal_idx].chih_sigma,
                                                        channel_cuts[crystal_idx].chihbar_sigma,
                                                        channel_cuts[crystal_idx].chih_pi,
                                                        channel_cuts[crystal_idx].chihbar_pi,
                                                        number_z)
                gfun.scalar_scalar_multiply_complex[b_num, d_num](cuda_reflect_sigma,
                                                                  cuda_reflect_total_sigma,
                                                                  cuda_reflect_total_sigma,
                                                                  number_z)
                gfun.scalar_scalar_multiply_complex[b_num, d_num](cuda_reflect_pi,
                                                                  cuda_reflect_total_pi,
                                                                  cuda_reflect_total_pi,
                                                                  number_z)

            # Get the intersection point on the second grating from the previous intersection point
            gfun.get_intersection_point[b_num, d_num](cuda_remain_path,
                                                      cuda_intersect,
                                                      cuda_kin_grid,
                                                      cuda_klen_grid,
                                                      cuda_remain_path,
                                                      cuda_intersect,
                                                      grating_pair[1].surface_point,
                                                      grating_pair[1].normal,
                                                      number_z)

            # Diffracted by the second grating
            gfun.get_square_grating_effect_non_zero[b_num, d_num](cuda_kin_grid,
                                                                  cuda_spec_vec,
                                                                  cuda_klen_grid,
                                                                  cuda_kin_grid,
                                                                  grating_pair[1].h,
                                                                  grating_pair[1].n,
                                                                  grating_pair[1].ab_ratio,
                                                                  grating_orders[1],
                                                                  grating_pair[1].base_wave_vector,
                                                                  number_z)
            # --------------------------------------------------------------------
            #  Step 8. Get the propagation phase
            # --------------------------------------------------------------------
            gfun.get_final_point[b_num, d_num](cuda_final_points,
                                               cuda_intersect,
                                               cuda_kin_grid,
                                               cuda_klen_grid,
                                               cuda_remain_path,
                                               number_z)

            # Get the propagational phase from the inital phase.
            gfun.get_relative_spatial_phase[b_num, d_num](cuda_phase,
                                                          cuda_final_points,
                                                          observation,
                                                          cuda_kin_grid,
                                                          pulse_k0_final,
                                                          number_z)

            # Add the phase
            gfun.scalar_vector_elementwise_multiply_complex[b_num, d_num](cuda_phase,
                                                                          cuda_spec_vec,
                                                                          cuda_spec_vec,
                                                                          number_z)

            # --------------------------------------------------------------------
            #  Step 9. Goes from the reciprocal space to the real space
            # --------------------------------------------------------------------
            # Save the result to the total reflect
            gfun.vector_expansion[b_num, d_num](cuda_spec_vec,
                                                cuda_spec_x,
                                                cuda_spec_y,
                                                cuda_spec_z,
                                                number_z)
            # Save the spec of the field
            gfun.fill_column_complex_fftshift[b_num, d_num](cuda_x_spec_2d,
                                                            cuda_spec_x,
                                                            y_idx,
                                                            idx_start_1,
                                                            num1,
                                                            idx_start_2,
                                                            num2)
            gfun.fill_column_complex_fftshift[b_num, d_num](cuda_y_spec_2d,
                                                            cuda_spec_y,
                                                            y_idx,
                                                            idx_start_1,
                                                            num1,
                                                            idx_start_2,
                                                            num2)
            gfun.fill_column_complex_fftshift[b_num, d_num](cuda_z_spec_2d,
                                                            cuda_spec_z,
                                                            y_idx,
                                                            idx_start_1,
                                                            num1,
                                                            idx_start_2,
                                                            num2)

            # Take the fourier transformation
            cufft.ifft(cuda_spec_x, cuda_x_field)
            cufft.ifft(cuda_spec_y, cuda_y_field)
            cufft.ifft(cuda_spec_z, cuda_z_field)

            # Update the data holder
            gfun.fill_column_complex_fftshift[b_num, d_num](cuda_x_field_2d,
                                                            cuda_x_field,
                                                            y_idx,
                                                            idx_start_1,
                                                            num1,
                                                            idx_start_2,
                                                            num2)

            gfun.fill_column_complex_fftshift[b_num, d_num](cuda_y_field_2d,
                                                            cuda_y_field,
                                                            y_idx,
                                                            idx_start_1,
                                                            num1,
                                                            idx_start_2,
                                                            num2)

            gfun.fill_column_complex_fftshift[b_num, d_num](cuda_z_field_2d,
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
        cuda_x_spec_2d.to_host()
        cuda_y_spec_2d.to_host()
        cuda_z_spec_2d.to_host()

        # Update the 3D variables.
        efield_3d[x_idx, :, :, 0] = x_field_2d
        efield_3d[x_idx, :, :, 1] = y_field_2d
        efield_3d[x_idx, :, :, 2] = z_field_2d
        efield_spec_3d[x_idx, :, :, 0] = x_spec_2d
        efield_spec_3d[x_idx, :, :, 1] = y_spec_2d
        efield_spec_3d[x_idx, :, :, 2] = z_spec_2d

        # Move the variables back to the GPU
        cuda_x_field_2d = cuda.to_device(x_field_2d)
        cuda_y_field_2d = cuda.to_device(y_field_2d)
        cuda_z_field_2d = cuda.to_device(z_field_2d)
        cuda_x_spec_2d = cuda.to_device(x_spec_2d)
        cuda_y_spec_2d = cuda.to_device(y_spec_2d)
        cuda_z_spec_2d = cuda.to_device(z_spec_2d)

    # Move the arrays back to the device for debugging.
    cuda_final_points.to_host()
    cuda_remain_path.to_host()
    cuda_spec_scalar.to_host()
    cuda_intersect.to_host()
    cuda_phase.to_host()

    cuda_kin_grid.to_host()
    cuda_klen_grid.to_host()

    cuda_reflect_pi.to_host()
    cuda_reflect_sigma.to_host()
    cuda_reflect_total_pi.to_host()
    cuda_reflect_total_sigma.to_host()

    cuda_coef.to_host()
    cuda_spec_x.to_host()
    cuda_spec_y.to_host()
    cuda_spec_z.to_host()
    cuda_spec_vec.to_host()

    cuda_x_field.to_host()
    cuda_y_field.to_host()
    cuda_z_field.to_host()

    # Create result dictionary

    check_dict = {"intersect_points": intersect_points,
                  "component_final_points": component_final_points,
                  "remaining_length": remaining_length,
                  "phase_grid": phase_grid,
                  "jacob_grid": jacob_grid,
                  "reflectivity_pi": reflect_pi,
                  "reflectivity_sigma": reflect_sigma,
                  "reflectivity_pi_tot": reflect_total_pi,
                  "reflectivity_sigma_tot": reflect_total_sigma
                  }

    result_3d_dict = {"efield_3d": efield_3d,
                      "efield_spec_3d": efield_spec_3d}

    result_2d_dict = {"x_field_2d": x_field_2d,
                      "y_field_2d": y_field_2d,
                      "z_field_2d": z_field_2d,
                      "x_spec_2d": x_spec_2d,
                      "y_spec_2d": y_spec_2d,
                      "z_spec_2d": z_spec_2d}

    return result_3d_dict, result_2d_dict, check_dict


def get_delay_line_field(channel_cuts,
                         total_path, observation,
                         my_pulse, pulse_delay_time, pulse_k0_final,
                         kx_grid, ky_grid, kz_grid,
                         number_x, number_y, number_z,
                         z_idx_range, num1, num2, d_num=512):
    """

    :param channel_cuts:
    :param total_path:
    :param observation:
    :param my_pulse:
    :param pulse_delay_time:
    :param pulse_k0_final: The output wave vector of the central wave length of the incident pulse.
    :param kx_grid:
    :param ky_grid:
    :param kz_grid:
    :param number_x:
    :param number_y:
    :param number_z:
    :param z_idx_range:
    :param num1:
    :param num2:
    :param d_num:
    :return:
    """
    # Get the inital points for the fft data collection
    idx_start_1 = number_z - num1
    idx_start_2 = 0

    tic = time.time()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #  [3D Blocks]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    efield_3d = np.zeros((number_x, number_y, z_idx_range, 3), dtype=np.complex128)
    efield_spec_3d = np.zeros((number_x, number_y, z_idx_range, 3), dtype=np.complex128)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #  [2D slices]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    x_field_2d = np.ascontiguousarray(np.zeros((number_y, z_idx_range), dtype=np.complex128))
    y_field_2d = np.ascontiguousarray(np.zeros((number_y, z_idx_range), dtype=np.complex128))
    z_field_2d = np.ascontiguousarray(np.zeros((number_y, z_idx_range), dtype=np.complex128))
    x_spec_2d = np.ascontiguousarray(np.zeros((number_y, z_idx_range), dtype=np.complex128))
    y_spec_2d = np.ascontiguousarray(np.zeros((number_y, z_idx_range), dtype=np.complex128))
    z_spec_2d = np.ascontiguousarray(np.zeros((number_y, z_idx_range), dtype=np.complex128))

    cuda_x_field_2d = cuda.to_device(x_field_2d)
    cuda_y_field_2d = cuda.to_device(y_field_2d)
    cuda_z_field_2d = cuda.to_device(z_field_2d)
    cuda_x_spec_2d = cuda.to_device(x_spec_2d)
    cuda_y_spec_2d = cuda.to_device(y_spec_2d)
    cuda_z_spec_2d = cuda.to_device(z_spec_2d)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #  [1D slices] Various intersection points, path length and phase
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    intersect_points = np.ascontiguousarray(np.zeros((number_z, 3), dtype=np.float64))
    component_final_points = np.ascontiguousarray(np.zeros((number_z, 3), dtype=np.float64))
    remaining_length = np.ascontiguousarray(np.zeros(number_z, dtype=np.float64))
    phase_grid = np.ascontiguousarray(np.ones(number_z, dtype=np.complex128))
    jacob_grid = np.ascontiguousarray(np.ones(number_z, dtype=np.float64))

    cuda_intersect = cuda.to_device(intersect_points)
    cuda_final_points = cuda.to_device(component_final_points)
    cuda_remain_path = cuda.to_device(remaining_length)
    cuda_phase = cuda.to_device(phase_grid)
    cuda_jacob = cuda.to_device(jacob_grid)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # [1D slices] reflect and time response
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    reflect_pi = np.ascontiguousarray(np.ones(number_z, dtype=np.complex128))
    reflect_total_pi = np.ascontiguousarray(np.ones(number_z, dtype=np.complex128))
    reflect_sigma = np.ascontiguousarray(np.ones(number_z, dtype=np.complex128))
    reflect_total_sigma = np.ascontiguousarray(np.ones(number_z, dtype=np.complex128))

    cuda_reflect_pi = cuda.to_device(reflect_pi)
    cuda_reflect_total_pi = cuda.to_device(reflect_total_pi)
    cuda_reflect_sigma = cuda.to_device(reflect_sigma)
    cuda_reflect_total_sigma = cuda.to_device(reflect_total_sigma)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # [1D slices] Vector field
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # The reciprocal space
    coef_grid = np.ascontiguousarray(np.zeros(number_z, dtype=np.complex128))  # Input spectrum without Jacobian
    scalar_spec_holder = np.ascontiguousarray(np.ones(number_z, dtype=np.complex128))  # With Jacobian
    vector_spec_holder = np.ascontiguousarray(np.zeros((number_z, 3), dtype=np.complex128))
    x_spec_holder = np.ascontiguousarray(np.zeros(number_z, dtype=np.complex128))
    y_spec_holder = np.ascontiguousarray(np.zeros(number_z, dtype=np.complex128))
    z_spec_holder = np.ascontiguousarray(np.zeros(number_z, dtype=np.complex128))
    x_field_holder = np.ascontiguousarray(np.zeros(number_z, dtype=np.complex128))
    y_field_holder = np.ascontiguousarray(np.zeros(number_z, dtype=np.complex128))
    z_field_holder = np.ascontiguousarray(np.zeros(number_z, dtype=np.complex128))

    cuda_coef = cuda.to_device(coef_grid)
    cuda_spec_scalar = cuda.to_device(scalar_spec_holder)
    cuda_spec_vec = cuda.to_device(vector_spec_holder)
    cuda_spec_x = cuda.to_device(x_spec_holder)
    cuda_spec_y = cuda.to_device(y_spec_holder)
    cuda_spec_z = cuda.to_device(z_spec_holder)
    cuda_x_field = cuda.to_device(x_field_holder)
    cuda_y_field = cuda.to_device(y_field_holder)
    cuda_z_field = cuda.to_device(z_field_holder)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #  [1D slices] k grid
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    kin_grid = np.ascontiguousarray(np.zeros((number_z, 3), dtype=np.float64))
    klen_grid = np.ascontiguousarray(np.zeros(number_z, dtype=np.float64))

    kz_grid = np.ascontiguousarray(kz_grid)
    kz_square = np.ascontiguousarray(np.square(kz_grid))

    cuda_kin_grid = cuda.to_device(kin_grid)
    cuda_klen_grid = cuda.to_device(klen_grid)
    cuda_kz_grid = cuda.to_device(kz_grid)
    cuda_kz_square = cuda.to_device(kz_square)

    toc = time.time()
    print("It takes {:.2f} seconds to prepare the variables.".format(toc - tic))
    ############################################################################################################
    # ----------------------------------------------------------------------------------------------------------
    #
    #                           Step 3: Calculate the field and save the data
    #
    # ----------------------------------------------------------------------------------------------------------
    ############################################################################################################
    # d_num = 512
    b_num = (number_z + d_num - 1) // d_num

    for x_idx in range(number_x):
        kx = kx_grid[x_idx]

        for y_idx in range(number_y):
            ky = ky_grid[y_idx]

            # --------------------------------------------------------------------
            #  Step 1. Get k_out mesh
            # --------------------------------------------------------------------
            gfun.init_kvec[b_num, d_num](cuda_kin_grid,
                                         cuda_klen_grid,
                                         cuda_kz_grid,
                                         cuda_kz_square,
                                         kx,
                                         ky,
                                         ky ** 2 + kx ** 2,
                                         number_z)

            gfun.init_jacobian[b_num, d_num](cuda_jacob, number_z)
            gfun.init_scalar_grid[b_num, d_num](cuda_remain_path, total_path, number_z)
            gfun.init_vector_grid[b_num, d_num](cuda_intersect, my_pulse.x0, 3, number_z)

            # --------------------------------------------------------------------
            #  Step 3. Get k_in mesh and the jacobian
            # --------------------------------------------------------------------
            for crystal_idx in [3, 2, 1, 0]:
                # Calculate the incident wave vector
                gfun.get_kin_and_jacobian[b_num, d_num](cuda_kin_grid,
                                                        cuda_jacob,
                                                        cuda_klen_grid,
                                                        cuda_kin_grid,
                                                        channel_cuts[crystal_idx].h,
                                                        channel_cuts[crystal_idx].normal,
                                                        channel_cuts[crystal_idx].dot_hn,
                                                        channel_cuts[crystal_idx].h_square,
                                                        number_z)

            # --------------------------------------------------------------------
            #  Step 5. Get the coefficient of each monochromatic component
            # --------------------------------------------------------------------
            # Calculate the corresponding coefficient in the incident pulse
            gfun.get_gaussian_pulse_spectrum[b_num, d_num](cuda_coef,
                                                           cuda_kin_grid,
                                                           float(pulse_delay_time),
                                                           my_pulse.sigma_mat,
                                                           my_pulse.scaling,
                                                           np.zeros(3, dtype=np.float64),
                                                           my_pulse.k0,
                                                           my_pulse.omega0,
                                                           my_pulse.n,
                                                           number_z)

            # --------------------------------------------------------------------
            #  Step 6. Calculate the Jacobian weighted vector spectrum
            # --------------------------------------------------------------------
            # Add Jacobian
            gfun.scalar_scalar_multiply_complex[b_num, d_num](cuda_coef,
                                                              cuda_jacob,
                                                              cuda_spec_scalar,
                                                              number_z
                                                              )

            # Get the vector field
            gfun.scalar_vector_multiply_complex[b_num, d_num](cuda_spec_scalar,
                                                              my_pulse.polar,
                                                              cuda_spec_vec,
                                                              number_z)

            # --------------------------------------------------------------------
            #  Step 7. Forward propagation
            # --------------------------------------------------------------------
            # Diffracted by the delay lines
            for crystal_idx in range(4):
                # Get the intersection point from the previous intersection point
                gfun.get_intersection_point[b_num, d_num](cuda_remain_path,
                                                          cuda_intersect,
                                                          cuda_kin_grid,
                                                          cuda_klen_grid,
                                                          cuda_remain_path,
                                                          cuda_intersect,
                                                          channel_cuts[crystal_idx].surface_point,
                                                          channel_cuts[crystal_idx].normal,
                                                          number_z)

                # Get the reflectivity
                gfun.get_bragg_reflection[b_num, d_num](cuda_reflect_sigma,
                                                        cuda_reflect_pi,
                                                        cuda_kin_grid,
                                                        cuda_spec_vec,
                                                        cuda_klen_grid,
                                                        cuda_kin_grid,
                                                        channel_cuts[crystal_idx].d,
                                                        channel_cuts[crystal_idx].h,
                                                        channel_cuts[crystal_idx].normal,
                                                        channel_cuts[crystal_idx].dot_hn,
                                                        channel_cuts[crystal_idx].h_square,
                                                        channel_cuts[crystal_idx].h_len,
                                                        channel_cuts[crystal_idx].chi0,
                                                        channel_cuts[crystal_idx].chih_sigma,
                                                        channel_cuts[crystal_idx].chihbar_sigma,
                                                        channel_cuts[crystal_idx].chih_pi,
                                                        channel_cuts[crystal_idx].chihbar_pi,
                                                        number_z)
                gfun.scalar_scalar_multiply_complex[b_num, d_num](cuda_reflect_sigma,
                                                                  cuda_reflect_total_sigma,
                                                                  cuda_reflect_total_sigma,
                                                                  number_z)
                gfun.scalar_scalar_multiply_complex[b_num, d_num](cuda_reflect_pi,
                                                                  cuda_reflect_total_pi,
                                                                  cuda_reflect_total_pi,
                                                                  number_z)

            # --------------------------------------------------------------------
            #  Step 8. Get the propagation phase
            # --------------------------------------------------------------------
            gfun.get_final_point[b_num, d_num](cuda_final_points,
                                               cuda_intersect,
                                               cuda_kin_grid,
                                               cuda_klen_grid,
                                               cuda_remain_path,
                                               number_z)

            # Get the propagational phase from the inital phase.
            gfun.get_relative_spatial_phase[b_num, d_num](cuda_phase,
                                                          cuda_final_points,
                                                          observation,
                                                          cuda_kin_grid,
                                                          pulse_k0_final,
                                                          number_z)

            # Add the phase
            gfun.scalar_vector_elementwise_multiply_complex[b_num, d_num](cuda_phase,
                                                                          cuda_spec_vec,
                                                                          cuda_spec_vec,
                                                                          number_z)

            # --------------------------------------------------------------------
            #  Step 9. Goes from the reciprocal space to the real space
            # --------------------------------------------------------------------
            # Save the result to the total reflect
            gfun.vector_expansion[b_num, d_num](cuda_spec_vec,
                                                cuda_spec_x,
                                                cuda_spec_y,
                                                cuda_spec_z,
                                                number_z)
            # Save the spec of the field
            gfun.fill_column_complex_fftshift[b_num, d_num](cuda_x_spec_2d,
                                                            cuda_spec_x,
                                                            y_idx,
                                                            idx_start_1,
                                                            num1,
                                                            idx_start_2,
                                                            num2)
            gfun.fill_column_complex_fftshift[b_num, d_num](cuda_y_spec_2d,
                                                            cuda_spec_y,
                                                            y_idx,
                                                            idx_start_1,
                                                            num1,
                                                            idx_start_2,
                                                            num2)
            gfun.fill_column_complex_fftshift[b_num, d_num](cuda_z_spec_2d,
                                                            cuda_spec_z,
                                                            y_idx,
                                                            idx_start_1,
                                                            num1,
                                                            idx_start_2,
                                                            num2)

            # Take the fourier transformation
            cufft.ifft(cuda_spec_x, cuda_x_field)
            cufft.ifft(cuda_spec_y, cuda_y_field)
            cufft.ifft(cuda_spec_z, cuda_z_field)

            # Update the data holder
            gfun.fill_column_complex_fftshift[b_num, d_num](cuda_x_field_2d,
                                                            cuda_x_field,
                                                            y_idx,
                                                            idx_start_1,
                                                            num1,
                                                            idx_start_2,
                                                            num2)

            gfun.fill_column_complex_fftshift[b_num, d_num](cuda_y_field_2d,
                                                            cuda_y_field,
                                                            y_idx,
                                                            idx_start_1,
                                                            num1,
                                                            idx_start_2,
                                                            num2)

            gfun.fill_column_complex_fftshift[b_num, d_num](cuda_z_field_2d,
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
        cuda_x_spec_2d.to_host()
        cuda_y_spec_2d.to_host()
        cuda_z_spec_2d.to_host()

        # Update the 3D variables.
        efield_3d[x_idx, :, :, 0] = x_field_2d
        efield_3d[x_idx, :, :, 1] = y_field_2d
        efield_3d[x_idx, :, :, 2] = z_field_2d
        efield_spec_3d[x_idx, :, :, 0] = x_spec_2d
        efield_spec_3d[x_idx, :, :, 1] = y_spec_2d
        efield_spec_3d[x_idx, :, :, 2] = z_spec_2d

        # Move the variables back to the GPU
        cuda_x_field_2d = cuda.to_device(x_field_2d)
        cuda_y_field_2d = cuda.to_device(y_field_2d)
        cuda_z_field_2d = cuda.to_device(z_field_2d)
        cuda_x_spec_2d = cuda.to_device(x_spec_2d)
        cuda_y_spec_2d = cuda.to_device(y_spec_2d)
        cuda_z_spec_2d = cuda.to_device(z_spec_2d)

    # Move the arrays back to the device for debugging.
    cuda_final_points.to_host()
    cuda_remain_path.to_host()
    cuda_spec_scalar.to_host()
    cuda_intersect.to_host()
    cuda_phase.to_host()

    cuda_kin_grid.to_host()
    cuda_klen_grid.to_host()

    cuda_reflect_pi.to_host()
    cuda_reflect_sigma.to_host()
    cuda_reflect_total_pi.to_host()
    cuda_reflect_total_sigma.to_host()

    cuda_coef.to_host()
    cuda_spec_x.to_host()
    cuda_spec_y.to_host()
    cuda_spec_z.to_host()
    cuda_spec_vec.to_host()

    cuda_x_field.to_host()
    cuda_y_field.to_host()
    cuda_z_field.to_host()

    # Create result dictionary

    check_dict = {"intersect_points": intersect_points,
                  "component_final_points": component_final_points,
                  "remaining_length": remaining_length,
                  "phase_grid": phase_grid,
                  "coef_grid": coef_grid,
                  "scalar_spec": scalar_spec_holder,
                  "jacob_grid": jacob_grid,
                  "reflectivity_pi": reflect_pi,
                  "reflectivity_sigma": reflect_sigma,
                  "reflectivity_pi_tot": reflect_total_pi,
                  "reflectivity_sigma_tot": reflect_total_sigma
                  }

    result_3d_dict = {"efield_3d": efield_3d,
                      "efield_spec_3d": efield_spec_3d}

    result_2d_dict = {"x_field_2d": x_field_2d,
                      "y_field_2d": y_field_2d,
                      "z_field_2d": z_field_2d,
                      "x_spec_2d": x_spec_2d,
                      "y_spec_2d": y_spec_2d,
                      "z_spec_2d": z_spec_2d}

    return result_3d_dict, result_2d_dict, check_dict
