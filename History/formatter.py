import numpy as np


def get_kout_with_misalignment(dtheta, my_pulse,
                               misalign_grating_1=(0., 0., 0.,),
                               misalign_grating_2=(0., 0., 0.,),
                               misalign_branch_1_crystal_1=(0., 0., 0.),
                               misalign_branch_1_crystal_2=(0., 0., 0.),
                               misalign_branch_2_crystal_1=(0., 0., 0.),
                               misalign_branch_2_crystal_2=(0., 0., 0.),
                               ):
    """
    All the misalignment arguments have a unit of radiant.
    These misalignment are represented with eular angle 'xyz'. I believe that this means that
    the full rotation matrix is constructed by x_rot_mat * y_rot_mat * z_rot_mat.

    However, that should not matter too much, since in this function, there is only one
    rotation around one axis.

    :param dtheta:
    :param my_pulse:
    :param misalign_grating_1:
    :param misalign_grating_2:
    :param misalign_branch_1_crystal_1:
    :param misalign_branch_1_crystal_2:
    :param misalign_branch_2_crystal_1:
    :param misalign_branch_2_crystal_2:
    :return:
    """
    # ----------------------------------------------------------------------------------------------------------
    #                       Step 2: Split
    # ----------------------------------------------------------------------------------------------------------
    grating_list = [crystal.RectangleGrating(), crystal.RectangleGrating()]

    grating_list[0].set_a(util.get_grating_period(dtheta=dtheta, klen_in=my_pulse.klen0) / 2)
    grating_list[0].set_b(util.get_grating_period(dtheta=dtheta, klen_in=my_pulse.klen0) / 2)
    grating_list[0].rotate_around_surface_point(eular_angle=misalign_grating_1)

    grating_list[1].set_a(util.get_grating_period(dtheta=dtheta, klen_in=my_pulse.klen0) / 2)
    grating_list[1].set_b(util.get_grating_period(dtheta=dtheta, klen_in=my_pulse.klen0) / 2)
    grating_list[1].rotate_around_surface_point(eular_angle=misalign_grating_2)

    # ----------------------------------------------------------------------------------------------------------
    #                       Step 3: Delay Lines
    # ----------------------------------------------------------------------------------------------------------
    # Some meta data for the delay line.
    reflect_num = 4
    h_length = 2. * np.pi / (1.9201 * 1e-4)
    hlen_vals = np.array([h_length, ] * reflect_num)

    # Some crystal properties
    chi0 = complex(-0.97631E-05, 0.14871E-06)
    chih_sigma = complex(0.59310E-05, -0.14320E-06)
    chihbar_sigma = complex(0.59310E-05, -0.14320E-06)
    chih_pi = complex(0.46945E-05, -0.11201E-06)
    chihbar_pi = complex(0.46945E-05, -0.11201E-06)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #                       Crystal for branch  1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set up the angles
    angle_offset_1 = dtheta
    bragg_angle_1 = np.radians(18.836) + 13e-6

    branch_angle_1 = lclsutil.get_delay_line_angles(angle_offset=angle_offset_1,
                                                    theta=bragg_angle_1 + np.pi / 2.,
                                                    rho=bragg_angle_1 - np.pi / 2.,
                                                    inclined_angle=np.radians(10.))
    surface_points_1 = np.zeros((reflect_num, 3), dtype=np.float64)

    # Set the misalignment angle

    # Initialize the crystals
    crystal_list_1 = lclsutil.get_crystal_list_delay_branch(hlen_vals=hlen_vals,
                                                            theta_vals=branch_angle_1[0],
                                                            rho_vals=branch_angle_1[1],
                                                            tau_vals=branch_angle_1[2],
                                                            surface_points=surface_points_1,
                                                            chi0=chi0,
                                                            chih_sigma=chih_sigma, chihbar_sigma=chihbar_sigma,
                                                            chih_pi=chih_pi, chihbar_pi=chihbar_pi,
                                                            misalign_1=misalign_branch_1_crystal_1,
                                                            misalign_2=misalign_branch_1_crystal_2)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #                       Crystal for branch  2
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Set up the angles
    angle_offset_2 = - dtheta
    bragg_angle_2 = np.radians(18.836) + 13e-6

    branch_angle_2 = lclsutil.get_delay_line_angles(angle_offset=angle_offset_2,
                                                    theta=1.5 * np.pi - bragg_angle_2,
                                                    rho=0.5 * np.pi - bragg_angle_2,
                                                    inclined_angle=np.radians(0.))
    surface_points_2 = np.zeros((reflect_num, 3), dtype=np.float64)

    # Initialize the crystals
    crystal_list_2 = lclsutil.get_crystal_list_delay_branch(hlen_vals=hlen_vals,
                                                            theta_vals=branch_angle_2[0],
                                                            rho_vals=branch_angle_2[1],
                                                            tau_vals=branch_angle_2[2],
                                                            surface_points=surface_points_2,
                                                            chi0=chi0,
                                                            chih_sigma=chih_sigma, chihbar_sigma=chihbar_sigma,
                                                            chih_pi=chih_pi, chihbar_pi=chihbar_pi,
                                                            misalign_1=misalign_branch_2_crystal_1,
                                                            misalign_2=misalign_branch_2_crystal_2)

    # ------------------------------------------------------
    #   Define the positions
    # ------------------------------------------------------
    path_list_1 = [5e6 - 10e4, 1e5, 5e5, 1.05e5, 6e6, 1e6]
    path_list_2 = [5e6, 1e5, 1e5, 1.05e5, 6e6, 1e6]
    delay_time = 800.

    # Adjust the path sections
    (path_list_1,
     path_list_2) = lclsutil.get_split_delay_configuration(delay_time=delay_time,
                                                           fix_branch_path=path_list_2,
                                                           var_branch_path=path_list_1,
                                                           fix_branch_crystal=crystal_list_2,
                                                           var_branch_crystal=crystal_list_1,
                                                           grating_pair=grating_list,
                                                           pulse_obj=my_pulse)
    (intersect_brunch_1,
     kout_brunch_1,
     intersect_brunch_2,
     kout_brunch_2) = lclsutil.get_light_path(pulse_obj=my_pulse,
                                              grating_list=grating_list,
                                              crystal_list_1=crystal_list_1,
                                              path_list_1=path_list_1,
                                              crystal_list_2=crystal_list_2,
                                              path_list_2=path_list_2)

    return intersect_brunch_1, kout_brunch_1, intersect_brunch_2, kout_brunch_2