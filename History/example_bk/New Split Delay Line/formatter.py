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

# Initialize the crystals
crystal_list_1 = lclsutil.get_crystal_list_delay_branch(num=reflect_num,
                                                        hlen_vals=hlen_vals,
                                                        theta_vals=branch_angle_1[0],
                                                        rho_vals=branch_angle_1[1],
                                                        tau_vals=branch_angle_1[2],
                                                        surface_points=np.copy(intersect_brunch_1[1:5]),
                                                        chi0=chi0,
                                                        chih_sigma=chih_sigma, chihbar_sigma=chihbar_sigma,
                                                        chih_pi=chih_pi, chihbar_pi=chihbar_pi,
                                                        misalign_1=misalign_branch_1_crystal_1,
                                                        misalign_2=misalign_branch_1_crystal_2)
# Initialize the crystals
crystal_list_2 = lclsutil.get_crystal_list_delay_branch(num=reflect_num,
                                                        hlen_vals=hlen_vals,
                                                        theta_vals=branch_angle_2[0],
                                                        rho_vals=branch_angle_2[1],
                                                        tau_vals=branch_angle_2[2],
                                                        surface_points=np.copy(intersect_brunch_2[1:5]),
                                                        chi0=chi0,
                                                        chih_sigma=chih_sigma, chihbar_sigma=chihbar_sigma,
                                                        chih_pi=chih_pi, chihbar_pi=chihbar_pi,
                                                        misalign_1=misalign_branch_2_crystal_1,
                                                        misalign_2=misalign_branch_2_crystal_2)
grating_list[0].set_surface_point(np.copy(intersect_brunch_1[0]))
grating_list[1].set_surface_point(np.copy(intersect_brunch_1[-2]))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get the observation point
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
observation = np.copy(intersect_brunch_2[-1])

total_path = pre_length + util.get_total_path_length(intersection_point_list=intersect_brunch_2)
print("The total propagation length is {:.2f}m.".format(total_path / 1e6))
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                  Change frame
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# """
(my_pulse,
 crystal_list_1,
 crystal_list_2,
 grating_list,
 obvservation) = lclsutil.get_split_delay_output_frame(displacement=-np.copy(intersect_brunch_1[-2]),
                                                       obvservation=observation,
                                                       pulse=my_pulse,
                                                       crystal_list_1=crystal_list_1,
                                                       crystal_list_2=crystal_list_2,
                                                       grating_pair=grating_list)
# """
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                  Get the momentum mesh
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
number_x = 2
number_y = 500
number_z = 10 ** 5
kx_grid, ky_grid, kz_grid, axis_info = lclsutil.get_k_mesh_3d(number_x=number_x,
                                                              number_y=number_y,
                                                              number_z=number_z,
                                                              delta_e_x=1e-50,
                                                              delta_e_y=3e-4,
                                                              delta_e_z=3e-4 / util.c)
kz_grid += my_pulse.klen0

# Apply fft shift
# kx_grid = np.ascontiguousarray(np.fft.fftshift(kx_grid))
kx_grid = np.zeros(1, np.float64)
number_x = 1
ky_grid = np.ascontiguousarray(np.fft.fftshift(ky_grid))
kz_grid = np.ascontiguousarray(np.fft.fftshift(kz_grid))
