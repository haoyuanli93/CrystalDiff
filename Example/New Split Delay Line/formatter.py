# Set the range of the index to save
z_idx_range = 500
num1 = 450
num2 = 50
d_num = 512

# -------------------------------------------------------------
#            Get Field for Branch 1
# -------------------------------------------------------------
(my_pulse_tmp,
 crystal_list_1_tmp,
 crystal_list_2_tmp,
 grating_list_tmp,
 obvservation_tmp,
 rot_mat_dict) = lclsutil.get_split_delay_output_frame_refined(kin=my_pulse.k0,
                                                               kout=kout_brunch_1[-2],
                                                               aux=np.array([1., 0., 0.], dtype=np.float64),
                                                               displacement=-np.copy(intersect_brunch_1[-2]),
                                                               observation=copy.deepcopy(observation),
                                                               pulse=copy.deepcopy(my_pulse),
                                                               crystal_list_1=copy.deepcopy(crystal_list_1),
                                                               crystal_list_2=copy.deepcopy(crystal_list_2),
                                                               grating_pair=copy.deepcopy(grating_list))

grating_list[0].set_order(1)
grating_list[1].set_order(-1)

tic = time.time()

(result_3d_dict_1,
 result_2d_dict_1,
 check_dict_1) = groutine.get_single_branch_split_delay_field_before_grating2(grating_pair=grating_list_tmp,
                                                                              channel_cuts=crystal_list_1_tmp,
                                                                              total_path=total_path,
                                                                              observation=observation_tmp,
                                                                              my_pulse=my_pulse_tmp,
                                                                              kx_grid=kx_grid,
                                                                              ky_grid=ky_grid,
                                                                              kz_grid=kz_grid,
                                                                              delay_time=delay_time,
                                                                              number_x=number_x,
                                                                              number_y=number_y,
                                                                              number_z=number_z,
                                                                              z_idx_range=z_idx_range,
                                                                              num1=num1,
                                                                              num2=num2,
                                                                              d_num=512)

toc = time.time()
print("It takes {:.2f} seconds to get the field for branch 1.".format(toc - tic))

# -------------------------------------------------------------
#            Get Field for Branch 2
# -------------------------------------------------------------
(my_pulse_tmp,
 crystal_list_1_tmp,
 crystal_list_2_tmp,
 grating_list_tmp,
 obvservation_tmp,
 rot_mat_dict) = lclsutil.get_split_delay_output_frame_refined(kin=my_pulse.k0,
                                                               kout=kout_brunch_2[-2],
                                                               aux=np.array([1., 0., 0.], dtype=np.float64),
                                                               displacement=-np.copy(intersect_brunch_2[-2]),
                                                               observation=copy.deepcopy(observation),
                                                               pulse=copy.deepcopy(my_pulse),
                                                               crystal_list_1=copy.deepcopy(crystal_list_1),
                                                               crystal_list_2=copy.deepcopy(crystal_list_2),
                                                               grating_pair=copy.deepcopy(grating_list))

grating_list[0].set_order(-1)
grating_list[1].set_order(1)

tic = time.time()

(result_3d_dict_2,
 result_2d_dict_2,
 check_dict_2) = groutine.get_single_branch_split_delay_field_before_grating2(grating_pair=grating_list_tmp,
                                                                              channel_cuts=crystal_list_2_tmp,
                                                                              total_path=total_path,
                                                                              observation=observation_tmp,
                                                                              my_pulse=my_pulse_tmp,
                                                                              kx_grid=kx_grid,
                                                                              ky_grid=ky_grid,
                                                                              kz_grid=kz_grid,
                                                                              delay_time=delay_time,
                                                                              number_x=number_x,
                                                                              number_y=number_y,
                                                                              number_z=number_z,
                                                                              z_idx_range=z_idx_range,
                                                                              num1=num1,
                                                                              num2=num2,
                                                                              d_num=512)

toc = time.time()
print("It takes {:.2f} seconds to get the field for branch 2.".format(toc - tic))
