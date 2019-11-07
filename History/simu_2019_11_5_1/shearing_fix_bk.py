import sys
import time

import h5py as h5
import numpy as np

sys.path.append(r"/home/haoyuan/my_repos/CrystalDiff")

from CrystalDiff import util, pulse, lclsutil, crystal, groutine

# Create the h5 file to save the result
file_name = "shear_fix_1fs_{}.h5".format(util.time_stamp())
with h5.File(file_name, 'w'):
    pass

############################################################################################################
############################################################################################################
#                                    Prepare the variables
############################################################################################################
############################################################################################################
# ----------------------------------------------------------------------------------------------------------
#                       Step 1: Pulse
# ----------------------------------------------------------------------------------------------------------

energy_center = 10.
pre_length = 1e6

# Set up the pulse
my_pulse = pulse.GaussianPulse3D()
my_pulse.set_pulse_properties(central_energy=energy_center,
                              polar=[0., 1., 0.],
                              sigma_x=708.2581446128465,
                              sigma_y=708.2581446128465,
                              sigma_z=1.,
                              x0=np.array([0., 0., -pre_length]))

# ----------------------------------------------------------------------------------------------------------
#                       Step 2: Split
# ----------------------------------------------------------------------------------------------------------
dtheta = np.arctan(1.5 * 1e-3 / 5.)  # This is the deviation angle.

grating_list = [crystal.RectangleGrating(), crystal.RectangleGrating()]
grating_list[0].set_a(util.get_grating_period(dtheta=dtheta, klen_in=my_pulse.klen0) / 2)
grating_list[0].set_b(util.get_grating_period(dtheta=dtheta, klen_in=my_pulse.klen0) / 2)

grating_list[1].set_a(util.get_grating_period(dtheta=dtheta, klen_in=my_pulse.klen0) / 2)
grating_list[1].set_b(util.get_grating_period(dtheta=dtheta, klen_in=my_pulse.klen0) / 2)

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
#                       Crystal to fix the shearing
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set up the angles
angle_offset_0 = 0.
bragg_angle_0 = np.radians(18.836) + 13e-6

branch_angle_0 = lclsutil.get_delay_line_angles(angle_offset=angle_offset_0,
                                                theta=bragg_angle_0 + np.pi / 2.,
                                                rho=bragg_angle_0 - np.pi / 2.,
                                                inclined_angle=np.radians(-10.))
surface_points_0 = np.zeros((reflect_num, 3), dtype=np.float64)

# Set the misalignment angle
misalign_branch_0_crystal_1 = [0., 0., 0.]
misalign_branch_0_crystal_2 = [0., 0., 0.]

# Initialize the crystals
shear_fix_crystal = lclsutil.get_crystal_list_delay_branch(hlen_vals=hlen_vals,
                                                           theta_vals=branch_angle_0[0],
                                                           rho_vals=branch_angle_0[1],
                                                           tau_vals=branch_angle_0[2],
                                                           surface_points=surface_points_0,
                                                           chi0=chi0,
                                                           chih_sigma=chih_sigma, chihbar_sigma=chihbar_sigma,
                                                           chih_pi=chih_pi, chihbar_pi=chihbar_pi,
                                                           misalign_1=misalign_branch_0_crystal_1,
                                                           misalign_2=misalign_branch_0_crystal_2)

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
misalign_branch_1_crystal_1 = [0., 0., 0.]
misalign_branch_1_crystal_2 = [0., 0., 0.]

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
                                                inclined_angle=np.radians(-10.))
surface_points_2 = np.zeros((reflect_num, 3), dtype=np.float64)

# Set the misalignment angle
misalign_branch_2_crystal_1 = [0., 0., 0.]
misalign_branch_2_crystal_2 = [0., 0., 0.]

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
path_list_0 = [0., 1e5, 1e5, 1e5]
dist_shear_crystal_grating = 5e5

path_list_1 = [5e6 - 10e4, 1e5, 5e5, 1.05e5, 6e6, 1e6]
path_list_2 = [5e6, 1e5, 1e5, 1.05e5, 6e6, 1e6]
delay_time = 20.

############################################################################################################
############################################################################################################
#                               Tune the position of the crystals
############################################################################################################
############################################################################################################
# Get the positions for the shearing fix branch
(intersect_branch_0,
 kout_branch_0) = lclsutil.get_point_with_definite_path(kin_vec=my_pulse.k0,
                                                        path_sections=path_list_0,
                                                        crystal_list=shear_fix_crystal)

# Adjust the path sections
(path_list_1,
 path_list_2) = lclsutil.get_split_delay_configuration(delay_time=delay_time,
                                                       fix_branch_path=path_list_2,
                                                       var_branch_path=path_list_1,
                                                       fix_branch_crystal=crystal_list_2,
                                                       var_branch_crystal=crystal_list_1,
                                                       grating_pair=grating_list,
                                                       kin=kout_branch_0[-1])

(intersect_branch_1,
 kout_brunch_1,
 intersect_branch_2,
 kout_brunch_2) = lclsutil.get_light_path(kin=kout_branch_0[-1],
                                          grating_list=grating_list,
                                          crystal_list_1=crystal_list_1,
                                          path_list_1=path_list_1,
                                          crystal_list_2=crystal_list_2,
                                          path_list_2=path_list_2)

# """
# Shift the intersection points due to the insertion of the new unit
shift = intersect_branch_0[-1] + dist_shear_crystal_grating * kout_branch_0[-1] / util.l2_norm(kout_branch_0[-1])
intersect_branch_1 += shift[np.newaxis, :]
intersect_branch_2 += shift[np.newaxis, :]

# Initialize the crystals
shear_fix_crystal = lclsutil.get_crystal_list_delay_branch(hlen_vals=hlen_vals,
                                                           theta_vals=branch_angle_0[0],
                                                           rho_vals=branch_angle_0[1],
                                                           tau_vals=branch_angle_0[2],
                                                           surface_points=np.copy(intersect_branch_0),
                                                           chi0=chi0,
                                                           chih_sigma=chih_sigma, chihbar_sigma=chihbar_sigma,
                                                           chih_pi=chih_pi, chihbar_pi=chihbar_pi,
                                                           misalign_1=misalign_branch_0_crystal_1,
                                                           misalign_2=misalign_branch_0_crystal_2)

# Initialize the crystals
crystal_list_1 = lclsutil.get_crystal_list_delay_branch(hlen_vals=hlen_vals,
                                                        theta_vals=branch_angle_1[0],
                                                        rho_vals=branch_angle_1[1],
                                                        tau_vals=branch_angle_1[2],
                                                        surface_points=np.copy(intersect_branch_1[1:5]),
                                                        chi0=chi0,
                                                        chih_sigma=chih_sigma, chihbar_sigma=chihbar_sigma,
                                                        chih_pi=chih_pi, chihbar_pi=chihbar_pi,
                                                        misalign_1=misalign_branch_1_crystal_1,
                                                        misalign_2=misalign_branch_1_crystal_2)
# Initialize the crystals
crystal_list_2 = lclsutil.get_crystal_list_delay_branch(hlen_vals=hlen_vals,
                                                        theta_vals=branch_angle_2[0],
                                                        rho_vals=branch_angle_2[1],
                                                        tau_vals=branch_angle_2[2],
                                                        surface_points=np.copy(intersect_branch_2[1:5]),
                                                        chi0=chi0,
                                                        chih_sigma=chih_sigma, chihbar_sigma=chihbar_sigma,
                                                        chih_pi=chih_pi, chihbar_pi=chihbar_pi,
                                                        misalign_1=misalign_branch_2_crystal_1,
                                                        misalign_2=misalign_branch_2_crystal_2)

grating_list[0].set_surface_point(np.copy(intersect_branch_2[0]))
grating_list[1].set_surface_point(np.copy(intersect_branch_2[-2]))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Get the observation point
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
observation = np.copy(intersect_branch_2[-1])
total_path = pre_length + np.sum(path_list_2) + np.sum(path_list_0) + dist_shear_crystal_grating

print("The total propagation length is {:.2f}m.".format(total_path / 1e6))
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                  Change frame
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
(my_pulse,
 shear_fix_crystal,
 crystal_list_1,
 crystal_list_2,
 grating_list,
 observation
 ) = lclsutil.get_split_delay_fix_shear_output_frame(displacement=-np.copy(intersect_branch_2[-1]),
                                                     observe=observation,
                                                     pulse=my_pulse,
                                                     crystal_fix_shear=shear_fix_crystal,
                                                     crystal_list_1=crystal_list_1,
                                                     crystal_list_2=crystal_list_2,
                                                     grating_pair=grating_list)

#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                  Get the momentum mesh
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
number_x = 250
number_y = 250
number_z = 10 ** 5
kx_grid, ky_grid, kz_grid, axis_info = lclsutil.get_k_mesh_3d(number_x=number_x,
                                                              number_y=number_y,
                                                              number_z=number_z,
                                                              delta_e_x=3e-5,
                                                              delta_e_y=3e-5,
                                                              delta_e_z=1e-3 / util.c)
kz_grid += my_pulse.klen0

# Apply fft shift
kx_grid = np.ascontiguousarray(np.fft.fftshift(kx_grid))
ky_grid = np.ascontiguousarray(np.fft.fftshift(ky_grid))
kz_grid = np.ascontiguousarray(np.fft.fftshift(kz_grid))

############################################################################################################
############################################################################################################
#                               Get the fields
############################################################################################################
############################################################################################################

# Set the range of the index to save
z_idx_range = 300
num1 = 270
num2 = 30
d_num = 512

# -------------------------------------------------------------
#            Get Field for Branch 2
# -------------------------------------------------------------
tic = time.time()

(result_3d_dict,
 result_2d_dict,
 check_dict
 ) = groutine.get_split_delay_single_branch_field_shear_fix(grating_pair=grating_list,
                                                            shear_fix_crystals=shear_fix_crystal,
                                                            channel_cuts=crystal_list_2,
                                                            total_path=total_path,
                                                            observation=observation,
                                                            my_pulse=my_pulse,
                                                            kx_grid=kx_grid,
                                                            ky_grid=ky_grid,
                                                            kz_grid=kz_grid,
                                                            pulse_delay_time=0.,
                                                            pulse_k0_final=np.array([0., 0., my_pulse.klen0]),
                                                            grating_orders=[-1., 1.],
                                                            number_x=number_x,
                                                            number_y=number_y,
                                                            number_z=number_z,
                                                            z_idx_range=z_idx_range,
                                                            num1=num1,
                                                            num2=num2,
                                                            d_num=512)

toc = time.time()
print("It takes {:.2f} seconds to get the field for branch 2.".format(toc - tic))

# Write the results to the h5 file
util.save_branch_result_to_h5file(file_name=file_name,
                                  io_type="r+",
                                  branch_name="fixed_delay",
                                  result_3d_dict=result_3d_dict,
                                  result_2d_dict=result_2d_dict,
                                  check_dict=check_dict)

# -------------------------------------------------------------
#            Get Field for Branch 1
# -------------------------------------------------------------

tic = time.time()

(result_3d_dict,
 result_2d_dict,
 check_dict
 ) = groutine.get_split_delay_single_branch_field_shear_fix(grating_pair=grating_list,
                                                            shear_fix_crystals=shear_fix_crystal,
                                                            channel_cuts=crystal_list_1,
                                                            total_path=total_path,
                                                            observation=observation,
                                                            my_pulse=my_pulse,
                                                            kx_grid=kx_grid,
                                                            ky_grid=ky_grid,
                                                            kz_grid=kz_grid,
                                                            pulse_delay_time=0.,
                                                            pulse_k0_final=np.array([0., 0., my_pulse.klen0]),
                                                            grating_orders=[1., -1.],
                                                            number_x=number_x,
                                                            number_y=number_y,
                                                            number_z=number_z,
                                                            z_idx_range=z_idx_range,
                                                            num1=num1,
                                                            num2=num2,
                                                            d_num=512)

toc = time.time()
print("It takes {:.2f} seconds to get the field for branch 1.".format(toc - tic))

# Write the results to the h5 file
util.save_branch_result_to_h5file(file_name=file_name,
                                  io_type="r+",
                                  branch_name="variable_delay",
                                  result_3d_dict=result_3d_dict,
                                  result_2d_dict=result_2d_dict,
                                  check_dict=check_dict)