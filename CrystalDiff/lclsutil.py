import numpy as np
from scipy.spatial.transform import Rotation

from CrystalDiff import crystal
from CrystalDiff import util

"""
This module contains the most high-level functions. None of the
other modules can depend on this module.
"""


def get_split_delay_configuration(delay_time, fix_branch_path, var_branch_path,
                                  fix_branch_crystal, var_branch_crystal, grating_pair, kin):
    """
    This function automatically change the configurations of the variable branch to
    match the delay time.

    :param delay_time:
    :param fix_branch_path:
    :param var_branch_path:
    :param fix_branch_crystal:
    :param var_branch_crystal:
    :param grating_pair:
    :param kin:
    :return:
    """
    # ---------------------------------------------------------
    # Step 1 : Get the original path
    # ---------------------------------------------------------

    # First check the lenght of the path since it might different
    (intersect_fixed,
     kout_fixed) = get_light_path_branch(kin=kin,
                                         grating_list=grating_pair,
                                         path_list=fix_branch_path,
                                         crystal_list=fix_branch_crystal,
                                         branch=-1)  # By default, -1 corresponds to the fixed branch.

    # ----------------------------------------------------------
    # Step 2 : Tune the final section of the fixed branch
    # to make sure the intersection point is close to the z axis
    # ----------------------------------------------------------

    # Get the new length
    new_length = - (intersect_fixed[-3][0] * kout_fixed[-2][0] +
                    intersect_fixed[-3][1] * kout_fixed[-2][1]) / (kout_fixed[-2][0] ** 2 + kout_fixed[-2][1] ** 2)
    new_length *= util.l2_norm(kout_fixed[-2])

    # Replace the old path lengths with the new one
    fix_branch_path[-2] = new_length

    # ----------------------------------------------------------
    # Step 3 : Tune the path sections of the variable branch
    # ----------------------------------------------------------
    tmp = np.sum(fix_branch_path[:5])
    var_branch_path[-2] = tmp - np.sum(var_branch_path[:4])
    var_branch_path[-1] = fix_branch_path[-1]

    # ----------------------------------------------------------
    # Step 3 : Adjust the path sections to match the delay time.
    # ----------------------------------------------------------
    # Find the momentum information
    (intersect_var,
     kout_var) = get_light_path_branch(kin=kin,
                                       grating_list=grating_pair,
                                       path_list=var_branch_path,
                                       crystal_list=var_branch_crystal,
                                       branch=1)  # By default, -1 corresponds to the fixed branch.

    delay_length = delay_time * util.c
    cos_theta = np.dot(kout_var[1], kout_var[2]) / util.l2_norm(kout_var[1]) / util.l2_norm(kout_var[2])
    delta = delay_length / 2. / (1 - cos_theta)

    # Change the variable path sections with the calculated length change
    var_branch_path[1] += delta
    var_branch_path[3] += delta
    var_branch_path[2] -= 2 * delta * cos_theta

    return var_branch_path, fix_branch_path


def get_split_delay_output_frame(displacement, obvservation, pulse, crystal_list_1, crystal_list_2, grating_pair):
    """
    Go to the output grating position.

    :param displacement:
    :param obvservation:
    :param pulse:
    :param crystal_list_1:
    :param crystal_list_2:
    :param grating_pair:
    :return:
    """
    # ------------------------------
    # Shift the position
    # ------------------------------
    pulse.x0 += displacement

    # Shift the crystal
    for my_crystal in crystal_list_1:
        my_crystal.shift(displacement=displacement)

    # Shift the crystal
    for my_crystal in crystal_list_2:
        my_crystal.shift(displacement=displacement)

    # Shift the observation position
    obvservation += displacement

    # Shift the grating
    for grating in grating_pair:
        grating.shift(displacement=displacement)

    return pulse, crystal_list_1, crystal_list_2, grating_pair, obvservation


def get_split_delay_fix_shear_output_frame(displacement, observe, pulse,
                                           crystal_fix_shear,
                                           crystal_list_1, crystal_list_2, grating_pair):
    """
    Go to the output grating position.

    :param displacement:
    :param observe:
    :param pulse:
    :param crystal_fix_shear:
    :param crystal_list_1:
    :param crystal_list_2:
    :param grating_pair:
    :return:
    """
    # ------------------------------
    # Shift the position
    # ------------------------------
    pulse.x0 += displacement

    # Shift the crystal
    for my_crystal in crystal_fix_shear:
        my_crystal.shift(displacement=displacement)

    # Shift the crystal
    for my_crystal in crystal_list_1:
        my_crystal.shift(displacement=displacement)

    # Shift the crystal
    for my_crystal in crystal_list_2:
        my_crystal.shift(displacement=displacement)

    # Shift the observation position
    observe += displacement

    # Shift the grating
    for grating in grating_pair:
        grating.shift(displacement=displacement)

    return pulse, crystal_fix_shear, crystal_list_1, crystal_list_2, grating_pair, observe


def get_delay_line_output_frame(displacement, observe, pulse, crystal_list):
    """
    Go to the output grating position.

    :param displacement:
    :param observe:
    :param pulse:
    :param crystal_list:
    :return:
    """
    # ------------------------------
    # Shift the position
    # ------------------------------
    pulse.x0 += displacement

    # Shift the crystal
    for my_crystal in crystal_list:
        my_crystal.shift(displacement=displacement)

    # Shift the observation position
    observe += displacement

    return pulse, crystal_list, observe


def get_split_delay_output_frame_refined(kin, kout, aux, displacement,
                                         observation, pulse,
                                         crystal_list_1, crystal_list_2, grating_pair,
                                         ):
    """
    Go to the position before the second grating.
    
    :param kin: 
    :param kout: 
    :param aux: 
    :param displacement: 
    :param observation: 
    :param pulse: 
    :param crystal_list_1: 
    :param crystal_list_2: 
    :param grating_pair: 
    :return: 
    """
    # Get the rotation matrix
    rot_mat_dict = get_rot_mat_dict(kin=kin, kout=kout, aux=aux)

    # ------------------------------
    # Shift the position
    # ------------------------------
    pulse.x0 += displacement

    # Shift the crystal
    for my_crystal in crystal_list_1:
        my_crystal.shift(displacement=displacement)

    # Shift the crystal
    for my_crystal in crystal_list_2:
        my_crystal.shift(displacement=displacement)

    # Shift the observation position
    observation += displacement

    # Shift the grating
    for grating in grating_pair:
        grating.shift(displacement=displacement)

    # ------------------------------
    # Rotate
    # ------------------------------
    # Rotate the pulse
    rot_mat = rot_mat_dict["Out-Pulse to In-Pulse"]
    pulse.sigma_mat = np.dot(rot_mat.T, np.dot(pulse.sigma_mat, rot_mat))

    # Change the polarization
    pulse.polar = np.dot(rot_mat_dict["In-Pulse to Out-Pulse"], pulse.polar)

    # Reference point of the pulse
    pulse.x0 = rot_mat_dict["Device to Out-Pulse"].dot(pulse.x0)

    # The central momentum of the pulse
    pulse.k0 = rot_mat_dict["Device to Out-Pulse"].dot(pulse.k0)
    pulse.n = pulse.k0 / util.l2_norm(pulse.k0)
    pulse.omega0 = util.l2_norm(pulse.k0) * util.c

    # Shift the crystal
    for my_crystal in crystal_list_1:
        my_crystal.rotate(rot_mat_dict["Device to Out-Pulse"])

    # Shift the crystal
    for my_crystal in crystal_list_2:
        my_crystal.rotate(rot_mat_dict["Device to Out-Pulse"])

    # Rotate the grating
    for grating in grating_pair:
        grating.rotate(rot_mat_dict["Device to Out-Pulse"])

    observation = rot_mat_dict["Device to Out-Pulse"].dot(observation)

    return pulse, crystal_list_1, crystal_list_2, grating_pair, observation, rot_mat_dict


def get_delay_line_angles(angle_offset, theta, rho,
                          inclined_angle=0.,
                          asymmetric_angle=0.):
    """
    I need to include deviation parameters with respect to the ideal position.

    :param angle_offset: The is the angle adjusted from the grating diffraction
    :param theta: The geometric bragg angle.
    :param rho: The angle associated with the normal direction
    :param inclined_angle: The angle associated for the inclination. This change the normal direction of
                            the reflection surface angle
    :param asymmetric_angle: The angle associated with the asymmetric reflection. This changes the
                                normal direction of the reflection surface angle.
    :return:
    """
    theta_vals = np.array([theta,
                           np.pi + theta,
                           - theta,
                           np.pi - theta]) + angle_offset

    rho_vals = np.array([rho,
                         np.pi + rho - asymmetric_angle,
                         - rho,
                         np.pi - rho + asymmetric_angle]) + angle_offset

    tau_vals = np.array([0., inclined_angle, - inclined_angle, 0.])

    return theta_vals, rho_vals, tau_vals


# --------------------------------------------------------------
#           Calculate the light path
# --------------------------------------------------------------
def get_light_path(kin, grating_list, crystal_list_1, path_list_1, crystal_list_2, path_list_2):
    """
    In this function, the first branch is the upper branch in the following diagram.
    The second branch is the lower branch in the following diagram.

                |                -            |
                |           -        -        |
                |       -               -     |     This is branch 1
                |   -     +1               -  |
    ------------|                             |---------------
                |   -     -1               -  |
                |       -              -      |     This is branch 2
                |           -       -         |
                |               -             |

    :param kin:
    :param grating_list:
    :param crystal_list_1:
    :param path_list_1:
    :param crystal_list_2:
    :param path_list_2:
    :return:
    """
    (intersect_branch_1,
     kout_branch_1) = get_light_path_branch(kin=kin,
                                            grating_list=grating_list,
                                            path_list=path_list_1,
                                            crystal_list=crystal_list_1,
                                            branch=1)

    (intersect_branch_2,
     kout_branch_2) = get_light_path_branch(kin=kin,
                                            grating_list=grating_list,
                                            path_list=path_list_2,
                                            crystal_list=crystal_list_2,
                                            branch=-1)

    return intersect_branch_1, kout_branch_1, intersect_branch_2, kout_branch_2


def get_light_path_branch(kin, grating_list, path_list, crystal_list, branch=1):
    """
    Get the light path for one of the branch.

    :param kin: The incident wave vector with a shape of (1,3)
    :param grating_list:
    :param path_list:
    :param crystal_list:
    :param branch:
    :return:
    """
    kin = np.reshape(kin, (1, 3))

    # Get kout from the first grating
    kout_tmp_1 = util.get_grating_output_momentum(grating_wavenum=branch * grating_list[0].base_wave_vector,
                                                  k_vec=kin)
    # Get the intersection point with the first crystal
    intersect_1 = path_list[0] * kout_tmp_1[0] / util.l2_norm(kout_tmp_1[0])

    # Get the first intersection point on the first crystal
    path_tmp = np.zeros(4)
    path_tmp[1:] = path_list[1:-2]
    intersection_points, kout_vec_list = get_point_with_definite_path(kin_vec=kout_tmp_1,
                                                                      path_sections=path_tmp,
                                                                      crystal_list=crystal_list)
    intersection_points += intersect_1
    # Calculate the change before the grating
    intersect_2 = intersection_points[-1] + path_list[-2] * kout_vec_list[-1] / util.l2_norm(kout_vec_list[-1])

    # Get the final output momentum
    kout_tmp_2 = util.get_grating_output_momentum(grating_wavenum=-branch * grating_list[1].base_wave_vector,
                                                  k_vec=kout_vec_list[-1])
    intersect_final = intersect_2 + path_list[-1] * kout_tmp_2[0] / util.l2_norm(kout_tmp_2[0])

    # Get branch 1 info
    intersect_branch = np.vstack((np.zeros(3, dtype=np.float64),
                                  intersect_1,
                                  intersection_points[1:],  # The first point is redundant.
                                  intersect_2,
                                  intersect_final))
    kout_branch = np.vstack((kin,
                             kout_tmp_1,
                             kout_vec_list,
                             kout_tmp_2))

    return intersect_branch, kout_branch


# --------------------------------------------------------------
#              Setup crystals
# --------------------------------------------------------------
def get_crystal_list_delay_branch(hlen_vals,
                                  theta_vals, rho_vals, tau_vals,
                                  surface_points,
                                  chi0, chih_sigma, chih_pi, chihbar_sigma, chihbar_pi,
                                  misalign=None):
    """
    This function is designed for the new split-delay specifically.

    :param hlen_vals:
    :param theta_vals:
    :param rho_vals:
    :param tau_vals:
    :param surface_points:
    :param chi0:
    :param chih_sigma:
    :param chih_pi:
    :param chihbar_sigma:
    :param chihbar_pi:
    :param misalign: The misalignment of the first channel-cut crystal.
                        This is a (2,) numpy array with a detype of float64

                        The first value is the rotation along the x axis.
                        The x axis is fixed to the crystal.

                        The second value is the rotation within the diffraction plane.
                        This is the y axis fixed on the crystal

                        The  value is the rotation along the along axis of the crystal.
                        This is the z axis fixed on the crystal

                        One first apply the x axis rotation. Then apply the y axis rotation.
                        In the end the z axis rotation
    :return:
    """
    # First get the list
    crystal_list = [crystal.CrystalBlock3D() for _ in range(4)]

    # ----------------------------------------------
    # Set h vectors, surface normal and surface position
    # For simplicity, we always assumes that the h vector is in the x-y plane of the device frame
    # ----------------------------------------------

    # Process the first channel-cut crystal
    for idx in range(4):

        # Extract the crystal object
        my_crystal = crystal_list[idx]
        my_crystal.set_thickness(1e6)

        # Set the h vector
        h_holder = np.zeros(3, dtype=np.float64)
        h_holder[1] = hlen_vals[idx]

        # Set the normal holder
        normal_holder = np.zeros(3, dtype=np.float64)
        normal_holder[1] = 1.

        # If there is a misalignment, then rotate this holder
        if misalign is not None:
            # Get the rotation matrix
            cc_rot = Rotation.from_euler('xyz', misalign[idx], degrees=False)

            # Rotate the holder
            h_holder = np.dot(cc_rot.as_dcm(), h_holder)
            normal_holder = np.dot(cc_rot.as_dcm(), normal_holder)

        # Apply the geometric configuration for the h vector
        rot_mat = Rotation.from_euler('x', np.pi / 2. - theta_vals[idx])
        my_crystal.set_h(np.dot(rot_mat.as_dcm(), h_holder))

        # Apply the geometric configuration for the normal vector
        rot_mat = Rotation.from_euler("zxy", [-tau_vals[idx],
                                              np.pi / 2 - rho_vals[idx],
                                              0])

        my_crystal.set_surface_normal(rot_mat.as_dcm().dot(normal_holder))

        # Set the surface point
        my_crystal.set_surface_position(surface_points[idx])

        # ----------------------------------------------
        # Set chi values
        # ----------------------------------------------
        my_crystal.set_chi0(chi0)
        my_crystal.set_chih_sigma(chih_sigma)
        my_crystal.set_chihbar_sigma(chihbar_sigma)
        my_crystal.set_chih_pi(chih_pi)
        my_crystal.set_chihbar_pi(chihbar_pi)

    return crystal_list


def get_crystal_list(num, hlen_vals, theta_vals, rho_vals, tau_vals, surface_points,
                     chi0, chih_sigma, chih_pi, chihbar_sigma, chihbar_pi):
    """
    This function is designed for lcls2 specifically.

    :param num:
    :param hlen_vals:
    :param theta_vals:
    :param rho_vals:
    :param tau_vals:
    :param surface_points:
    :param chi0:
    :param chih_sigma:
    :param chih_pi:
    :param chihbar_sigma:
    :param chihbar_pi:
    :return:
    """
    # First get the list
    crystal_list = [crystal.CrystalBlock3D() for _ in range(num)]

    # ----------------------------------------------
    # Set h vectors, surface normal and surface position
    # For simplicity, we always assumes that the h vector is in the x-y plane of the device frame
    # ----------------------------------------------
    for idx in range(num):
        my_crystal = crystal_list[idx]

        my_crystal.set_thickness(1e6)
        my_crystal.set_h(np.array([0.,
                                   hlen_vals[idx] * np.sin(theta_vals[idx]),
                                   hlen_vals[idx] * np.cos(theta_vals[idx])]))
        normal = np.array([np.sin(tau_vals[idx]),
                           np.cos(tau_vals[idx]) * np.sin(rho_vals[idx]),
                           np.cos(tau_vals[idx]) * np.cos(rho_vals[idx])])
        my_crystal.set_surface_normal(normal)
        my_crystal.set_surface_position(surface_points[idx])

        # ----------------------------------------------
        # Set chi values
        # ----------------------------------------------
        my_crystal.set_chi0(chi0)
        my_crystal.set_chih_sigma(chih_sigma)
        my_crystal.set_chihbar_sigma(chihbar_sigma)
        my_crystal.set_chih_pi(chih_pi)
        my_crystal.set_chihbar_pi(chihbar_pi)

    return crystal_list


def get_point_with_definite_path(kin_vec, path_sections, crystal_list):
    # Get k0 information to calculation
    k0_vec = kin_vec.reshape((1, 3))
    k_len = np.array([util.l2_norm(k0_vec[0])])
    k_len = k_len.reshape((1,))

    # Get the number of crystals
    num = len(crystal_list)

    # Prepare holders for the calculation
    intersect_list = np.zeros((num + 1, 3), dtype=np.float64)
    kout_list = np.zeros((num, 3), dtype=np.float64)

    for idx in range(num):
        # Update the intersection point
        intersect_list[idx + 1] = (intersect_list[idx] +
                                   k0_vec[0] * path_sections[idx] / util.l2_norm(k0_vec[0]))

        # Get the reflected wavevector
        diffract_info = util.get_output_wave_vector(k0_grid=k0_vec,
                                                    k_grid=k_len,
                                                    crystal_h=crystal_list[idx].h,
                                                    crystal_normal=crystal_list[idx].normal)
        k0_vec = diffract_info["kh_grid"]

        # Get the reflected momentum
        kout_list[idx] = k0_vec[0]

    return intersect_list[1:], kout_list


def get_intersection_point(kin_vec, init_point, crystal_list):
    """
    Get the array of the diffracted momentum and the intersection momentum

    :param kin_vec:
    :param init_point:
    :param crystal_list:
    :return:
    """
    # Get the reflection number
    reflect_num = len(crystal_list)

    # Create holders
    kout_array = np.zeros((reflect_num, 3), dtype=np.float64)
    intersect_array = np.zeros((reflect_num, 3), dtype=np.float64)

    # Copy the variables to avoid potential modification
    s = np.reshape(init_point, (1, 3))
    k = np.reshape(kin_vec, (1, 3))

    # Loop through all the reflections
    for idx in range(reflect_num):
        # Get the intersection point
        intersect_tmp = util.get_intersection_point(s=s,
                                                    k=k,
                                                    n=crystal_list[idx].normal,
                                                    x0=crystal_list[idx].surface_point)
        intersect_array[idx] = intersect_tmp[0]

        # Get the output momentum
        kout_tmp = util.get_output_wave_vector(k0_grid=k,
                                               k_grid=util.l2_norm_batch(k),
                                               crystal_h=crystal_list[idx].h,
                                               crystal_normal=crystal_list[idx].normal)

        # Update the k and s and kout holder
        kout_array[idx] = kout_tmp["kh_grid"][0]

        k = np.copy(kout_tmp["kh_grid"])
        s = np.copy(intersect_tmp)

    return intersect_array, kout_array


# --------------------------------------------------------------
#              Change Frame
# --------------------------------------------------------------
def get_rot_mat_dict(kin, kout, aux=np.array([1., 0., 0.])):
    """
    Here, we assume that the aux is the x axis and does not change very much.
    :param kin:
    :param kout:
    :param aux:
    :return:
    """
    # Get a holder
    rot_mat_dict = {}

    # -------------------------------------------------
    # Device to the incident pulse
    # -------------------------------------------------
    tmp_z = kin / util.l2_norm(kin)

    tmp_y = np.cross(tmp_z, aux)
    tmp_y /= util.l2_norm(tmp_y)

    tmp_x = np.cross(tmp_y, tmp_z)
    tmp_x /= util.l2_norm(tmp_x)

    # Rotation matrix from the incident pulse to the device
    rot_mat_dict.update({"Device to In-Pulse": np.vstack([tmp_x, tmp_y, tmp_z])})
    rot_mat_dict.update({"In-Pulse to Device": rot_mat_dict["Device to In-Pulse"].T})

    # -------------------------------------------------
    # Device to the output pulse
    # -------------------------------------------------
    # When the input pulse and the output pulse are not along the same direction.
    new_z = kout / util.l2_norm(kout)

    new_y = np.cross(new_z, aux)
    new_y /= util.l2_norm(new_y)

    new_x = np.cross(new_y, new_z)
    new_x /= util.l2_norm(new_x)

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


def get_to_kout_frame(kin, kout, displacement, obvservation, pulse, crystal_list, aux):
    # Get the rotation matrix
    rot_mat_dict = get_rot_mat_dict(kin=kin, kout=kout, aux=aux)

    # ------------------------------
    # Shift the position
    # ------------------------------
    pulse.x0 += displacement

    # Shift the crystal
    for my_crystal in crystal_list:
        my_crystal.shift(displacement=displacement)

    obvservation += displacement
    # ------------------------------
    # Rotate
    # ------------------------------
    # Rotate the pulse
    rot_mat = rot_mat_dict["Out-Pulse to In-Pulse"]
    pulse.sigma_mat = np.dot(rot_mat.T, np.dot(pulse.sigma_mat, rot_mat))

    # Change the polarization
    pulse.polar = np.dot(rot_mat_dict["In-Pulse to Out-Pulse"], pulse.polar)

    # Reference point of the pulse
    pulse.x0 = rot_mat_dict["Device to Out-Pulse"].dot(pulse.x0)

    # The central momentum of the pulse
    pulse.k0 = rot_mat_dict["Device to Out-Pulse"].dot(pulse.k0)
    pulse.n = pulse.k0 / util.l2_norm(pulse.k0)
    pulse.omega0 = util.l2_norm(pulse.k0) * util.c

    # Shift the crystal
    for my_crystal in crystal_list:
        my_crystal.rotate(rot_mat_dict["Device to Out-Pulse"])

    obvservation = rot_mat_dict["Device to Out-Pulse"].dot(obvservation)

    return pulse, crystal_list, obvservation, rot_mat_dict


# --------------------------------------------------------------
#              Get k mesh
# --------------------------------------------------------------
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
    kx_grid = np.ascontiguousarray(util.kev_to_wave_number(energy=energy_grid_x))
    ky_grid = np.ascontiguousarray(util.kev_to_wave_number(energy=energy_grid_y))
    kz_grid = np.ascontiguousarray(util.kev_to_wave_number(energy=energy_grid_z))

    # Get the spatial mesh along x axis
    dkx = util.kev_to_wave_number(energy=energy_grid_x[1] - energy_grid_x[0])
    x_range = np.pi * 2 / dkx

    x_idx = np.linspace(start=-x_range / 2., stop=x_range / 2., num=number_x)
    x_idx_tick = ["{:.2f}".format(x) for x in x_idx]

    # Get the spatial mesh along y axis
    dky = util.kev_to_wave_number(energy=energy_grid_y[1] - energy_grid_y[0])
    y_range = np.pi * 2 / dky

    y_idx = np.linspace(start=-y_range / 2., stop=y_range / 2., num=number_y)
    y_idx_tick = ["{:.2f}".format(x) for x in y_idx]

    # Get the spatial mesh along z axis
    dkz = util.kev_to_wave_number(energy=energy_grid_z[1] - energy_grid_z[0])
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
                 "z_time_idx": np.divide(z_idx, util.c),
                 "z_time_tick": ["{:.2f}".format(x) for x in np.divide(z_idx, util.c)],

                 "de_x_in_meV": np.linspace(start=- delta_e_x * 1e6,
                                            stop=+ delta_e_x * 1e6,
                                            num=number_x)}
    return kx_grid, ky_grid, kz_grid, axis_info


# --------------------------------------------------------------
#              For DuMond Diagram
# --------------------------------------------------------------
def get_klen_and_angular_mesh(k_num, theta_num, phi_num, energy_range, theta_range, phi_range):
    # Get the corresponding energy mesh
    energy_grid = np.linspace(start=energy_range[0], stop=energy_range[1], num=k_num)
    # Get the k grid
    klen_grid = np.ascontiguousarray(util.kev_to_wave_number(energy=energy_grid))

    # Get theta grid
    theta_grid = np.linspace(start=theta_range[0], stop=theta_range[1], num=theta_num)

    # Get phi grid
    phi_grid = np.linspace(start=phi_range[0], stop=phi_range[1], num=phi_num)

    info_dict = {"energy_grid": energy_grid,
                 "klen_grid": klen_grid,
                 "theta_grid": theta_grid,
                 "phi_grid": phi_grid}
    return info_dict
