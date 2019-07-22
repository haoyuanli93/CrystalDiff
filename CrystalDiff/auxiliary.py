import numpy as np
from CrystalDiff import util
from CrystalDiff import crystal

"""
This module contains the most high-level functions. None of the
other modules can depend on this module.
"""


################################################################################
#   2019/7/6   New and more general functions
################################################################################
def get_crystal_list_lcls2(num, hlen_vals, theta_vals, rho_vals, tau_vals, surface_points,
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


################################################################################
#   2019/6/29   New and more general functions
################################################################################
def get_crystal_list_bk(num, hlen_vals, rho_vals, theta_vals, tau_vals, surface_points,
                        chi0, chih_sigma, chih_pi, chihbar_sigma, chihbar_pi):
    # First get the list
    crystal_list = [crystal.CrystalBlock3D() for _ in range(num)]

    # ----------------------------------------------
    # Set h vectors, surface normal and surface position
    # For simplicity, we always assumes that the h vector is in the x-y plane of the device frame
    # ----------------------------------------------
    for idx in range(num):
        my_crystal = crystal_list[idx]

        my_crystal.set_thickness(1e6)
        my_crystal.set_h(np.array([hlen_vals[idx] * np.cos(rho_vals[idx]),
                                   hlen_vals[idx] * np.sin(rho_vals[idx]), 0.]))
        normal = np.array([np.cos(tau_vals[idx]) * np.cos(theta_vals[idx]),
                           np.cos(tau_vals[idx]) * np.sin(theta_vals[idx]),
                           np.sin(tau_vals[idx])])
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


def get_intersection_point(kin_vec, path_sections, crystal_list):
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
        diffract_info = util.derive_reflect_wavevec_and_rhoh_epsilon(k0_array=k0_vec,
                                                                     k_array=k_len,
                                                                     reciprocal_lattice=crystal_list[idx].h,
                                                                     z=crystal_list[idx].normal)
        k0_vec = diffract_info["kh_array"]

        # Get the reflected momentum
        kout_list[idx] = k0_vec[0]

    return intersect_list[1:], kout_list


def get_to_kout_frame_lcls2(kin, kout, h, displacement, obvservation, pulse, crystal_list):
    # Get the rotation matrix
    rot_mat_dict = util.get_rot_mat_dict_lcls_2(kin=kin, kout=kout, h_vec=h, aux=np.array([1., 0., 0.]))

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


def get_klen_and_angular_mesh(k_num, theta_num, phi_num, energy_range, theta_range, phi_range):
    # Get the corresponding energy mesh
    energy_array = np.linspace(start=energy_range[0], stop=energy_range[1], num=k_num)
    # Get the k grid
    klen_grid = np.ascontiguousarray(util.kev_to_wave_number(energy=energy_array))

    # Get theta grid
    theta_grid = np.linspace(start=theta_range[0], stop=theta_range[1], num=theta_num)

    # Get phi grid
    phi_grid = np.linspace(start=phi_range[0], stop=phi_range[1], num=phi_num)

    info_dict = {"energy_array": energy_array,
                 "klen_array": klen_grid,
                 "theta_array": theta_grid,
                 "phi_array": phi_grid}
    return info_dict


def get_k_mesh_3d(number_x, number_y, number_z, delta_e_x, delta_e_y, delta_e_z):
    # Get the corresponding energy mesh
    energy_array_x = np.linspace(start=- delta_e_x,
                                 stop=+ delta_e_x,
                                 num=number_x)
    energy_array_y = np.linspace(start=- delta_e_y,
                                 stop=+ delta_e_y,
                                 num=number_y)
    energy_array_z = np.linspace(start=- delta_e_z,
                                 stop=+ delta_e_z,
                                 num=number_z)

    # Get the k grid
    kx_grid = np.ascontiguousarray(util.kev_to_wave_number(energy=energy_array_x))
    ky_grid = np.ascontiguousarray(util.kev_to_wave_number(energy=energy_array_y))
    kz_grid = np.ascontiguousarray(util.kev_to_wave_number(energy=energy_array_z))

    # Get the spatial mesh along x axis
    dkx = util.kev_to_wave_number(energy=energy_array_x[1] - energy_array_x[0])
    x_range = np.pi * 2 / dkx

    x_idx = np.linspace(start=-x_range / 2., stop=x_range / 2., num=number_x)
    x_idx_tick = ["{:.2f}".format(x) for x in x_idx]

    # Get the spatial mesh along y axis
    dky = util.kev_to_wave_number(energy=energy_array_y[1] - energy_array_y[0])
    y_range = np.pi * 2 / dky

    y_idx = np.linspace(start=-y_range / 2., stop=y_range / 2., num=number_y)
    y_idx_tick = ["{:.2f}".format(x) for x in y_idx]

    # Get the spatial mesh along z axis
    dkz = util.kev_to_wave_number(energy=energy_array_z[1] - energy_array_z[0])
    z_range = np.pi * 2 / dkz

    z_idx = np.linspace(start=-z_range / 2., stop=z_range / 2., num=number_z)
    z_idx_tick = ["{:.2f}".format(x) for x in z_idx]

    # Assemble the indexes and labels
    axis_info = {"x_range": x_range,
                 "x_idx": x_idx,
                 "x_idx_tick": x_idx_tick,
                 "dkx": dkx,
                 "energy_array_x": energy_array_x,

                 "y_range": y_range,
                 "y_idx": y_idx,
                 "y_idx_tick": y_idx_tick,
                 "dky": dky,
                 "energy_array_y": energy_array_y,

                 "z_range": z_range,
                 "z_idx": z_idx,
                 "z_idx_tick": z_idx_tick,
                 "dkz": dkz,
                 "energy_array_z": energy_array_z,
                 "z_time_idx": np.divide(z_idx, util.c),
                 "z_time_tick": ["{:.2f}".format(x) for x in np.divide(z_idx, util.c)],

                 "de_u_in_meV": np.linspace(start=- delta_e_x * 1e6,
                                            stop=+ delta_e_x * 1e6,
                                            num=number_x)}
    return kx_grid, ky_grid, kz_grid, axis_info


def get_k_mesh_3d_backup(number_u, number_v, number_w, delta_e_u, delta_e_v, delta_e_w, energy_center):
    # Get the corresponding energy mesh
    energy_array_x = np.linspace(start=- delta_e_u + energy_center,
                                 stop=+ delta_e_u + energy_center,
                                 num=number_u)
    energy_array_y = np.linspace(start=- delta_e_v,
                                 stop=+ delta_e_v,
                                 num=number_v)
    energy_array_z = np.linspace(start=- delta_e_w,
                                 stop=+ delta_e_w,
                                 num=number_w)

    # Get the k grid
    kx_grid = np.ascontiguousarray(util.kev_to_wave_number(energy=energy_array_x))
    ky_grid = np.ascontiguousarray(util.kev_to_wave_number(energy=energy_array_y))
    kz_grid = np.ascontiguousarray(util.kev_to_wave_number(energy=energy_array_z))

    # Get the spatial mesh along x axis
    dkx = util.kev_to_wave_number(energy=energy_array_x[1] - energy_array_x[0])
    x_range = np.pi * 2 / dkx

    x_idx = np.linspace(start=0., stop=x_range, num=number_u)
    x_idx_tick = ["{:.2f}".format(x) for x in x_idx]

    # Get the spatial mesh along y axis
    dky = util.kev_to_wave_number(energy=energy_array_y[1] - energy_array_y[0])
    y_range = np.pi * 2 / dky

    y_idx = np.linspace(start=-y_range / 2., stop=y_range / 2., num=number_v)
    y_idx_tick = ["{:.2f}".format(x) for x in y_idx]

    # Get the spatial mesh along z axis
    dkz = util.kev_to_wave_number(energy=energy_array_z[1] - energy_array_z[0])
    z_range = np.pi * 2 / dkz

    z_idx = np.linspace(start=-z_range / 2., stop=z_range / 2., num=number_w)
    z_idx_tick = ["{:.2f}".format(x) for x in z_idx]

    # Assemble the indexes and labels
    axis_info = {"x_range": x_range,
                 "x_idx": x_idx,
                 "x_idx_tick": x_idx_tick,
                 "dkx": dkx,
                 "energy_array_x": energy_array_x,

                 "y_range": y_range,
                 "y_idx": y_idx,
                 "y_idx_tick": y_idx_tick,
                 "dky": dky,
                 "energy_array_y": energy_array_y,

                 "z_range": z_range,
                 "z_idx": z_idx,
                 "z_idx_tick": z_idx_tick,
                 "dkz": dkz,
                 "energy_array_z": energy_array_z,

                 "de_u_in_meV": np.linspace(start=- delta_e_u * 1e6,
                                            stop=+ delta_e_u * 1e6,
                                            num=number_u)}
    return kx_grid, ky_grid, kz_grid, axis_info


################################################################################
#   Previous versions
################################################################################
def prepare_coplane_delay_line(h_len, rho_vals, theta_vals, surface_points,
                               chi0, chih_sigma, chih_pi, chihbar_sigma, chihbar_pi):
    """
    The angles are defined according to the slides.

    :param h_len:
    :param rho_vals:
    :param theta_vals:
    :param surface_points:
    :param chi0:
    :param chih_sigma:
    :param chih_pi:
    :param chihbar_sigma:
    :param chihbar_pi:
    :return:
    """

    # ----------------------------------------------
    # Create crystal
    # ----------------------------------------------
    my_crystal_1 = crystal.CrystalBlock3D()
    my_crystal_2 = crystal.CrystalBlock3D()
    my_crystal_3 = crystal.CrystalBlock3D()
    my_crystal_4 = crystal.CrystalBlock3D()

    # ----------------------------------------------
    # Set h vectors, surface normal and surface position
    # ----------------------------------------------
    my_crystal_1.set_thickness(1e6)
    my_crystal_1.set_h(np.array([h_len * np.cos(rho_vals[0]), h_len * np.sin(rho_vals[0]), 0.]))
    my_crystal_1.set_surface_normal(np.array([np.cos(theta_vals[0]), -np.sin(theta_vals[0]), 0.]))
    my_crystal_1.set_surface_position(surface_points[0])

    my_crystal_2.set_thickness(1e6)
    my_crystal_2.set_h(np.array([h_len * np.cos(rho_vals[1]), - h_len * np.sin(rho_vals[1]), 0.]))
    my_crystal_2.set_surface_normal(np.array([np.cos(theta_vals[1]), np.sin(theta_vals[1]), 0.]))
    my_crystal_2.set_surface_position(surface_points[1])

    my_crystal_3.set_thickness(1e6)
    my_crystal_3.set_h(np.array([h_len * np.cos(rho_vals[2]), - h_len * np.sin(rho_vals[2]), 0.]))
    my_crystal_3.set_surface_normal(np.array([np.cos(theta_vals[2]), np.sin(theta_vals[2]), 0.]))
    my_crystal_3.set_surface_position(surface_points[2])

    my_crystal_4.set_thickness(1e6)
    my_crystal_4.set_h(np.array([h_len * np.cos(rho_vals[3]), h_len * np.sin(rho_vals[3]), 0.]))
    my_crystal_4.set_surface_normal(np.array([np.cos(theta_vals[3]), -np.sin(theta_vals[3]), 0.]))
    my_crystal_4.set_surface_position(surface_points[3])

    # ----------------------------------------------
    # Set chi values
    # ----------------------------------------------
    my_crystal_1.set_chi0(chi0)
    my_crystal_1.set_chih_sigma(chih_sigma)
    my_crystal_1.set_chihbar_sigma(chihbar_sigma)
    my_crystal_1.set_chih_pi(chih_pi)
    my_crystal_1.set_chihbar_pi(chihbar_pi)

    my_crystal_2.set_chi0(chi0)
    my_crystal_2.set_chih_sigma(chih_sigma)
    my_crystal_2.set_chihbar_sigma(chihbar_sigma)
    my_crystal_2.set_chih_pi(chih_pi)
    my_crystal_2.set_chihbar_pi(chihbar_pi)

    my_crystal_3.set_chi0(chi0)
    my_crystal_3.set_chih_sigma(chih_sigma)
    my_crystal_3.set_chihbar_sigma(chihbar_sigma)
    my_crystal_3.set_chih_pi(chih_pi)
    my_crystal_3.set_chihbar_pi(chihbar_pi)

    my_crystal_4.set_chi0(chi0)
    my_crystal_4.set_chih_sigma(chih_sigma)
    my_crystal_4.set_chihbar_sigma(chihbar_sigma)
    my_crystal_4.set_chih_pi(chih_pi)
    my_crystal_4.set_chihbar_pi(chihbar_pi)

    return [my_crystal_1, my_crystal_2, my_crystal_3, my_crystal_4]


def prepare_inclined_delay_line(h_len, rho_vals, theta_vals, tau_vals, surface_points,
                                chi0, chih_sigma, chih_pi, chihbar_sigma, chihbar_pi):
    """
    The angles are defined according to the slides.

    :param h_len:
    :param rho_vals:
    :param theta_vals:
    :param tau_vals
    :param surface_points:
    :param chi0:
    :param chih_sigma:
    :param chih_pi:
    :param chihbar_sigma:
    :param chihbar_pi:
    :return:
    """

    # ----------------------------------------------
    # Create crystal
    # ----------------------------------------------
    my_crystal_1 = crystal.CrystalBlock3D()
    my_crystal_2 = crystal.CrystalBlock3D()
    my_crystal_3 = crystal.CrystalBlock3D()
    my_crystal_4 = crystal.CrystalBlock3D()

    # ----------------------------------------------
    # Set h vectors, surface normal and surface position
    # ----------------------------------------------
    my_crystal_1.set_thickness(1e6)
    my_crystal_1.set_h(np.array([h_len * np.cos(rho_vals[0]),
                                 h_len * np.sin(rho_vals[0]), 0.]))
    normal = np.array([np.cos(tau_vals[0]) * np.cos(theta_vals[0]),
                       -np.cos(tau_vals[0]) * np.sin(theta_vals[0]),
                       np.sin(tau_vals[0])])
    my_crystal_1.set_surface_normal(normal)
    my_crystal_1.set_surface_position(surface_points[0])

    my_crystal_2.set_thickness(1e6)
    my_crystal_2.set_h(np.array([h_len * np.cos(rho_vals[1]),
                                 - h_len * np.sin(rho_vals[1]), 0.]))
    normal = np.array([np.cos(tau_vals[1]) * np.cos(theta_vals[1]),
                       np.cos(tau_vals[1]) * np.sin(theta_vals[1]),
                       np.sin(tau_vals[1])])
    my_crystal_2.set_surface_normal(normal)
    my_crystal_2.set_surface_position(surface_points[1])

    my_crystal_3.set_thickness(1e6)
    my_crystal_3.set_h(np.array([h_len * np.cos(rho_vals[2]),
                                 - h_len * np.sin(rho_vals[2]), 0.]))
    normal = np.array([np.cos(tau_vals[2]) * np.cos(theta_vals[2]),
                       np.cos(tau_vals[2]) * np.sin(theta_vals[2]),
                       np.sin(tau_vals[2])])
    my_crystal_3.set_surface_normal(normal)
    my_crystal_3.set_surface_position(surface_points[2])

    my_crystal_4.set_thickness(1e6)
    my_crystal_4.set_h(np.array([h_len * np.cos(rho_vals[3]),
                                 h_len * np.sin(rho_vals[3]), 0.]))
    normal = np.array([np.cos(tau_vals[3]) * np.cos(theta_vals[3]),
                       -np.cos(tau_vals[3]) * np.sin(theta_vals[3]),
                       np.sin(tau_vals[3])])
    my_crystal_4.set_surface_normal(normal)
    my_crystal_4.set_surface_position(surface_points[3])

    # ----------------------------------------------
    # Set chi values
    # ----------------------------------------------
    my_crystal_1.set_chi0(chi0)
    my_crystal_1.set_chih_sigma(chih_sigma)
    my_crystal_1.set_chihbar_sigma(chihbar_sigma)
    my_crystal_1.set_chih_pi(chih_pi)
    my_crystal_1.set_chihbar_pi(chihbar_pi)

    my_crystal_2.set_chi0(chi0)
    my_crystal_2.set_chih_sigma(chih_sigma)
    my_crystal_2.set_chihbar_sigma(chihbar_sigma)
    my_crystal_2.set_chih_pi(chih_pi)
    my_crystal_2.set_chihbar_pi(chihbar_pi)

    my_crystal_3.set_chi0(chi0)
    my_crystal_3.set_chih_sigma(chih_sigma)
    my_crystal_3.set_chihbar_sigma(chihbar_sigma)
    my_crystal_3.set_chih_pi(chih_pi)
    my_crystal_3.set_chihbar_pi(chihbar_pi)

    my_crystal_4.set_chi0(chi0)
    my_crystal_4.set_chih_sigma(chih_sigma)
    my_crystal_4.set_chihbar_sigma(chihbar_sigma)
    my_crystal_4.set_chih_pi(chih_pi)
    my_crystal_4.set_chihbar_pi(chihbar_pi)

    return [my_crystal_1, my_crystal_2, my_crystal_3, my_crystal_4]


def get_delay_line_intersection_point(kin_vec, path_sections, crystal_list):
    kin_vec = kin_vec.reshape((1, 3))
    k_len = np.array([util.l2_norm(kin_vec[0])])
    k_len = k_len.reshape((1,))

    intersection_point_list = [np.zeros(3, dtype=np.float64), ]
    # -----------------
    # Crystal 1
    # -----------------
    # Get the reflected wavevector
    diffract_info_1 = util.derive_reflect_wavevec_and_rhoh_epsilon(k0_array=kin_vec,
                                                                   k_array=k_len,
                                                                   reciprocal_lattice=crystal_list[0].h,
                                                                   z=crystal_list[0].normal)
    kout_vec_1 = diffract_info_1["kh_array"]

    # ------------------
    # Crystal 2
    # ------------------
    # Assume that the diffracted pulse propagates some distance and then came across the second crystal.
    intersect_2 = intersection_point_list[-1] + kout_vec_1[0] * path_sections[0] / util.l2_norm(kout_vec_1[0])
    intersection_point_list.append(intersect_2)

    # Get the reflected wavevector
    diffract_info_2 = util.derive_reflect_wavevec_and_rhoh_epsilon(k0_array=kout_vec_1,
                                                                   k_array=k_len,
                                                                   reciprocal_lattice=crystal_list[1].h,
                                                                   z=crystal_list[1].normal)
    kout_vec_2 = diffract_info_2["kh_array"]

    # ------------------
    # Crystal 3
    # ------------------
    # Assume that the diffracted pulse propagates some distance and then came across the third crystal.
    intersect_3 = intersection_point_list[-1] + kout_vec_2[0] * path_sections[1] / util.l2_norm(kout_vec_2[0])
    intersection_point_list.append(intersect_3)

    # Get the reflected wavevector
    diffract_info_3 = util.derive_reflect_wavevec_and_rhoh_epsilon(k0_array=kout_vec_2,
                                                                   k_array=k_len,
                                                                   reciprocal_lattice=crystal_list[2].h,
                                                                   z=crystal_list[2].normal)
    kout_vec_3 = diffract_info_3["kh_array"]

    # ------------------
    # Crystal 4
    # ------------------
    # Assume that the diffracted pulse propagates some distance and then came across the third crystal.
    intersect_4 = intersection_point_list[-1] + kout_vec_3[0] * path_sections[2] / util.l2_norm(kout_vec_3[0])
    intersection_point_list.append(intersect_4)

    # Get the reflected wavevector
    diffract_info_4 = util.derive_reflect_wavevec_and_rhoh_epsilon(k0_array=kout_vec_3,
                                                                   k_array=k_len,
                                                                   reciprocal_lattice=crystal_list[3].h,
                                                                   z=crystal_list[3].normal)
    kout_vec_4 = diffract_info_4["kh_array"]

    return intersection_point_list, np.vstack([kout_vec_1[0],
                                               kout_vec_2[0],
                                               kout_vec_3[0],
                                               kout_vec_4[0]])


def get_k_mesh(number_u, number_v, delta_e_u, delta_e_v, energy_center):
    energy_array_u = np.linspace(start=- delta_e_u + energy_center,
                                 stop=+ delta_e_u + energy_center,
                                 num=number_u)
    energy_array_v = np.linspace(start=- delta_e_v,
                                 stop=+ delta_e_v,
                                 num=number_v)

    ku_grid = np.ascontiguousarray(util.kev_to_wave_number(energy=energy_array_u))
    kv_grid = np.ascontiguousarray(util.kev_to_wave_number(energy=energy_array_v))

    # Get the t mesh along x axis
    dku = util.kev_to_wave_number(energy=energy_array_u[1] - energy_array_u[0])
    u_range = np.pi * 2 / dku

    u_idx = np.linspace(start=0., stop=u_range, num=number_u)
    u_idx_tick = ["{:.2f}".format(x) for x in u_idx]

    # Get the t mesh along z axis
    dkz = util.kev_to_wave_number(energy=energy_array_v[1] - energy_array_v[0])
    v_range = np.pi * 2 / dkz

    v_idx = np.linspace(start=-v_range / 2., stop=v_range / 2., num=number_v)
    v_idx_tick = ["{:.2f}".format(x) for x in v_idx]

    # Assemble the indexes and labels
    axis_info = {"u_idx": u_idx,
                 "u_idx_tick": u_idx_tick,
                 "dku": dku,
                 "dkv": dkz,
                 "v_idx": v_idx,
                 "v_idx_tick": v_idx_tick,
                 "u_range": u_range,
                 "v_range": v_range,
                 "energy_array_u": energy_array_u,
                 "energy_array_v": energy_array_v,
                 "de_u_in_meV": np.linspace(start=- delta_e_u * 1e6,
                                            stop=+ delta_e_u * 1e6,
                                            num=number_u)}
    return ku_grid, kv_grid, axis_info
