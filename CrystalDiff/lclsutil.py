import numpy as np
from CrystalDiff import util
from CrystalDiff import crystal

"""
This module contains the most high-level functions. None of the
other modules can depend on this module.
"""


# --------------------------------------------------------------
#              Setup crystals
# --------------------------------------------------------------
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
        diffract_info = util.get_output_wave_vector(k0_grid=k0_vec,
                                                    k_grid=k_len,
                                                    crystal_h=crystal_list[idx].h,
                                                    z=crystal_list[idx].normal)
        k0_vec = diffract_info["kh_grid"]

        # Get the reflected momentum
        kout_list[idx] = k0_vec[0]

    return intersect_list[1:], kout_list


# --------------------------------------------------------------
#              Change Frame
# --------------------------------------------------------------
def get_to_kout_frame(kin, kout, displacement, obvservation, pulse, crystal_list, aux):
    # Get the rotation matrix
    rot_mat_dict = util.get_rot_mat_dict(kin=kin, kout=kout, aux=aux)

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
