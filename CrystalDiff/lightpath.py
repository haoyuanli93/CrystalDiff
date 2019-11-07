import numpy as np

from CrystalDiff import util


def adjust_path_length(delay_time,
                       fix_branch_path, fix_branch_crystal,
                       var_branch_path, var_branch_crystal,
                       grating_pair,
                       kin):
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
     kout_fixed) = get_light_path_branch(kin_vec=kin,
                                         grating_list=grating_pair,
                                         path_list=fix_branch_path,
                                         crystal_list=fix_branch_crystal,
                                         g_orders=[-1, 1])  # By default, -1 corresponds to the fixed branch.

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
    path_diff = sum(var_branch_path) - sum(fix_branch_path)
    var_branch_path[-2] -= path_diff

    # ----------------------------------------------------------
    # Step 3 : Adjust the path sections to match the delay time.
    # ----------------------------------------------------------
    # Find the momentum information
    (intersect_var,
     kout_var) = get_light_path_branch(kin_vec=kin,
                                       grating_list=grating_pair,
                                       path_list=var_branch_path,
                                       crystal_list=var_branch_crystal,
                                       g_orders=[1, -1])  # By default, -1 corresponds to the fixed branch.

    delay_length = delay_time * util.c
    cos_theta = np.dot(kout_var[2], kout_var[3]) / util.l2_norm(kout_var[3]) / util.l2_norm(kout_var[2])
    delta = delay_length / 2. / (1 - cos_theta)

    # Change the variable path sections with the calculated length change
    var_branch_path[1] += delta
    var_branch_path[3] += delta
    var_branch_path[2] -= 2 * delta * cos_theta

    # ----------------------------------------------------------
    # Step 4 : Get the corresponding intersection position
    # ----------------------------------------------------------
    (intersect_fixed,
     kout_fixed) = get_light_path_branch(kin_vec=kin,
                                         grating_list=grating_pair,
                                         path_list=fix_branch_path,
                                         crystal_list=fix_branch_crystal,
                                         g_orders=[-1, 1])

    (intersect_var,
     kout_var) = get_light_path_branch(kin_vec=kin,
                                       grating_list=grating_pair,
                                       path_list=var_branch_path,
                                       crystal_list=var_branch_crystal,
                                       g_orders=[1, -1])

    return (fix_branch_path, kout_fixed, intersect_fixed,
            var_branch_path, kout_var, intersect_var)


def adjust_path_length_2cc_vs_2cc(delay_time,
                                  fix_branch_path, fix_branch_crystal,
                                  var_branch_path, var_branch_crystal,
                                  grating_pair,
                                  kin):
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
     kout_fixed) = get_light_path_branch(kin_vec=kin,
                                         grating_list=grating_pair,
                                         path_list=fix_branch_path,
                                         crystal_list=fix_branch_crystal,
                                         g_orders=[-1, 1])  # By default, -1 corresponds to the fixed branch.

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
    path_diff = np.sum(var_branch_path) - np.sum(fix_branch_path)
    var_branch_path[-2] -= path_diff

    # ----------------------------------------------------------
    # Step 3 : Adjust the path sections to match the delay time.
    # ----------------------------------------------------------
    # Find the momentum information
    (intersect_var,
     kout_var) = get_light_path_branch(kin_vec=kin,
                                       grating_list=grating_pair,
                                       path_list=var_branch_path,
                                       crystal_list=var_branch_crystal,
                                       g_orders=[1, -1])  # By default, -1 corresponds to the fixed branch.

    delay_length = delay_time * util.c
    cos_theta = np.dot(kout_var[2], kout_var[3]) / util.l2_norm(kout_var[2]) / util.l2_norm(kout_var[3])
    delta = delay_length / 2. / (1 - cos_theta)

    # Change the variable path sections with the calculated length change
    var_branch_path[1] += delta
    var_branch_path[3] += delta
    var_branch_path[2] -= 2 * delta * cos_theta

    # ----------------------------------------------------------
    # Step 4 : Get the corresponding intersection position
    # ----------------------------------------------------------

    return var_branch_path, fix_branch_path


def get_light_path_branch(kin_vec, grating_list, path_list, crystal_list, g_orders):
    """
    Get the light path for one of the branch.

    :param kin_vec: The incident wave vector with a shape of (3,)
    :param grating_list:
    :param path_list:
    :param crystal_list:
    :param g_orders: The diffraction orders of the gratings
    :return:
    """

    # Get kout from the first grating
    kout_g1 = kin_vec + g_orders[0] * grating_list[0].base_wave_vector

    # Get the intersection point on the Bragg crystal
    intersect_1 = path_list[0] * kout_g1 / util.l2_norm(kout_g1)

    # Get the intersection point on the rest of the crystals and the second grating.
    intersect_list, kout_vec_list = get_point_with_definite_path(kin_vec=kout_g1,
                                                                 path_sections=path_list[1:-1],
                                                                 crystal_list=crystal_list,
                                                                 init_point=intersect_1)

    # Get the final output momentum
    kout_g2 = kout_vec_list[-1] + g_orders[1] * grating_list[1].base_wave_vector

    # Calculate the observation point
    intersect_final = intersect_list[-1] + path_list[-1] * kout_g2 / util.l2_norm(kout_g2)

    # Get branch 1 info
    num = len(path_list) + 1

    intersect_branch = np.zeros((num, 3), dtype=np.float64)
    intersect_branch[1, :] = intersect_1[:]
    intersect_branch[2:-1, :] = intersect_list[:, :]
    intersect_branch[-1, :] = intersect_final[:]

    kout_branch = np.zeros((num, 3), dtype=np.float64)
    kout_branch[0, :] = kin_vec[:]
    kout_branch[1, :] = kout_g1[:]
    kout_branch[2:-1, :] = kout_vec_list[:, :]
    kout_branch[-1, :] = kout_g2[:]

    return intersect_branch, kout_branch


def get_point_with_definite_path(kin_vec, path_sections, crystal_list, init_point):
    """
    Provide the crystals, calculate teh corresponding intersection points.

    :param kin_vec:
    :param path_sections:
    :param crystal_list:
    :param init_point:
    :return:
    """
    # Get the number of crystals
    num = len(crystal_list)

    # Prepare holders for the calculation
    intersect_list = np.zeros((num, 3), dtype=np.float64)
    kout_list = np.zeros((num, 3), dtype=np.float64)

    # Copy the initial point
    init = np.copy(init_point)
    kin = np.copy(kin_vec)

    for idx in range(num):
        # Get the reflected wavevector
        kout = util.get_bragg_kout(kin=kin,
                                   h=crystal_list[idx].h,
                                   normal=crystal_list[idx].normal,
                                   compare_length=False)

        # Get the next intersection point
        intersect = init + kout * path_sections[idx] / util.l2_norm(kout)

        # Update the intersection list and the kout list
        kout_list[idx, :] = kout[:]
        intersect_list[idx, :] = intersect[:]

        # Update the kin and init
        init = np.copy(intersect)
        kin = np.copy(kout)

    return intersect_list, kout_list
