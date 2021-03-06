import numpy as np

from CrystalDiff import util


def adjust_path_length(fix_branch_path, fix_branch_crystal,
                       var_branch_path, var_branch_crystal,
                       grating_pair,
                       kin,
                       delay_time=None):
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
    (intersect_fixed,
     kout_fixed) = get_light_path_branch(kin_vec=kin,
                                         grating_list=grating_pair,
                                         path_list=fix_branch_path,
                                         crystal_list=fix_branch_crystal,
                                         g_orders=[-1, 1])  # By default, -1 corresponds to the fixed branch.
    (intersect_var,
     kout_var) = get_light_path_branch(kin_vec=kin,
                                       grating_list=grating_pair,
                                       path_list=var_branch_path,
                                       crystal_list=var_branch_crystal,
                                       g_orders=[1, -1])
    # ----------------------------------------------------------
    # Step 2 : Tune the final section of the fixed branch
    # to make sure the intersection point is close to the z axis
    # ----------------------------------------------------------
    # Make sure the fixed delay branch intersect with the y axis
    sine = np.abs(kout_fixed[-2, 1] / np.linalg.norm(kout_fixed[-2]))
    fix_branch_path[-2] = np.abs(intersect_fixed[-3][1]) / sine

    # Update our understanding of the path of the branch with fixed delay
    (intersect_fixed,
     kout_fixed) = get_light_path_branch(kin_vec=kin,
                                         grating_list=grating_pair,
                                         path_list=fix_branch_path,
                                         crystal_list=fix_branch_crystal,
                                         g_orders=[-1, 1])  # By default, -1 corresponds to the fixed branch.
    # ----------------------------------------------------------
    # Step 3 : Tune the path sections of the variable branch
    #          so that the light path intersect with the fixed branch on the second grating
    # ----------------------------------------------------------
    term_1 = np.linalg.norm(np.cross(intersect_fixed[-2] - intersect_var[-4], kout_var[-3]))
    term_2 = np.linalg.norm(np.cross(kout_var[-2], kout_var[-3])) / np.linalg.norm(kout_var[-2])
    var_branch_path[-2] = term_1 / term_2

    term_1 = np.linalg.norm(np.cross(intersect_fixed[-2] - intersect_var[-4], kout_var[-2]))
    term_2 = np.linalg.norm(np.cross(kout_var[-3], kout_var[-2])) / np.linalg.norm(kout_var[-3])
    var_branch_path[-3] = term_1 / term_2

    # Tune channel-cut pair together to find the delay zero position
    path_diff = np.sum(fix_branch_path[:-1]) - np.sum(var_branch_path[:-1])
    klen = util.l2_norm(kout_var[-3])
    cos_theta = np.dot(kout_var[-2], kout_var[-3]) / klen ** 2
    delta = path_diff / 2. / (1 - cos_theta)

    # Change the variable path sections with the calculated length change
    var_branch_path[-5] += delta
    var_branch_path[-3] += delta
    var_branch_path[-4] -= 2 * delta * cos_theta

    # ----------------------------------------------------------
    # Step 3 : Adjust the path sections to match the delay time.
    # ----------------------------------------------------------
    if delay_time is not None:
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
        var_branch_path[-5] += delta
        var_branch_path[-3] += delta
        var_branch_path[-4] -= 2 * delta * cos_theta

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


##########################################################################
#       Functions used to adjust the crystal position
##########################################################################
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


##########################################################################
#       Find the trajectory when the crystals are fixed
##########################################################################
def get_light_trajectory_with_total_path(kin_vec,
                                         init_point,
                                         total_path,
                                         crystal_list,
                                         g_orders):
    """
    Get the light path for one of the branch.

    :param kin_vec: The incident wave vector with a shape of (3,)
    :param init_point:
    :param total_path:
    :param crystal_list:
    :param g_orders: The diffraction orders of the gratings
    :return:
    """
    # Create holder for the calculation
    num = len(crystal_list) + 1
    intersect_array = np.zeros((num, 3), dtype=np.float64)
    kout_array = np.zeros((num, 3), dtype=np.float64)

    intersect_array[0, :] = np.copy(init_point)
    kout_array[0, :] = np.copy(kin_vec)

    # Create auxiliary variables for the calculation
    g_idx = 0  # The index for grating
    remain_path = np.copy(total_path)

    # Loop through all the crystals to get the path
    for idx in range(1, num):
        crystal_obj = crystal_list[idx - 1]
        # Get the output wave vector
        if crystal_obj.type == "Crystal: Bragg Reflection":
            kout_array[idx, :] = util.get_bragg_kout(kin=kout_array[idx - 1],
                                                     h=crystal_obj.h,
                                                     normal=crystal_obj.normal,
                                                     compare_length=False)
        if crystal_obj.type == "Transmissive Grating":
            kout_array[idx, :] = (kout_array[idx - 1] +
                                  g_orders[g_idx] * crystal_obj.base_wave_vector)
            # Update the index of the grating
            g_idx += 1

        # The last intersection point need to be calculated individually
        if idx == num - 1:
            break

        # Get the new intersection point
        intersect_array[idx] = util.get_intersection(s=intersect_array[idx - 1],
                                                     k=kout_array[idx],
                                                     n=crystal_list[idx].normal,
                                                     x0=crystal_list[idx].surface_point)
        # Get the remaining path
        path_tmp = util.l2_norm(intersect_array[idx] - intersect_array[idx - 1])

        # If the path length is not long enough to cover the whole device, then stop early.
        if path_tmp > remain_path:
            intersect_array[idx] = (intersect_array[idx - 1] +
                                    kout_array[idx] * remain_path / util.l2_norm(kout_array[idx]))
            return intersect_array, kout_array
        else:
            remain_path -= path_tmp

    # If the path length is long though to cover the whole path, then calculate the final point
    intersect_array[-1] = (intersect_array[-2] +
                           kout_array[-1] * remain_path / util.l2_norm(kout_array[-1]))

    return intersect_array, kout_array


##########################################################################
#       Create Delay time
##########################################################################

def get_distance_for_delay_time():
    """
    There, I would like to calculate the distance over which I need to move to change
    the delay time.
    :return:
    """
    pass
