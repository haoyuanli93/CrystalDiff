import numpy as np

from CrystalDiff import util, lightpath


################################################################################################
#                            Not Essential
################################################################################################
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
     kout_branch_1) = lightpath.get_light_path_branch(kin_vec=kin,
                                                      grating_list=grating_list,
                                                      path_list=path_list_1,
                                                      crystal_list=crystal_list_1,
                                                      g_orders=[1, -1])

    (intersect_branch_2,
     kout_branch_2) = lightpath.get_light_path_branch(kin_vec=kin,
                                                      grating_list=grating_list,
                                                      path_list=path_list_2,
                                                      crystal_list=crystal_list_2,
                                                      g_orders=[-1, 1])

    return intersect_branch_1, kout_branch_1, intersect_branch_2, kout_branch_2


def get_intersection_point(kin_vec, init_point, crystal_list):
    """
    Calculate the intersection points given the initial point and the crystals.

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
    s = np.copy(init_point)
    k = np.copy(kin_vec)

    # Loop through all the reflections
    for idx in range(reflect_num):
        # Get the intersection point
        intersect_array[idx] = util.get_intersection_point(s=s,
                                                           k=k,
                                                           n=crystal_list[idx].normal,
                                                           x0=crystal_list[idx].surface_point)

        # Update the k and s and kout holder
        kout_array[idx] = util.get_bragg_kout(kin=k,
                                              h=crystal_list[idx].h,
                                              normal=crystal_list[idx].normal)

        k = np.copy(kout_array[idx])
        s = np.copy(intersect_array[idx])

    return intersect_array, kout_array
