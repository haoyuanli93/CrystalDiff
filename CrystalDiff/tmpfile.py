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



