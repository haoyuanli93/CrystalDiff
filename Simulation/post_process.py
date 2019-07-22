import numpy as np
import h5py as h5
from skimage.measure import block_reduce
import time
import sys

"""
This script aims to extract the simulation field data and save them to a h5 file.
During this process, a down-sampling of s factor of 2.
"""


def post_process(source_file, output_folder, simulation_tag):
    # Step 1: Open the source file to get the data
    with h5.File(source_file) as h5file:
        group = h5file["raw_data"]
        # Load the data
        data = np.array(group["electric field"])

        # Step 2: Take the Fourier transform of this dataset
        data = np.fft.ifft(np.fft.ifft(data, axis=0), axis=1)
        field_in_real_space = np.fft.ifftshift(data, axes=(0, 1))

        # Step 3: Downsample the data
        ds_field = block_reduce(field_in_real_space,
                                block_size=(2, 2, 2, 1),
                                func=np.sum)

    # Step 4: Save the data to the proper file
    with h5.File("{}/{}_vector_field.h5".format(output_folder, simulation_tag), 'w') as vector_field_h5:
        vector_field_h5.create_dataset("field",
                                       data=field_in_real_space,
                                       chunks=True,
                                       compression="gzip")

    # Step 4: Save the data to the proper file
    with h5.File("{}/{}_y_field.h5".format(output_folder, simulation_tag), 'w') as y_field_h5:
        y_field_h5.create_dataset("field",
                                  data=field_in_real_space[:, :, :, 1],
                                  chunks=True,
                                  compression="gzip")

    # Step 4: Save the data to the proper file
    with h5.File("{}/{}_vector_field_ds.h5".format(output_folder,
                                                   simulation_tag), 'w') as vector_field_ds_h5:
        vector_field_ds_h5.create_dataset("field",
                                          data=ds_field,
                                          chunks=True,
                                          compression="gzip")

    # Step 4: Save the data to the proper file
    with h5.File("{}/{}_y_field_ds.h5".format(output_folder, simulation_tag), 'w') as y_field_ds_h5:
        y_field_ds_h5.create_dataset("field",
                                     data=ds_field[:, :, :, 1],
                                     chunks=True,
                                     compression="gzip")


# Change the STDOUT stream to the file to save the log.
stdout = sys.stdout
sys.stdout = open('./log.txt', 'w')

# ------------------------------------------------------------------------
#            For asymmetric 23 configuration
# ------------------------------------------------------------------------

# Build the dictionary for the process
hfolder = "/home/haoyuan/data_disk/simulation2/asymmetric/"
ofolder = "/home/haoyuan/data_disk/simulation2/post_process"  # home folder

# Process the simulation for asymmetric23
file_list = ["asymmetric23_3_result_2019_07_14_17_05_23.h5",
             "asymmetric23_5_result_2019_07_14_18_01_42.h5",
             "asymmetric23_7_result_2019_07_14_19_17_00.h5",
             "asymmetric23_9_result_2019_07_14_20_44_21.h5"]
simu_list = ["3", "5", "7", "9"]
simu_type = "asymmetric23"

for l in range(4):
    # Get the file name
    file_name = file_list[l]
    print("Begin processing file {}.".format(file_name))
    tic = time.time()

    # Process the file
    post_process(source_file=hfolder + file_name,
                 output_folder=ofolder,
                 simulation_tag="{}_{}".format(simu_type, simu_list[l]))

    toc = time.time()
    print("It takes {:.2f} to process file {}.".format(toc - tic, file_name))

# ------------------------------------------------------------------------
#            For asymmetric 24 configuration
# ------------------------------------------------------------------------

# Build the dictionary for the process
hfolder = "/home/haoyuan/data_disk/simulation2/asymmetric/"
ofolder = "/home/haoyuan/data_disk/simulation2/post_process"  # home folder

# Process the simulation for asymmetric23
file_list = ["asymmetric24_3_result_2019_07_14_22_28_43.h5",
             "asymmetric24_5_result_2019_07_14_23_24_49.h5",
             "asymmetric24_7_result_2019_07_15_00_39_21.h5",
             "asymmetric24_9_result_2019_07_15_02_10_55.h5"]
simu_list = ["3", "5", "7", "9"]
simu_type = "asymmetric24"

for l in range(4):
    # Get the file name
    file_name = file_list[l]
    print("Begin processing file {}.".format(file_name))
    tic = time.time()

    # Process the file
    post_process(source_file=hfolder + file_name,
                 output_folder=ofolder,
                 simulation_tag="{}_{}".format(simu_type, simu_list[l]))

    toc = time.time()
    print("It takes {:.2f} to process file {}.".format(toc - tic, file_name))

# ------------------------------------------------------------------------
#            For inclined configuration
# ------------------------------------------------------------------------

# Build the dictionary for the process
hfolder = "/home/haoyuan/data_disk/simulation2/inclined/"
ofolder = "/home/haoyuan/data_disk/simulation2/post_process"  # home folder

# Process the simulation for asymmetric23
file_list = ["inclined_3_result_2019_07_14_06_17_41.h5",
             "inclined_5_result_2019_07_14_07_12_56.h5",
             "inclined_7_result_2019_07_14_08_27_22.h5",
             "inclined_9_result_2019_07_14_09_59_19.h5"]
simu_list = ["3", "5", "7", "9"]
simu_type = "inclined"

for l in range(4):
    # Get the file name
    file_name = file_list[l]
    print("Begin processing file {}.".format(file_name))
    tic = time.time()

    # Process the file
    post_process(source_file=hfolder + file_name,
                 output_folder=ofolder,
                 simulation_tag="{}_{}".format(simu_type, simu_list[l]))

    toc = time.time()
    print("It takes {:.2f} to process file {}.".format(toc - tic, file_name))

# Change the stdout back
sys.stdout = stdout
