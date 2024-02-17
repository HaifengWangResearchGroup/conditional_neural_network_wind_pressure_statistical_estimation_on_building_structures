"""
MIT License



Neural Network Read Data
------------------------
This file contains functions that read data for neural network training.

Author:
    Haifeng Wang

"""

import os

import numpy


def function_get_training_data(
        str_file_name="",
        bool_verbose=False):
    """
    Nots:
        must run in project root folder
    ----------------------------------------------------------------------------
    return:
        array2D_hyper_input[i_sample, i_feature]
            i_feature:
                0: float_roof_slope
                1: int_exposure
                2: float_model_scale
                3: int_leakage_case
                4: float_eaves_height
                5: float_angle_degree
                6: float_width
                7: float_depth
                8: float_heigh
                9: float_z0
                10: int_side_number
        array2D_model_input[i_sample, i_coord]
        array2D_model_output[i_sample, i_parameter]
    """
    path_file = "data/wind_tunnel/TPU_LR_cache/" + str_file_name

    assert os.path.exists(path_file)

    rawdata = numpy.load(path_file, allow_pickle=True)
    array2D_hyper_input = rawdata["array2D_hyper_input"]
    array2D_model_input = rawdata["array2D_model_input"]
    array2D_model_output = rawdata["array2D_model_output"]
    array2D_tap_coordinate = rawdata["array2D_tap_coordinate"]
    array2D_model_output_normalized \
        = rawdata["array2D_model_output_normalized"]
    list_path_file_loaded = rawdata["list_path_file_loaded"]
    list_index_training = rawdata["list_index_training"]
    list_index_testing = rawdata["list_index_testing"]

    if array2D_hyper_input.shape[1] == 6:
        array2D_hyper_input[:, 1] /= 90
        array2D_hyper_input[:, 2] /= 0.016
        array2D_hyper_input[:, 3] /= 0.016
        array2D_hyper_input[:, 4] /= 0.016
        array2D_hyper_input[:, 5] /= 6
    return [
        array2D_hyper_input, array2D_model_input, array2D_model_output,
        array2D_tap_coordinate, array2D_model_output_normalized,
        list_path_file_loaded,
        list_index_training, list_index_testing]


def function_get_min_max():
    list_min_max = [
        [-1.8424679040908813, 0.9058182239532471],
        [0.0952516719698906, 0.8578644394874573],
        [-3.07995343208313, 1.641658902168274],
        [2.699600900258527, 24.928558392698665],
    ]
    return list_min_max
