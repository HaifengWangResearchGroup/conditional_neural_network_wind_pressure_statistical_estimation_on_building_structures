import numpy

from matplotlib import pyplot


def function_visualize_predict_all_files(array2d_label, array2d_pred):
    int_batch_size = array2d_label.shape[0]

    pyplot.figure()
    value_size_pred = 1
    value_size_label = 0.5
    array_indices = numpy.arange(0, int_batch_size, 1)
    i_index = 0
    temp_arg_sorg = numpy.argsort(array2d_label[array_indices, i_index])
    temp_array_y_pred \
        = array2d_pred[array_indices[temp_arg_sorg], i_index]
    temp_array_y_label \
        = array2d_label[array_indices[temp_arg_sorg], i_index]
    pyplot.scatter(
        numpy.arange(temp_array_y_pred.size), temp_array_y_pred,
        label="prediction",
        c="black",
        s=value_size_pred,
        alpha=0.3,
        edgecolors="none")
    pyplot.scatter(
        numpy.arange(temp_array_y_label.size), temp_array_y_label,
        label="label",
        c="blue",
        s=value_size_label,
        alpha=0.3,
        edgecolors="none")
    pyplot.xlabel("Sorted data index")
    pyplot.ylabel("Statistics")
    pyplot.legend(ncol=2)
    pyplot.show()

    return None


def function_visualize_predict_single_file(
        array2d_label, array2d_pred, int_number_files):

    # Show value-sorted prediction
    pyplot.figure()
    list_moment_name = ["Mean", "STD", "max", "min"]
    list_color = ["red", "green", "blue", "black"]
    value_size_pred = 2
    value_size_label = 3
    value_alpha = 0.55
    array_indices \
        = numpy.arange(0, array2d_label.shape[0] // int_number_files, 1)

    i_moment = 0

    temp_arg_sorg = numpy.argsort(array2d_label[array_indices, i_moment])
    temp_array_y_pred \
        = array2d_pred[array_indices[temp_arg_sorg], i_moment]
    temp_array_y_data \
        = array2d_label[array_indices[temp_arg_sorg], i_moment]
    pyplot.scatter(
        array_indices - array_indices.min(), temp_array_y_pred,
        label="Estimation: {}".format(list_moment_name[i_moment]),
        c=list_color[i_moment * 2 + 0], 
        s=value_size_pred,
        alpha=value_alpha,
        marker='*', edgecolor="none")
    pyplot.scatter(
        array_indices - array_indices.min(), temp_array_y_data,
        label="Measurement: {}".format(list_moment_name[i_moment]),
        c=list_color[i_moment * 2 + 1], 
        s=value_size_label,
        alpha=value_alpha,
        marker='o', edgecolor="none")

    pyplot.legend(ncol=1, loc="upper left")
    pyplot.xlim([0, array_indices.max() - array_indices.min()])
    pyplot.xlabel("Sorted data index")
    pyplot.ylabel("Statistics")
    pyplot.show()

    return None
