import numpy as np
import numba as nb
from numba.typed import List


def to_typed_list(ls):
    """Convert a list of Numpy Arrays into a Numba Typed List."""
    typed_list = List()

    for x in ls:
        typed_list.append(x)

    return typed_list


def remove_non_digits(s):
    return "".join(filter(str.isdigit, s))


def convert_trec_y_true(y):
    y_nb_ls = List()
    for key in y.keys():
        row = []
        for k, v in y[key].items():
            if v > 0:
                row.append([int(remove_non_digits(k)), int(v)])
        y_nb_ls.append(np.array(row))
    return y_nb_ls


def convert_trec_y_pred(y):
    keys = list(y.keys())
    y_np_array = np.empty((len(keys), len(y[keys[0]])))

    for i, k in enumerate(keys):
        sorted_result_dict = {
            k: v
            for k, v in sorted(
                list(y[k].items())[::-1], reverse=True, key=lambda item: item[1]
            )
        }
        row = []
        for d in sorted_result_dict.keys():
            row.append(int(remove_non_digits(d)))

        y_np_array[nb.int64(i)] = np.array(row)

    return y_np_array
