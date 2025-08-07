from ..decorators import create_typed_dict, create_typed_list, maybe_njit

# Handle Numba-specific imports conditionally
try:
    from numba import types

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


def create_empty_results_dict():
    return create_typed_dict(
        key_type=types.unicode_type if NUMBA_AVAILABLE else None,
        value_type=types.float64 if NUMBA_AVAILABLE else None,
    )


def create_empty_results_dict_list(length):
    return create_typed_list(
        initial_list=[create_empty_results_dict() for _ in range(length)]
    )


def convert_results_dict_list_to_run(q_ids, results_dict_list):
    combined_run = create_typed_dict()

    for i, q_id in enumerate(q_ids):
        combined_run[q_id] = results_dict_list[i]

    return combined_run
