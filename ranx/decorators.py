"""Conditional decorators for optional Numba support."""

from typing import Any, Callable, Dict, List, Union

from .config import use_numba

# Check if Numba is available
try:
    from numba import jit, njit, prange
    from numba.typed import Dict as TypedDict
    from numba.typed import List as TypedList

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create dummy objects for when Numba is not available
    TypedDict = dict
    TypedList = list


def maybe_njit(*args, **kwargs):
    """
    Conditional njit decorator that falls back to identity function when Numba is disabled.

    This decorator will apply Numba's njit compilation when:
    1. Numba is available (installed)
    2. Numba is enabled via configuration

    Otherwise, it returns the function unchanged (pure Python execution).

    Args:
        *args, **kwargs: Same arguments as numba.njit

    Returns:
        Function decorator that conditionally applies Numba compilation
    """

    def decorator(func):
        if NUMBA_AVAILABLE and use_numba():
            return njit(*args, **kwargs)(func)
        else:
            return func

    # Handle the case where maybe_njit is used without arguments: @maybe_njit
    if len(args) == 1 and callable(args[0]) and not kwargs:
        func = args[0]
        if NUMBA_AVAILABLE and use_numba():
            return njit()(func)
        else:
            return func

    return decorator


def maybe_jit(*args, **kwargs):
    """
    Conditional jit decorator that falls back to identity function when Numba is disabled.

    Similar to maybe_njit but uses jit instead of njit.

    Args:
        *args, **kwargs: Same arguments as numba.jit

    Returns:
        Function decorator that conditionally applies Numba compilation
    """

    def decorator(func):
        if NUMBA_AVAILABLE and use_numba():
            return jit(*args, **kwargs)(func)
        else:
            return func

    # Handle the case where maybe_jit is used without arguments: @maybe_jit
    if len(args) == 1 and callable(args[0]) and not kwargs:
        func = args[0]
        if NUMBA_AVAILABLE and use_numba():
            return jit()(func)
        else:
            return func

    return decorator


# We need to handle prange differently since Numba needs to know about it at compile time
if NUMBA_AVAILABLE:
    from numba import prange as numba_prange

    def maybe_prange(*args, **kwargs):
        """
        Conditional prange that falls back to regular range when Numba is disabled.

        Returns:
            prange when Numba is available and enabled, otherwise range
        """
        if use_numba():
            return numba_prange(*args, **kwargs)
        else:
            return range(*args, **kwargs)

else:

    def maybe_prange(*args, **kwargs):
        """Fallback to range when Numba is not available."""
        return range(*args, **kwargs)


def create_typed_dict(key_type=None, value_type=None, initial_dict=None):
    """
    Create a typed dictionary that falls back to regular dict when Numba is disabled.

    Args:
        key_type: Numba type for keys (ignored when Numba disabled)
        value_type: Numba type for values (ignored when Numba disabled)
        initial_dict: Initial dictionary to populate

    Returns:
        numba.typed.Dict when Numba enabled, regular dict otherwise
    """
    if NUMBA_AVAILABLE and use_numba():
        if initial_dict:
            typed_dict = TypedDict()
            for k, v in initial_dict.items():
                typed_dict[k] = v
            return typed_dict
        elif key_type is not None and value_type is not None:
            return TypedDict.empty(key_type, value_type)
        else:
            return TypedDict()
    else:
        return dict(initial_dict) if initial_dict else {}


def create_typed_list(item_type=None, initial_list=None):
    """
    Create a typed list that falls back to regular list when Numba is disabled.

    Args:
        item_type: Numba type for items (ignored when Numba disabled)
        initial_list: Initial list to populate

    Returns:
        numba.typed.List when Numba enabled, regular list otherwise
    """
    if NUMBA_AVAILABLE and use_numba():
        if initial_list:
            typed_list = TypedList()
            for item in initial_list:
                typed_list.append(item)
            return typed_list
        elif item_type is not None:
            return TypedList.empty_list(item_type)
        else:
            return TypedList()
    else:
        return list(initial_list) if initial_list else []
