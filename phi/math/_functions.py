import functools
import numbers
import re
import time
from functools import partial

import numpy as np

from .backend import default_backend, choose_backend, Solve, LinearSolve, Backend
from ._shape import BATCH_DIM, CHANNEL_DIM, SPATIAL_DIM, Shape, EMPTY_SHAPE, spatial_shape, shape as shape_
from . import extrapolation as extrapolation
from ._tensors import Tensor, tensor, broadcastable_native_tensors, NativeTensor, CollapsedTensor, TensorStack, \
    custom_op2, tensors
from phi.math.backend._scipy_backend import SCIPY_BACKEND


def all_available(*values: Tensor):
    """
    Tests if the values of all given tensors are known and can be read at this point.
    
    Tensors are typically available when the backend operates in eager mode.

    Args:
      values: tensors to check
      *values: Tensor: 

    Returns:
      bool

    """
    for value in values:
        natives = value._natives()
        natives_available = [choose_backend(native).is_available(native) for native in natives]
        if not all(natives_available):
            return False
    return True


def print_(value: Tensor = None, name: str = None):
    """
    Print a tensor with no more than two spatial dimensions, splitting it along all batch and channel dimensions.
    
    Unlike regular printing, the primary dimension, typically x, is oriented to the right.

    Args:
      name: name of the tensor
      value: tensor-like
      value: Tensor:  (Default value = None)
      name: str:  (Default value = None)

    Returns:

    """
    if value is None:
        print()
        return
    if name is not None:
        print(" " * 16 + name)
    value = tensor(value)
    dim_order = tuple(sorted(value.shape.spatial.names, reverse=True))
    if value.shape.spatial_rank == 0:
        print(value.numpy())
    elif value.shape.spatial_rank == 1:
        for index_dict in value.shape.non_spatial.meshgrid():
            if value.shape.non_spatial.volume > 1:
                print(f"--- {', '.join('%s=%d' % (name, idx) for name, idx in index_dict.items())} ---")
            text = np.array2string(value[index_dict].numpy(dim_order), precision=2, separator=', ', max_line_width=np.inf)
            print(' ' + re.sub('[\[\]]', '', text))
    elif value.shape.spatial_rank == 2:
        for index_dict in value.shape.non_spatial.meshgrid():
            if value.shape.non_spatial.volume > 1:
                print(f"--- {', '.join('%s=%d' % (name, idx) for name, idx in index_dict.items())} ---")
            text = np.array2string(value[index_dict].numpy(dim_order), precision=2, separator=', ', max_line_width=np.inf)
            print(' ' + re.sub('[\[\]]', '', re.sub('\],', '', text)))
    else:
        raise NotImplementedError('Can only print tensors with up to 2 spatial dimensions.')


def _initialize(uniform_initializer, shape=EMPTY_SHAPE, dtype=None, **dimensions):
    shape &= shape_(**dimensions)
    if shape.is_non_uniform:
        stack_dim = shape.shape.without('dims')[0:1]
        shapes = shape.unstack(stack_dim.name)
        tensors = [_initialize(uniform_initializer, s, dtype) for s in shapes]
        return _stack(tensors, stack_dim.name, stack_dim.types[0])
    else:
        return uniform_initializer(shape, dtype)


def zeros(shape=EMPTY_SHAPE, dtype=None, **dimensions):
    """
    Define a tensor with specified shape with value 0 / False everywhere.
    
    This method may not immediately allocate the memory to store the values.

    Args:
      shape: base tensor shape (Default value = EMPTY_SHAPE)
      dtype: data type (Default value = None)
      dimensions: additional dimensions, types are determined from names
      **dimensions: 

    Returns:
      tensor of specified shape

    """
    return _initialize(lambda shape, dtype: CollapsedTensor(NativeTensor(default_backend().zeros((), dtype=dtype), EMPTY_SHAPE), shape), shape, dtype, **dimensions)


def zeros_like(tensor: Tensor):
    return zeros(tensor.shape, dtype=tensor.dtype)


def ones(shape=EMPTY_SHAPE, dtype=None, **dimensions):
    """
    Define a tensor with specified shape with value 1 / True everywhere.
    
    This method may not immediately allocate the memory to store the values.

    Args:
      shape: base tensor shape (Default value = EMPTY_SHAPE)
      dtype: data type (Default value = None)
      dimensions: additional dimensions, types are determined from names
      **dimensions: 

    Returns:
      tensor of specified shape

    """
    return _initialize(lambda shape, dtype: CollapsedTensor(NativeTensor(default_backend().ones((), dtype=dtype), EMPTY_SHAPE), shape), shape, dtype, **dimensions)


def ones_like(tensor: Tensor):
    return zeros(tensor.shape, dtype=tensor.dtype) + 1


def random_normal(shape=EMPTY_SHAPE, dtype=None, **dimensions):

    def uniform_random_normal(shape, dtype):
        native = choose_backend(*shape.sizes, prefer_default=True).random_normal(shape.sizes)
        native = native if dtype is None else native.astype(dtype)
        return NativeTensor(native, shape)

    return _initialize(uniform_random_normal, shape, dtype, **dimensions)


def random_uniform(shape=EMPTY_SHAPE, dtype=None, **dimensions):

    def uniform_random_uniform(shape, dtype):
        native = choose_backend(*shape.sizes, prefer_default=True).random_uniform(shape.sizes)
        native = native if dtype is None else native.astype(dtype)
        return NativeTensor(native, shape)

    return _initialize(uniform_random_uniform, shape, dtype, **dimensions)


def transpose(value, axes):
    if isinstance(value, Tensor):
        return CollapsedTensor(value, value.shape[axes])
    else:
        return choose_backend(value).transpose(value, axes)


def fftfreq(resolution, dtype=None):
    """
    Returns the discrete Fourier transform sample frequencies.
    These are the frequencies corresponding to the components of the result of `math.fft` on a tensor of shape `resolution`.

    Args:
      resolution: grid resolution measured in cells
      dtype: data type of the returned tensor (Default value = None)

    Returns:
      tensor holding the frequencies of the corresponding values computed by math.fft

    """
    resolution = spatial_shape(resolution)
    k = meshgrid(**{dim: np.fft.fftfreq(int(n)) for dim, n in resolution.named_sizes})
    return to_float(k) if dtype is None else cast(k, dtype)


def meshgrid(**dimensions):
    """
    generate a TensorStack meshgrid from keyword dimensions

    Args:
      **dimensions: 

    Returns:

    """
    assert 'vector' not in dimensions
    dimensions = {dim: tuple(range(val)) if isinstance(val, int) else val for dim, val in dimensions.items()}
    indices_list = choose_backend(*dimensions.values(), prefer_default=True).meshgrid(*dimensions.values())
    single_shape = Shape([len(val) for val in dimensions.values()], dimensions.keys(), [SPATIAL_DIM] * len(dimensions))
    channels = [NativeTensor(t, single_shape) for t in indices_list]
    return TensorStack(channels, 'vector', CHANNEL_DIM)


def channel_stack(values, dim: str):
    return _stack(values, dim, CHANNEL_DIM)


def batch_stack(values, dim: str = 'batch'):
    return _stack(values, dim, BATCH_DIM)


def spatial_stack(values, dim: str):
    return _stack(values, dim, SPATIAL_DIM)


def _stack(values: tuple or list,
           dim: str,
           dim_type: str):
    def inner_stack(*values):
        return TensorStack(values, dim, dim_type)

    result = broadcast_op(inner_stack, values)
    return result


def concat(values: tuple or list, dim: str) -> Tensor:
    """
    Concatenates a sequence of tensors along one dimension.
    The shapes of all values must be equal, except for the size of the concat dimension.

    Args:
      values: Tensors to concatenate
      dim: concat dimension, must be present in all values
      values: tuple or list: 
      dim: str: 

    Returns:
      concatenated tensor

    """
    broadcast_shape = values[0].shape
    natives = [v.native(order=broadcast_shape.names) for v in values]
    backend = choose_backend(*natives)
    concatenated = backend.concat(natives, broadcast_shape.index(dim))
    return NativeTensor(concatenated, broadcast_shape.with_sizes(backend.staticshape(concatenated)))


def spatial_pad(value, pad_width: tuple or list, mode: 'extrapolation.Extrapolation') -> Tensor:
    value = tensor(value)
    return pad(value, {n: w for n, w in zip(value.shape.spatial.names, pad_width)}, mode=mode)


def pad(value: Tensor, widths: dict, mode: 'extrapolation.Extrapolation') -> Tensor:
    """
    Pads a tensor along the specified dimensions, determining the added values using the given extrapolation.
    
    This is equivalent to calling `mode.pad(value, widths)`.

    Args:
      value: tensor to be padded
      widths: name: str -> (lower: int, upper: int)
      mode: Extrapolation object
      value: Tensor: 
      widths: dict: 
      mode: 'extrapolation.Extrapolation': 

    Returns:
      padded Tensor

    """
    return mode.pad(value, widths)


def closest_grid_values(grid: Tensor, coordinates: Tensor, extrap: 'extrapolation.Extrapolation'):
    """
    Finds the neighboring grid points in all spatial directions and returns their values.
    The result will have 2^d values for each vector in coordiantes in d dimensions.

    Args:
      extrap: grid extrapolation
      grid: grid data. The grid is spanned by the spatial dimensions of the tensor
      coordinates: tensor with 1 channel dimension holding vectors pointing to locations in grid index space
      grid: Tensor: 
      coordinates: Tensor: 
      extrap: 'extrapolation.Extrapolation': 

    Returns:
      Tensor of shape (batch, coord_spatial, grid_spatial=(2, 2,...), grid_channel)

    """
    # alternative method: pad array for all 2^d combinations, then stack to simplify gather_nd.
    assert all(name not in grid.shape for name in coordinates.shape.spatial.names), 'grid and coordinates must have different spatial dimensions'
    # --- Pad tensor where transform is not possible ---
    non_copy_pad = {dim: (0 if extrap[dim, 0].is_copy_pad else 1, 0 if extrap[dim, 1].is_copy_pad else 1) for dim in grid.shape.spatial.names}
    grid = extrap.pad(grid, non_copy_pad)
    coordinates += [not extrap[dim, 0].is_copy_pad for dim in grid.shape.spatial.names]
    # --- Transform coordiantes ---
    min_coords = to_int(floor(coordinates))
    max_coords = extrap.transform_coordinates(min_coords + 1, grid.shape)
    min_coords = extrap.transform_coordinates(min_coords, grid.shape)

    def left_right(is_hi_by_axis_left, ax_idx):
        is_hi_by_axis_right = is_hi_by_axis_left | np.array([ax == ax_idx for ax in range(grid.shape.spatial_rank)])
        coords_left = where(is_hi_by_axis_left, max_coords, min_coords)
        coords_right = where(is_hi_by_axis_right, max_coords, min_coords)
        if ax_idx == grid.shape.spatial_rank - 1:
            values_left = gather(grid, coords_left)
            values_right = gather(grid, coords_right)
        else:
            values_left = left_right(is_hi_by_axis_left, ax_idx + 1)
            values_right = left_right(is_hi_by_axis_right, ax_idx + 1)
        return spatial_stack([values_left, values_right], grid.shape.spatial.names[ax_idx])

    result = left_right(np.array([False] * grid.shape.spatial_rank), 0)
    return result


def grid_sample(grid: Tensor, coordinates: Tensor, extrap: 'extrapolation.Extrapolation'):
    result = broadcast_op(functools.partial(_grid_sample, extrap=extrap), [grid, coordinates])
    return result


def _grid_sample(grid: Tensor, coordinates: Tensor, extrap: 'extrapolation.Extrapolation'):
    coord_names = ['_coord_' + dim.name if dim.is_spatial else dim.name for dim in coordinates.shape.unstack()]
    coordinates = coordinates._with_shape_replaced(coordinates.shape.with_names(coord_names))
    neighbors = closest_grid_values(grid, coordinates, extrap)
    binary = meshgrid(**{dim: (0, 1) for dim in grid.shape.spatial.names})
    right_weights = coordinates % 1
    binary, right_weights = join_spaces(binary, right_weights)
    weights = prod(binary * right_weights + (1 - binary) * (1 - right_weights), 'vector')
    result = sum_(neighbors * weights, dim=grid.shape.spatial.names)
    result_names = [dim.name[7:] if dim.is_spatial else dim.name for dim in result.shape.unstack()]
    result = result._with_shape_replaced(result.shape.with_names(result_names))
    return result


def join_spaces(*tensors):
    spatial = functools.reduce(lambda s1, s2: s1.combined(s2, combine_spatial=True), [t.shape.spatial for t in tensors])
    return [CollapsedTensor(t, t.shape.non_spatial & spatial) for t in tensors]


def broadcast_op(operation: callable,
                 tensors: tuple or list,
                 iter_dims: set or tuple or list = None,
                 no_return=False):
    if iter_dims is None:
        iter_dims = set()
        for tensor in tensors:
            if isinstance(tensor, TensorStack) and tensor.requires_broadcast:
                iter_dims.add(tensor.stack_dim_name)
    if len(iter_dims) == 0:
        return operation(*tensors)
    else:
        dim = next(iter(iter_dims))
        dim_type = None
        size = None
        unstacked = []
        for tensor in tensors:
            if dim in tensor.shape:
                unstacked_tensor = tensor.unstack(dim)
                unstacked.append(unstacked_tensor)
                if size is None:
                    size = len(unstacked_tensor)
                    dim_type = tensor.shape.get_type(dim)
                else:
                    assert size == len(unstacked_tensor)
                    assert dim_type == tensor.shape.get_type(dim)
            else:
                unstacked.append(tensor)
        result_unstacked = []
        for i in range(size):
            gathered = [t[i] if isinstance(t, tuple) else t for t in unstacked]
            result_unstacked.append(broadcast_op(operation, gathered, iter_dims=set(iter_dims) - {dim}))
        if not no_return:
            return TensorStack(result_unstacked, dim, dim_type)


def reshape(value: Tensor, *operations: str):
    # '(x, y) -> list', 'batch -> (batch=5, group=2)'
    raise NotImplementedError()


def join_dimensions(value: Tensor, dims: Shape or tuple or list, joined_dim_name: str):
    order = value.shape.order_group(dims)
    native = value.native(order)
    types = value.shape.get_type(dims)
    dim_type = types[0] if len(set(types)) == 1 else BATCH_DIM
    first_dim_index = min(*value.shape.index(dims))
    new_shape = value.shape.without(dims).expand(value.shape.only(dims).volume, joined_dim_name, dim_type, pos=first_dim_index)
    native = choose_backend(native).reshape(native, new_shape.sizes)
    return NativeTensor(native, new_shape)


def where(condition: Tensor or float or int, value_true: Tensor or float or int, value_false: Tensor or float or int):
    """
    Builds a tensor by choosing either values from `value_true` or `value_false` depending on `condition`.
    If `condition` is not of type boolean, non-zero values are interpreted as True.
    
    This function requires non-None values for `value_true` and `value_false`.
    To get the indices of True / non-zero values, use :func:`nonzero`.

    Args:
      condition: determines where to choose values from value_true or from value_false
      value_true: values to pick where condition != 0 / True
      value_false: values to pick where condition == 0 / False
      condition: Tensor or float or int: 
      value_true: Tensor or float or int: 
      value_false: Tensor or float or int: 

    Returns:
      tensor containing dimensions of all inputs

    """
    condition, value_true, value_false = tensors(condition, value_true, value_false)
    shape, (c, vt, vf) = broadcastable_native_tensors(condition, value_true, value_false)
    result = choose_backend(c, vt, vf).where(c, vt, vf)
    return NativeTensor(result, shape)


def nonzero(value: Tensor, list_dim='nonzero', index_dim='vector'):
    """
    Get spatial indices of non-zero / True values.
    
    Batch dimensions are preserved by this operation.
    If channel dimensions are present, this method returns the indices where any entry is nonzero.

    Args:
      value: spatial tensor to find non-zero / True values in.
      list_dim: name of dimension listing non-zero values (Default value = 'nonzero')
      index_dim: name of index dimension (Default value = 'vector')
      value: Tensor: 

    Returns:
      tensor of shape (batch dims..., list_dim=#non-zero, index_dim=value.shape.spatial_rank)

    """
    if value.shape.channel_rank > 0:
        value = sum_(abs(value), value.shape.channel.names)

    def unbatched_nonzero(value):
        native = value.native()
        backend = choose_backend(native)
        indices = backend.nonzero(native)
        indices_shape = Shape(backend.staticshape(indices), (list_dim, index_dim), (BATCH_DIM, CHANNEL_DIM))
        return NativeTensor(indices, indices_shape)

    return broadcast_op(unbatched_nonzero, [value], iter_dims=value.shape.batch.names)


def _reduce(value: Tensor or list or tuple,
            dim: str or tuple or list or Shape or None,
            native_function: callable,
            collapsed_function: callable = lambda inner_reduced, collapsed_dims_to_reduce: inner_reduced,
            unaffected_function: callable = lambda value: value) -> Tensor:
    """

    Args:
        value:
        dim:
        native_function:
        collapsed_function: handles collapsed dimensions, called as `collapsed_function(inner_reduced, collapsed_dims_to_reduce)`
        unaffected_function: returns `unaffected_function(value)` if `len(dims) > 0` but none of them are part of `value`

    Returns:

    """
    if dim in ((), [], EMPTY_SHAPE):
        return value
    if isinstance(value, (tuple, list)):
        values = [tensor(v) for v in value]
        value = _stack(values, '_reduce', BATCH_DIM)
        if dim is None:
            pass  # continue below
        elif dim == 0:
            dim = '_reduce'
        else:
            raise ValueError('dim must be 0 or None when passing a sequence of tensors')
    else:
        value = tensor(value)
    dims = _resolve_dims(dim, value.shape)
    if all([dim not in value.shape for dim in dims]):
        return unaffected_function(value)  # no dim to sum over
    if isinstance(value, NativeTensor):
        native = value.native()
        result = native_function(choose_backend(native), native, dim=value.shape.index(dims))
        return NativeTensor(result, value.shape.without(dims))
    if isinstance(value, CollapsedTensor):
        inner_reduce = _reduce(value.tensor, dims, native_function, collapsed_function, unaffected_function)
        collapsed_dims = value.shape.without(value.tensor.shape)
        final_shape = value.shape.without(dims)
        total_reduce = collapsed_function(inner_reduce, collapsed_dims.only(dims))
        return CollapsedTensor(total_reduce, final_shape)
    elif isinstance(value, TensorStack):
        # --- inner reduce ---
        inner_axes = [dim for dim in dims if dim != value.stack_dim_name]
        red_inners = [_reduce(t, inner_axes, native_function, collapsed_function, unaffected_function) for t in value.tensors]
        # --- outer reduce ---
        from ._track import ShiftLinOp, sum_operators
        if value.stack_dim_name in dims:
            if any([isinstance(t, ShiftLinOp) for t in red_inners]):
                return sum(red_inners[1:], red_inners[0])
            natives = [t.native() for t in red_inners]
            result = native_function(choose_backend(*natives), natives, dim=0)  # TODO not necessary if tensors are CollapsedTensors
            return NativeTensor(result, red_inners[0].shape)
        else:
            return TensorStack(red_inners, value.stack_dim_name, value.stack_dim_type)
    else:
        raise NotImplementedError(f"{type(value)} not supported. Only (NativeTensor, TensorStack) allowed.")


def _resolve_dims(dim: str or tuple or list or Shape or None,
                  shape: Shape) -> tuple or list:
    if dim is None:
        return shape.names
    if isinstance(dim, (tuple, list)):
        return dim
    if isinstance(dim, str):
        return [dim]
    if isinstance(dim, Shape):
        return dim.names
    raise ValueError(dim)


def sum_(value: Tensor or list or tuple,
         dim: str or int or tuple or list or None or Shape = None) -> Tensor:
    return _reduce(value, dim,
                   native_function=lambda backend, native, dim: backend.sum(native, dim),
                   collapsed_function=lambda inner, red_shape: inner * red_shape.volume)


def prod(value: Tensor or list or tuple,
         dim: str or int or tuple or list or None or Shape = None) -> Tensor:
    return _reduce(value, dim,
                   native_function=lambda backend, native, dim: backend.prod(native, dim),
                   collapsed_function=lambda inner, red_shape: inner ** red_shape.volume)


def mean(value: Tensor or list or tuple,
         dim: str or int or tuple or list or None or Shape = None) -> Tensor:
    return _reduce(value, dim,
                   native_function=lambda backend, native, dim: backend.mean(native, dim))


def std(value: Tensor or list or tuple,
         dim: str or int or tuple or list or None or Shape = None) -> Tensor:
    return _reduce(value, dim,
                   native_function=lambda backend, native, dim: backend.std(native, dim),
                   collapsed_function=lambda inner, red_shape: inner,
                   unaffected_function=lambda value: value * 0)


def any_(boolean_tensor: Tensor or list or tuple,
         dim: str or int or tuple or list or None or Shape = None) -> Tensor:
    return _reduce(boolean_tensor, dim,
                   native_function=lambda backend, native, dim: backend.any(native, dim))


def all_(boolean_tensor: Tensor or list or tuple,
         dim: str or int or tuple or list or None or Shape = None) -> Tensor:
    return _reduce(boolean_tensor, dim,
                   native_function=lambda backend, native, dim: backend.all(native, dim))


def max_(value: Tensor or list or tuple,
         dim: str or int or tuple or list or None or Shape = None) -> Tensor:
    return _reduce(value, dim,
                   native_function=lambda backend, native, dim: backend.max(native, dim))


def min_(value: Tensor or list or tuple,
         dim: str or int or tuple or list or None or Shape = None) -> Tensor:
    return _reduce(value, dim,
                   native_function=lambda backend, native, dim: backend.min(native, dim))


def dot(a, b, axes):
    raise NotImplementedError()


def matmul(A, b):
    raise NotImplementedError()


def einsum(equation, *tensors):
    raise NotImplementedError()


def _backend_op1(x: Tensor, unbound_method):
    return x._op1(lambda native: getattr(choose_backend(native), unbound_method.__name__)(native))


def abs(x: Tensor):
    return _backend_op1(x, Backend.abs)


def sign(x: Tensor):
    return _backend_op1(x, Backend.sign)


def round_(x: Tensor):
    return _backend_op1(x, Backend.round)


def ceil(x: Tensor):
    return _backend_op1(x, Backend.ceil)


def floor(x: Tensor):
    return _backend_op1(x, Backend.floor)


def sqrt(x: Tensor):
    return _backend_op1(x, Backend.sqrt)


def exp(x: Tensor):
    return _backend_op1(x, Backend.exp)


def to_float(x: Tensor) -> Tensor:
    """
    Converts the given tensor to floating point format with the currently specified precision.
    
    The precision can be set globally using `math.set_global_precision()` and locally using `with math.precision()`.
    
    See the `phi.math` module documentation at https://tum-pbs.github.io/PhiFlow/Math.html

    Args:
      x: values to convert
      x: Tensor: 

    Returns:
      Tensor of same shape as `x`

    """
    return _backend_op1(x, Backend.to_float)


def to_int(x: Tensor, int64=False):
    return x._op1(lambda native: choose_backend(native).to_int(native, int64=int64))


def to_complex(x: Tensor):
    return _backend_op1(x, Backend.to_complex)


def isfinite(x: Tensor):
    return _backend_op1(x, Backend.isfinite)


def imag(complex: Tensor):
    return _backend_op1(complex, Backend.imag)


def real(complex: Tensor):
    return _backend_op1(complex, Backend.real)


def cast(x: Tensor, dtype):
    return x._op1(lambda native: choose_backend(native).cast(native, dtype=dtype))


def sin(x):
    return _backend_op1(x, Backend.sin)


def cos(x):
    return _backend_op1(x, Backend.cos)


def divide_no_nan(x, y):
    return custom_op2(x, y, divide_no_nan, lambda x_, y_: choose_backend(x_, y_).divide_no_nan(x_, y_), lambda y_, x_: divide_no_nan(x_, y_), lambda y_, x_: choose_backend(x_, y_).divide_no_nan(x_, y_))


def maximum(x, y):
    return custom_op2(x, y, maximum, lambda x_, y_: choose_backend(x_, y_).maximum(x_, y_))


def minimum(x, y):
    return custom_op2(x, y, minimum, lambda x_, y_: choose_backend(x_, y_).minimum(x_, y_))


def clip(x, minimum, maximum):
    def _clip(x, minimum, maximum):
        new_shape, (x_, min_, max_) = broadcastable_native_tensors(*tensors(x, minimum, maximum))
        result_tensor = choose_backend(x_, min_, max_).clip(x_, min_, max_)
        return NativeTensor(result_tensor, new_shape)
    return broadcast_op(_clip, tensors(x, minimum, maximum))


def with_custom_gradient(function, inputs, gradient, input_index=0, output_index=None, name_base='custom_gradient_func'):
    raise NotImplementedError()


def conv(value: Tensor, kernel: Tensor, padding='same'):
    raise NotImplementedError()


def unstack(value: Tensor, dim=0):
    assert isinstance(value, Tensor)
    return value.unstack(value.shape.names[dim])


def boolean_mask(x: Tensor, mask):
    raise NotImplementedError()


def gather(value: Tensor, indices: Tensor):
    v_ = value.native()
    i_ = indices.native()
    backend = choose_backend(v_, i_)
    if value.shape.channel_rank == 0:
        v_ = backend.expand_dims(v_, -1)
    result = backend.gather_nd(v_, i_, batch_dims=value.shape.batch_rank)
    if value.shape.channel_rank == 0:
        result = result[..., 0]
    new_shape = value.shape.batch & indices.shape.non_channel & value.shape.channel
    return NativeTensor(result, new_shape)


def scatter(indices: Tensor, values: Tensor, size: Shape, scatter_dims, duplicates_handling='undefined', outside_handling='discard'):
    """
    Create a dense tensor from sparse values.

    Args:
      indices: n-dimensional indices corresponding to values
      values: values to scatter at indices
      size: spatial size of dense tensor
      scatter_dims: dimensions of values/indices to reduce during scattering
      duplicates_handling: one of ('undefined', 'add', 'mean', 'any') (Default value = 'undefined')
      outside_handling: one of ('discard', 'clamp', 'undefined') (Default value = 'discard')
      indices: Tensor: 
      values: Tensor: 
      size: Shape: 

    Returns:

    """
    indices_ = indices.native()
    values_ = values.native(values.shape.combined(indices.shape.non_channel).names)
    backend = choose_backend(indices_, values_)
    result_ = backend.scatter(indices_, values_, tuple(size), duplicates_handling=duplicates_handling, outside_handling=outside_handling)
    result_shape = size & indices.shape.batch & values.shape.non_spatial
    result_shape = result_shape.without(scatter_dims)
    return NativeTensor(result_, result_shape)


def fft(x: Tensor):
    """
    Performs a fast Fourier transform (FFT) on all spatial dimensions of x.
    
    The inverse operation is :func:`ifft`.

    Args:
      x: tensor of type float or complex
      x: Tensor: 

    Returns:
      FFT(x) of type complex

    """
    native, assemble = _invertible_standard_form(x)
    result = choose_backend(native).fft(native)
    return assemble(result)


def ifft(k: Tensor):
    native, assemble = _invertible_standard_form(k)
    result = choose_backend(native).ifft(native)
    return assemble(result)


def dtype(x):
    if isinstance(x, Tensor):
        return x.dtype
    else:
        return choose_backend(x).dtype(x)


def tile(value, multiples):
    raise NotImplementedError()


def expand_channel(value: Tensor, dim_name: str, dim_size: int = 1):
    return _expand(value, dim_name, dim_size, CHANNEL_DIM)


def _expand(value: Tensor, dim_name: str, dim_size: int, dim_type: str):
    value = tensor(value)
    new_shape = value.shape.expand(dim_size, dim_name, dim_type)
    if isinstance(value, CollapsedTensor):
        return CollapsedTensor(value.tensor, new_shape)
    else:
        return CollapsedTensor(value, new_shape)


def sparse_tensor(indices, values, shape):
    raise NotImplementedError()


def _invertible_standard_form(value: Tensor):
    """
    Reshapes the tensor into the shape (batch, spatial..., channel) with a single batch and channel dimension.

    Args:
      value: tensor to reshape
      value: Tensor: 

    Returns:
      reshaped native tensor, inverse function

    """
    normal_order = value.shape.normal_order()
    native = value.native(normal_order.names)
    backend = choose_backend(native)
    standard_form = (value.shape.batch.volume,) + value.shape.spatial.sizes + (value.shape.channel.volume,)
    reshaped = backend.reshape(native, standard_form)

    def assemble(reshaped):
        un_reshaped = backend.reshape(reshaped, backend.shape(native))
        return NativeTensor(un_reshaped, normal_order)

    return reshaped, assemble


def close(*tensors, rel_tolerance=1e-5, abs_tolerance=0):
    """
    Checks whether all tensors have equal values within the specified tolerance.
    
    Does not check that the shapes exactly match.
    Tensors with different shapes are reshaped before comparing.

    Args:
      tensors: tensor or tensor-like (constant) each
      rel_tolerance: relative tolerance (Default value = 1e-5)
      abs_tolerance: absolute tolerance (Default value = 0)
      *tensors: 

    Returns:

    """
    tensors = [tensor(t) for t in tensors]
    for other in tensors[1:]:
        if not _close(tensors[0], other, rel_tolerance=rel_tolerance, abs_tolerance=abs_tolerance):
            return False
    return True


def _close(tensor1, tensor2, rel_tolerance=1e-5, abs_tolerance=0):
    if tensor2 is tensor1:
        return True
    new_shape, (native1, native2) = broadcastable_native_tensors(tensor1, tensor2)
    np1 = choose_backend(native1).numpy(native1)
    np2 = choose_backend(native2).numpy(native2)
    return np.allclose(np1, np2, rel_tolerance, abs_tolerance)


def assert_close(*tensors, rel_tolerance=1e-5, abs_tolerance=0):
    """
    Checks that all tensors have equal values within the specified tolerance.
    Raises an AssertionError if the values of this tensor are not within tolerance of any of the other tensors.
    
    Does not check that the shapes exactly match.
    Tensors with different shapes are reshaped before comparing.

    Args:
      tensors: tensor or tensor-like (constant) each
      rel_tolerance: relative tolerance (Default value = 1e-5)
      abs_tolerance: absolute tolerance (Default value = 0)
      *tensors: 

    Returns:

    """
    any_tensor = next(filter(lambda t: isinstance(t, Tensor), tensors))
    if any_tensor is None:
        tensors = [tensor(t) for t in tensors]
    else:  # use Tensor to infer dimensions
        tensors = [any_tensor._tensor(t) for t in tensors]
    tensors = [t.tensor if isinstance(t, CollapsedTensor) else t for t in tensors]
    for other in tensors[1:]:
        _assert_close(tensors[0], other, rel_tolerance=rel_tolerance, abs_tolerance=abs_tolerance)


def _assert_close(tensor1, tensor2, rel_tolerance=1e-5, abs_tolerance=0):
    if tensor2 is tensor1:
        return
    if isinstance(tensor2, (int, float, bool)):
        np.testing.assert_allclose(tensor1.numpy(), tensor2, rel_tolerance, abs_tolerance)

    def inner_assert_close(tensor1, tensor2):
        new_shape, (native1, native2) = broadcastable_native_tensors(tensor1, tensor2)
        np1 = choose_backend(native1).numpy(native1)
        np2 = choose_backend(native2).numpy(native2)
        if not np.allclose(np1, np2, rel_tolerance, abs_tolerance):
            np.testing.assert_allclose(np1, np2, rel_tolerance, abs_tolerance)

    broadcast_op(inner_assert_close, [tensor1, tensor2], no_return=True)


def solve(operator, y: Tensor, x0: Tensor, solve_params: Solve, callback=None):
    if not isinstance(solve_params, LinearSolve):
        raise NotImplementedError("Only linear solve is currently supported. Pass a LinearSolve object")
    if solve_params.solver not in (None, 'CG'):
        raise NotImplementedError("Only 'CG' solver currently supported")

    from ._track import lin_placeholder, ShiftLinOp
    x0, y = tensors(x0, y)
    batch = (y.shape & x0.shape).batch
    backend = choose_backend(x0.native(), y.native())
    x0_native = backend.reshape(x0.native(), (x0.shape.batch.volume, x0.shape.non_batch.volume))
    y_native = backend.reshape(y.native(), (y.shape.batch.volume, y.shape.non_batch.volume))
    if callable(operator):
        operator_or_matrix = None
        if solve_params.solver_arguments['bake'] == 'sparse':
            track_time = time.time()
            x_track = lin_placeholder(x0)
            Ax_track = operator(x_track)
            assert isinstance(Ax_track, ShiftLinOp), 'Baking sparse matrix failed. Make sure only supported linear operations are used.'
            track_time = time.time() - track_time
            build_time = time.time()
            operator_or_matrix = Ax_track.build_sparse_coordinate_matrix()
            # TODO reshape x0, y so that independent dimensions are batch
            build_time = time.time() - build_time
            # print_(tensor(operator_or_matrix.todense(), names='x,y'))
        if operator_or_matrix is None:
            def operator_or_matrix(native_x):
                native_x_shaped = backend.reshape(native_x, x0.shape.non_batch.sizes)
                x = NativeTensor(native_x_shaped, x0.shape.non_batch)
                Ax = operator(x)
                Ax_native = backend.reshape(Ax.native(), backend.shape(native_x))
                return Ax_native
    else:
        operator_or_matrix = backend.reshape(operator.native(), (y.shape.non_batch.volume, x0.shape.non_batch.volume))

    loop_time = time.time()
    converged, x, iterations = backend.conjugate_gradient(operator_or_matrix, y_native, x0_native, solve_params.relative_tolerance, solve_params.absolute_tolerance, solve_params.max_iterations, 'implicit', callback)
    loop_time = time.time() - loop_time
    print(f"CG   track: {round(track_time * 1000)} ms  \tbuild: {round(build_time * 1000)} ms  \tloop: {round(loop_time * 1000)} ms / {iterations} iterations")
    converged = choose_backend(converged).reshape(converged, batch.sizes)
    x = backend.reshape(x, batch.sizes + x0.shape.non_batch.sizes)
    iterations = choose_backend(iterations).reshape(iterations, batch.sizes)
    return NativeTensor(converged, batch), NativeTensor(x, batch.combined(x0.shape.non_batch)), NativeTensor(iterations, batch)
