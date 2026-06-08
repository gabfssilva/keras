import functools

import ml_dtypes
import mlx.core as mx

from keras.src import backend
from keras.src.backend.mlx.core import _flip
from keras.src.backend.mlx.core import convert_to_tensor
from keras.src.random.seed_generator import draw_seed

RESIZE_INTERPOLATIONS = (
    "bilinear",
    "nearest",
    "lanczos3",
    "lanczos5",
    "bicubic",
)
AFFINE_TRANSFORM_INTERPOLATIONS = {  # map to order
    "nearest": 0,
    "bilinear": 1,
}
AFFINE_TRANSFORM_FILL_MODES = {
    "constant",
    "nearest",
    "wrap",
    "mirror",
    "reflect",
}
MAP_COORDINATES_FILL_MODES = {
    "constant",
    "nearest",
    "wrap",
    "mirror",
    "reflect",
}
SCALE_AND_TRANSLATE_METHODS = {
    "linear",
    "bilinear",
    "trilinear",
    "cubic",
    "bicubic",
    "tricubic",
    "lanczos3",
    "lanczos5",
}


def rgb_to_grayscale(images, data_format=None):
    images = convert_to_tensor(images)
    data_format = backend.standardize_data_format(data_format)
    channels_axis = -1 if data_format == "channels_last" else -3
    if len(images.shape) not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )
    original_dtype = images.dtype
    compute_dtype = backend.result_type(images.dtype, float)
    images = images.astype(mx.float32)

    rgb_weights = mx.array([0.2989, 0.5870, 0.1140], dtype=mx.float32)
    # tensordot along the channel axis
    grayscales = mx.tensordot(images, rgb_weights, axes=(channels_axis, -1))
    grayscales = mx.expand_dims(grayscales, axis=channels_axis)
    return grayscales.astype(original_dtype)


def rgb_to_hsv(images, data_format=None):
    images = convert_to_tensor(images)
    dtype = backend.standardize_dtype(images.dtype)
    data_format = backend.standardize_data_format(data_format)
    channels_axis = -1 if data_format == "channels_last" else -3
    if len(images.shape) not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )
    if not backend.is_float_dtype(dtype):
        raise ValueError(
            "Invalid images dtype: expected float dtype. "
            f"Received: images.dtype={dtype}"
        )
    eps = ml_dtypes.finfo(dtype).eps
    images = mx.where(mx.abs(images) < eps, 0.0, images)
    red, green, blue = mx.split(images, 3, channels_axis)
    red = mx.squeeze(red, axis=channels_axis)
    green = mx.squeeze(green, axis=channels_axis)
    blue = mx.squeeze(blue, axis=channels_axis)

    def rgb_planes_to_hsv_planes(r, g, b):
        value = mx.maximum(mx.maximum(r, g), b)
        minimum = mx.minimum(mx.minimum(r, g), b)
        range_ = value - minimum

        safe_value = mx.where(value > 0, value, 1.0)
        safe_range = mx.where(range_ > 0, range_, 1.0)

        saturation = mx.where(value > 0, range_ / safe_value, 0.0)
        norm = 1.0 / (6.0 * safe_range)

        hue = mx.where(
            value == g,
            norm * (b - r) + 2.0 / 6.0,
            norm * (r - g) + 4.0 / 6.0,
        )
        hue = mx.where(value == r, norm * (g - b), hue)
        hue = mx.where(range_ > 0, hue, 0.0) + (hue < 0.0).astype(
            hue.dtype
        )
        return hue, saturation, value

    images = mx.stack(
        rgb_planes_to_hsv_planes(red, green, blue), axis=channels_axis
    )
    return images.astype(dtype)


def hsv_to_rgb(images, data_format=None):
    images = convert_to_tensor(images)
    dtype = images.dtype
    data_format = backend.standardize_data_format(data_format)
    channels_axis = -1 if data_format == "channels_last" else -3
    if len(images.shape) not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )
    if not backend.is_float_dtype(dtype):
        raise ValueError(
            "Invalid images dtype: expected float dtype. "
            f"Received: images.dtype={backend.standardize_dtype(dtype)}"
        )
    hue, saturation, value = mx.split(images, 3, channels_axis)
    hue = mx.squeeze(hue, axis=channels_axis)
    saturation = mx.squeeze(saturation, axis=channels_axis)
    value = mx.squeeze(value, axis=channels_axis)

    def hsv_planes_to_rgb_planes(hue, saturation, value):
        dh = (hue % 1.0) * 6.0
        dr = mx.clip(mx.abs(dh - 3.0) - 1.0, 0.0, 1.0)
        dg = mx.clip(2.0 - mx.abs(dh - 2.0), 0.0, 1.0)
        db = mx.clip(2.0 - mx.abs(dh - 4.0), 0.0, 1.0)
        one_minus_s = 1.0 - saturation

        red = value * (one_minus_s + saturation * dr)
        green = value * (one_minus_s + saturation * dg)
        blue = value * (one_minus_s + saturation * db)
        return red, green, blue

    images = mx.stack(
        hsv_planes_to_rgb_planes(hue, saturation, value), axis=channels_axis
    )
    return images.astype(dtype)


def resize(
    images,
    size,
    interpolation="bilinear",
    antialias=False,
    crop_to_aspect_ratio=False,
    pad_to_aspect_ratio=False,
    fill_mode="constant",
    fill_value=0.0,
    data_format=None,
):
    data_format = backend.standardize_data_format(data_format)
    if interpolation not in RESIZE_INTERPOLATIONS:
        raise ValueError(
            "Invalid value for argument `interpolation`. Expected of one "
            f"{RESIZE_INTERPOLATIONS}. Received: "
            f"interpolation={interpolation}"
        )
    if fill_mode != "constant":
        raise ValueError(
            "Invalid value for argument `fill_mode`. Only `'constant'` "
            f"is supported. Received: fill_mode={fill_mode}"
        )
    if pad_to_aspect_ratio and crop_to_aspect_ratio:
        raise ValueError(
            "Only one of `pad_to_aspect_ratio` & `crop_to_aspect_ratio` "
            "can be `True`."
        )
    if not len(size) == 2:
        raise ValueError(
            "Argument `size` must be a tuple of two elements "
            f"(height, width). Received: size={size}"
        )
    size = tuple(size)
    target_height, target_width = size
    if len(images.shape) == 4:
        if data_format == "channels_last":
            size = (images.shape[0],) + size + (images.shape[-1],)
        else:
            size = (images.shape[0], images.shape[1]) + size
    elif len(images.shape) == 3:
        if data_format == "channels_last":
            size = size + (images.shape[-1],)
        else:
            size = (images.shape[0],) + size
    else:
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )

    if crop_to_aspect_ratio:
        shape = images.shape
        if data_format == "channels_last":
            height, width = shape[-3], shape[-2]
        else:
            height, width = shape[-2], shape[-1]
        crop_height = int(float(width * target_height) / target_width)
        crop_height = max(min(height, crop_height), 1)
        crop_width = int(float(height * target_width) / target_height)
        crop_width = max(min(width, crop_width), 1)
        crop_box_hstart = int(float(height - crop_height) / 2)
        crop_box_wstart = int(float(width - crop_width) / 2)
        if data_format == "channels_last":
            if len(images.shape) == 4:
                images = images[
                    :,
                    crop_box_hstart : crop_box_hstart + crop_height,
                    crop_box_wstart : crop_box_wstart + crop_width,
                    :,
                ]
            else:
                images = images[
                    crop_box_hstart : crop_box_hstart + crop_height,
                    crop_box_wstart : crop_box_wstart + crop_width,
                    :,
                ]
        else:
            if len(images.shape) == 4:
                images = images[
                    :,
                    :,
                    crop_box_hstart : crop_box_hstart + crop_height,
                    crop_box_wstart : crop_box_wstart + crop_width,
                ]
            else:
                images = images[
                    :,
                    crop_box_hstart : crop_box_hstart + crop_height,
                    crop_box_wstart : crop_box_wstart + crop_width,
                ]
    elif pad_to_aspect_ratio:
        shape = images.shape
        batch_size = images.shape[0]
        if data_format == "channels_last":
            height, width, channels = shape[-3], shape[-2], shape[-1]
        else:
            channels, height, width = shape[-3], shape[-2], shape[-1]
        pad_height = int(float(width * target_height) / target_width)
        pad_height = max(height, pad_height)
        pad_width = int(float(height * target_width) / target_height)
        pad_width = max(width, pad_width)
        img_box_hstart = int(float(pad_height - height) / 2)
        img_box_wstart = int(float(pad_width - width) / 2)

        if data_format == "channels_last":
            if img_box_hstart > 0:
                if len(images.shape) == 4:
                    padded_img = mx.concatenate(
                        [
                            mx.full(
                                (batch_size, img_box_hstart, width, channels),
                                fill_value,
                                dtype=images.dtype,
                            ),
                            images,
                            mx.full(
                                (batch_size, img_box_hstart, width, channels),
                                fill_value,
                                dtype=images.dtype,
                            ),
                        ],
                        axis=1,
                    )
                else:
                    padded_img = mx.concatenate(
                        [
                            mx.full(
                                (img_box_hstart, width, channels),
                                fill_value,
                                dtype=images.dtype,
                            ),
                            images,
                            mx.full(
                                (img_box_hstart, width, channels),
                                fill_value,
                                dtype=images.dtype,
                            ),
                        ],
                        axis=0,
                    )
            elif img_box_wstart > 0:
                if len(images.shape) == 4:
                    padded_img = mx.concatenate(
                        [
                            mx.full(
                                (batch_size, height, img_box_wstart, channels),
                                fill_value,
                                dtype=images.dtype,
                            ),
                            images,
                            mx.full(
                                (batch_size, height, img_box_wstart, channels),
                                fill_value,
                                dtype=images.dtype,
                            ),
                        ],
                        axis=2,
                    )
                else:
                    padded_img = mx.concatenate(
                        [
                            mx.full(
                                (height, img_box_wstart, channels),
                                fill_value,
                                dtype=images.dtype,
                            ),
                            images,
                            mx.full(
                                (height, img_box_wstart, channels),
                                fill_value,
                                dtype=images.dtype,
                            ),
                        ],
                        axis=1,
                    )
            else:
                padded_img = images
        else:
            if img_box_hstart > 0:
                if len(images.shape) == 4:
                    padded_img = mx.concatenate(
                        [
                            mx.full(
                                (batch_size, channels, img_box_hstart, width),
                                fill_value,
                                dtype=images.dtype,
                            ),
                            images,
                            mx.full(
                                (batch_size, channels, img_box_hstart, width),
                                fill_value,
                                dtype=images.dtype,
                            ),
                        ],
                        axis=2,
                    )
                else:
                    padded_img = mx.concatenate(
                        [
                            mx.full(
                                (channels, img_box_hstart, width),
                                fill_value,
                                dtype=images.dtype,
                            ),
                            images,
                            mx.full(
                                (channels, img_box_hstart, width),
                                fill_value,
                                dtype=images.dtype,
                            ),
                        ],
                        axis=1,
                    )
            elif img_box_wstart > 0:
                if len(images.shape) == 4:
                    padded_img = mx.concatenate(
                        [
                            mx.full(
                                (batch_size, channels, height, img_box_wstart),
                                fill_value,
                                dtype=images.dtype,
                            ),
                            images,
                            mx.full(
                                (batch_size, channels, height, img_box_wstart),
                                fill_value,
                                dtype=images.dtype,
                            ),
                        ],
                        axis=3,
                    )
                else:
                    padded_img = mx.concatenate(
                        [
                            mx.full(
                                (channels, height, img_box_wstart),
                                fill_value,
                                dtype=images.dtype,
                            ),
                            images,
                            mx.full(
                                (channels, height, img_box_wstart),
                                fill_value,
                                dtype=images.dtype,
                            ),
                        ],
                        axis=2,
                    )
            else:
                padded_img = images
        images = padded_img

    return _resize(images, size, method=interpolation, antialias=antialias)


# ---------------------------------------------------------------------------
# Resize internals
# ---------------------------------------------------------------------------


def _compute_weight_mat(
    input_size, output_size, scale, translation, kernel, antialias
):
    inv_scale = 1.0 / scale
    kernel_scale = max(inv_scale, 1.0) if antialias else 1.0

    sample_f = (
        (mx.arange(output_size).astype(mx.float32) + 0.5) * inv_scale
        - translation * inv_scale
        - 0.5
    )

    x = (
        mx.abs(
            mx.expand_dims(sample_f, 0)
            - mx.expand_dims(mx.arange(input_size).astype(mx.float32), 1)
        )
        / kernel_scale
    )

    weights = kernel(x)

    total_weight_sum = mx.sum(weights, axis=0, keepdims=True)
    eps = mx.finfo(mx.float32).eps
    weights = mx.where(
        mx.abs(total_weight_sum) > 1000.0 * eps,
        weights / mx.where(total_weight_sum != 0, total_weight_sum, 1),
        0,
    )

    input_size_minus_0_5 = input_size - 0.5
    return mx.where(
        mx.expand_dims(
            mx.logical_and(sample_f >= -0.5, sample_f <= input_size_minus_0_5),
            0,
        ),
        weights,
        0,
    )


def _resize(image, shape, method, antialias):
    if method == "nearest":
        return _resize_nearest(image, shape)
    else:
        kernel = _kernels.get(method, None)
    if kernel is None:
        raise ValueError("Unknown resize method")

    spatial_dims = tuple(
        i for i in range(len(shape)) if image.shape[i] != shape[i]
    )
    scale = [
        shape[d] / image.shape[d] if image.shape[d] != 0 else 1.0
        for d in spatial_dims
    ]

    return _scale_and_translate(
        image,
        shape,
        spatial_dims,
        scale,
        [0.0] * len(spatial_dims),
        kernel,
        antialias,
    )


def _resize_nearest(x, output_shape):
    input_shape = x.shape
    spatial_dims = tuple(
        i
        for i in range(len(input_shape))
        if input_shape[i] != output_shape[i]
    )

    for d in spatial_dims:
        m, n = input_shape[d], output_shape[d]
        offsets = (mx.arange(n).astype(mx.float32) + 0.5) * m / n
        offsets = mx.floor(offsets).astype(mx.int32)
        x = mx.take(x, offsets, axis=d)
    return x


def _fill_triangle_kernel(x):
    return mx.maximum(mx.array(0.0), 1 - mx.abs(x))


def _fill_keys_cubic_kernel(x):
    out = ((1.5 * x - 2.5) * x) * x + 1.0
    out = mx.where(x >= 1.0, ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0, out)
    return mx.where(x >= 2.0, mx.array(0.0), out)


def _fill_lanczos_kernel(radius, x):
    import math as _math
    pi = _math.pi
    y = radius * mx.sin(pi * x) * mx.sin(pi * x / radius)
    out = mx.where(
        x > 1e-3,
        y / mx.where(x != 0, pi**2 * x**2, 1),
        1,
    )
    return mx.where(x > radius, mx.array(0.0), out)


_kernels = {
    "linear": _fill_triangle_kernel,
    "bilinear": _fill_triangle_kernel,
    "cubic": _fill_keys_cubic_kernel,
    "bicubic": _fill_keys_cubic_kernel,
    "lanczos3": lambda x: _fill_lanczos_kernel(3.0, x),
    "lanczos5": lambda x: _fill_lanczos_kernel(5.0, x),
}


def _scale_and_translate(
    x, output_shape, spatial_dims, scale, translation, kernel, antialias
):
    input_shape = x.shape

    if len(spatial_dims) == 0:
        return x

    is_integer = x.dtype in (
        mx.int8,
        mx.int16,
        mx.int32,
        mx.int64,
        mx.uint8,
        mx.uint16,
        mx.uint32,
        mx.uint64,
    )
    if is_integer:
        output = x.astype(mx.float32)
        use_rounding = True
    else:
        output = x
        use_rounding = False

    for i, d in enumerate(spatial_dims):
        d = d % x.ndim
        m, n = input_shape[d], output_shape[d]

        w = _compute_weight_mat(
            m, n, scale[i], translation[i], kernel, antialias
        ).astype(output.dtype)
        output = mx.tensordot(output, w, axes=[[d], [0]])
        output = mx.moveaxis(output, -1, d)

    if use_rounding:
        x_min = x.min()
        x_max = x.max()
        output = mx.clip(mx.round(output), x_min.astype(output.dtype), x_max.astype(output.dtype))
        output = output.astype(x.dtype)
    return output


# ---------------------------------------------------------------------------
# Affine transform
# ---------------------------------------------------------------------------


def affine_transform(
    images,
    transform,
    interpolation="bilinear",
    fill_mode="constant",
    fill_value=0,
    data_format=None,
):
    data_format = backend.standardize_data_format(data_format)
    if interpolation not in AFFINE_TRANSFORM_INTERPOLATIONS.keys():
        raise ValueError(
            "Invalid value for argument `interpolation`. Expected of one "
            f"{set(AFFINE_TRANSFORM_INTERPOLATIONS.keys())}. Received: "
            f"interpolation={interpolation}"
        )
    if fill_mode not in AFFINE_TRANSFORM_FILL_MODES:
        raise ValueError(
            "Invalid value for argument `fill_mode`. Expected of one "
            f"{AFFINE_TRANSFORM_FILL_MODES}. Received: "
            f"fill_mode={fill_mode}"
        )

    images = convert_to_tensor(images)
    transform = convert_to_tensor(transform)

    if len(images.shape) not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )
    if len(transform.shape) not in (1, 2):
        raise ValueError(
            "Invalid transform rank: expected rank 1 (single transform) "
            "or rank 2 (batch of transforms). Received input with shape: "
            f"transform.shape={transform.shape}"
        )

    input_dtype = backend.standardize_dtype(images.dtype)
    compute_dtype = backend.result_type(input_dtype, "float32")
    images = images.astype(compute_dtype)
    transform = transform.astype(compute_dtype)

    need_squeeze = False
    if len(images.shape) == 3:
        images = mx.expand_dims(images, axis=0)
        need_squeeze = True
    if len(transform.shape) == 1:
        transform = mx.expand_dims(transform, axis=0)

    if data_format == "channels_first":
        images = mx.transpose(images, (0, 2, 3, 1))

    batch_size = images.shape[0]

    # Build coordinate grid in MLX
    grids = mx.meshgrid(
        *[mx.arange(s) for s in images.shape[1:]], indexing="ij"
    )
    indices = mx.stack(grids, axis=-1).astype(mx.float32)
    indices = mx.broadcast_to(
        mx.expand_dims(indices, 0),
        (batch_size,) + indices.shape,
    )

    # Swap values to match numpy backend convention
    a0 = transform[:, 0]
    a2 = transform[:, 2]
    b1 = transform[:, 4]
    b2 = transform[:, 5]
    # Build swapped transform: [b1, t1, b2, t3, a0, a2, t6, t7]
    transform_swapped = mx.stack([
        b1, transform[:, 1], b2, transform[:, 3],
        a0, a2, transform[:, 6], transform[:, 7],
    ], axis=-1)

    transform_swapped = mx.pad(
        transform_swapped, [(0, 0), (0, 1)], constant_values=1
    )
    transform_swapped = mx.reshape(transform_swapped, (batch_size, 3, 3))
    offset = transform_swapped[:, 0:2, 2]
    offset = mx.pad(offset, [(0, 0), (0, 1)])
    # Zero out the offset in the transform matrix
    transform_swapped = mx.concatenate([
        mx.concatenate([transform_swapped[:, 0:2, 0:2],
                        mx.zeros((batch_size, 2, 1))], axis=2),
        transform_swapped[:, 2:3, :],
    ], axis=1)

    transform_mx = transform_swapped.astype(mx.float32)
    offset_mx = offset.astype(mx.float32)

    # transform the indices
    coordinates = mx.einsum("Bhwij, Bjk -> Bhwik", indices, transform_mx)
    coordinates = mx.moveaxis(coordinates, source=-1, destination=1)
    coordinates = coordinates + mx.reshape(
        offset_mx, (*offset_mx.shape, 1, 1, 1)
    )

    # apply affine transformation
    affined = mx.stack(
        [
            map_coordinates(
                images[i],
                coordinates[i],
                order=AFFINE_TRANSFORM_INTERPOLATIONS[interpolation],
                fill_mode=fill_mode,
                fill_value=fill_value,
            )
            for i in range(batch_size)
        ],
        axis=0,
    )

    if data_format == "channels_first":
        affined = mx.transpose(affined, (0, 3, 1, 2))
    if need_squeeze:
        affined = mx.squeeze(affined, axis=0)
    return affined.astype(input_dtype)


# ---------------------------------------------------------------------------
# Perspective transform
# ---------------------------------------------------------------------------


def perspective_transform(
    images,
    start_points,
    end_points,
    interpolation="bilinear",
    fill_value=0,
    data_format=None,
):
    data_format = backend.standardize_data_format(data_format)
    start_points = convert_to_tensor(start_points)
    end_points = convert_to_tensor(end_points)

    if interpolation not in AFFINE_TRANSFORM_INTERPOLATIONS:
        raise ValueError(
            "Invalid value for argument `interpolation`. Expected of one "
            f"{AFFINE_TRANSFORM_INTERPOLATIONS}. Received: "
            f"interpolation={interpolation}"
        )

    if len(images.shape) not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )

    if start_points.ndim not in (2, 3) or start_points.shape[-2:] != (4, 2):
        raise ValueError(
            "Invalid start_points shape: expected (4,2) for a single image"
            f" or (N,4,2) for a batch. Received shape: {start_points.shape}"
        )
    if end_points.ndim not in (2, 3) or end_points.shape[-2:] != (4, 2):
        raise ValueError(
            "Invalid end_points shape: expected (4,2) for a single image"
            f" or (N,4,2) for a batch. Received shape: {end_points.shape}"
        )
    if start_points.shape != end_points.shape:
        raise ValueError(
            "start_points and end_points must have the same shape."
            f" Received start_points.shape={start_points.shape}, "
            f"end_points.shape={end_points.shape}"
        )

    input_dtype = images.dtype
    if backend.standardize_dtype(input_dtype) == "float16":
        images = images.astype(mx.float32)

    need_squeeze = False
    if len(images.shape) == 3:
        images = mx.expand_dims(images, axis=0)
        need_squeeze = True

    if len(start_points.shape) == 2:
        start_points = mx.expand_dims(start_points, axis=0)
    if len(end_points.shape) == 2:
        end_points = mx.expand_dims(end_points, axis=0)

    if data_format == "channels_first":
        images = mx.transpose(images, (0, 2, 3, 1))

    batch_size, height, width, channels = images.shape

    transforms = compute_homography_matrix(start_points, end_points)

    if len(transforms.shape) == 1:
        transforms = mx.expand_dims(transforms, axis=0)
    if transforms.shape[0] == 1 and batch_size > 1:
        transforms = mx.tile(transforms, (batch_size, 1))

    x, y = mx.meshgrid(
        mx.arange(width).astype(mx.float32),
        mx.arange(height).astype(mx.float32),
        indexing="xy",
    )

    results = []
    for i in range(batch_size):
        a0 = transforms[i, 0]
        a1 = transforms[i, 1]
        a2 = transforms[i, 2]
        a3 = transforms[i, 3]
        a4 = transforms[i, 4]
        a5 = transforms[i, 5]
        a6 = transforms[i, 6]
        a7 = transforms[i, 7]
        denom = a6 * x + a7 * y + 1.0
        x_in = (a0 * x + a1 * y + a2) / denom
        y_in = (a3 * x + a4 * y + a5) / denom

        coords = mx.stack(
            [mx.reshape(y_in, (-1,)), mx.reshape(x_in, (-1,))], axis=0
        )

        mapped_channels = []
        for channel in range(channels):
            channel_img = images[i, :, :, channel]
            mapped_channel = map_coordinates(
                channel_img,
                coords,
                order=AFFINE_TRANSFORM_INTERPOLATIONS[interpolation],
                fill_mode="constant",
                fill_value=fill_value,
            )
            mapped_channels.append(mx.reshape(mapped_channel, (height, width)))
        results.append(mx.stack(mapped_channels, axis=-1))

    output = mx.stack(results, axis=0)

    if data_format == "channels_first":
        output = mx.transpose(output, (0, 3, 1, 2))
    if need_squeeze:
        output = mx.squeeze(output, axis=0)
    output = output.astype(input_dtype)
    return output


def compute_homography_matrix(start_points, end_points):
    start_points = convert_to_tensor(start_points)
    end_points = convert_to_tensor(end_points)
    dtype = backend.result_type(start_points.dtype, end_points.dtype, float)
    compute_dtype = backend.result_type(dtype, "float32")

    sp = start_points.astype(mx.float32)
    ep = end_points.astype(mx.float32)

    start_x1, start_y1 = sp[:, 0, 0], sp[:, 0, 1]
    start_x2, start_y2 = sp[:, 1, 0], sp[:, 1, 1]
    start_x3, start_y3 = sp[:, 2, 0], sp[:, 2, 1]
    start_x4, start_y4 = sp[:, 3, 0], sp[:, 3, 1]

    end_x1, end_y1 = ep[:, 0, 0], ep[:, 0, 1]
    end_x2, end_y2 = ep[:, 1, 0], ep[:, 1, 1]
    end_x3, end_y3 = ep[:, 2, 0], ep[:, 2, 1]
    end_x4, end_y4 = ep[:, 3, 0], ep[:, 3, 1]

    z = mx.zeros_like(end_x1)
    o = mx.ones_like(end_x1)

    coefficient_matrix = mx.stack(
        [
            mx.stack(
                [end_x1, end_y1, o, z, z, z,
                 -start_x1 * end_x1, -start_x1 * end_y1], axis=-1
            ),
            mx.stack(
                [z, z, z, end_x1, end_y1, o,
                 -start_y1 * end_x1, -start_y1 * end_y1], axis=-1
            ),
            mx.stack(
                [end_x2, end_y2, o, z, z, z,
                 -start_x2 * end_x2, -start_x2 * end_y2], axis=-1
            ),
            mx.stack(
                [z, z, z, end_x2, end_y2, o,
                 -start_y2 * end_x2, -start_y2 * end_y2], axis=-1
            ),
            mx.stack(
                [end_x3, end_y3, o, z, z, z,
                 -start_x3 * end_x3, -start_x3 * end_y3], axis=-1
            ),
            mx.stack(
                [z, z, z, end_x3, end_y3, o,
                 -start_y3 * end_x3, -start_y3 * end_y3], axis=-1
            ),
            mx.stack(
                [end_x4, end_y4, o, z, z, z,
                 -start_x4 * end_x4, -start_x4 * end_y4], axis=-1
            ),
            mx.stack(
                [z, z, z, end_x4, end_y4, o,
                 -start_y4 * end_x4, -start_y4 * end_y4], axis=-1
            ),
        ],
        axis=1,
    )

    target_vector = mx.stack(
        [start_x1, start_y1, start_x2, start_y2,
         start_x3, start_y3, start_x4, start_y4],
        axis=-1,
    )
    target_vector = mx.expand_dims(target_vector, axis=-1)

    homography_matrix = mx.linalg.solve(
        coefficient_matrix, target_vector, stream=mx.cpu
    )
    homography_matrix = mx.reshape(homography_matrix, [-1, 8])
    return homography_matrix.astype(compute_dtype)


# ---------------------------------------------------------------------------
# map_coordinates
# ---------------------------------------------------------------------------


def map_coordinates(
    inputs, coordinates, order, fill_mode="constant", fill_value=0.0
):
    inputs = convert_to_tensor(inputs)
    coordinates = convert_to_tensor(coordinates)
    if coordinates.shape[0] != len(inputs.shape):
        raise ValueError(
            "First dim of `coordinates` must be the same as the rank of "
            "`inputs`. "
            f"Received inputs with shape: {inputs.shape} and coordinate "
            f"leading dim of {coordinates.shape[0]}"
        )
    if len(coordinates.shape) < 2:
        raise ValueError(
            "Invalid coordinates rank: expected at least rank 2."
            f" Received input with shape: {coordinates.shape}"
        )
    if fill_mode not in MAP_COORDINATES_FILL_MODES:
        raise ValueError(
            "Invalid value for argument `fill_mode`. Expected one of "
            f"{MAP_COORDINATES_FILL_MODES}. Received: "
            f"fill_mode={fill_mode}"
        )
    if order not in range(2):
        raise ValueError(
            "Invalid value for argument `order`. Expected one of "
            f"{[0, 1]}. Received: order={order}"
        )

    # Pure MLX implementation of map_coordinates
    ndim = len(inputs.shape)
    input_dtype = backend.standardize_dtype(inputs.dtype)
    compute_dtype = backend.result_type(input_dtype, "float32")
    inputs = inputs.astype(compute_dtype)
    coordinates = coordinates.astype(compute_dtype)

    def _apply_fill_mode(coord, size):
        if fill_mode == "constant":
            return coord
        elif fill_mode == "nearest":
            return mx.clip(coord, 0, size - 1)
        elif fill_mode == "wrap":
            return coord % size
        elif fill_mode == "mirror":
            # mirror: period = 2*(size-1), reflect at boundaries
            period = 2 * (size - 1) if size > 1 else 1
            coord = mx.abs(coord) % period
            return mx.where(coord >= size, period - coord, coord)
        elif fill_mode == "reflect":
            # reflect (symmetric): period = 2*size
            period = 2 * size if size > 0 else 1
            coord = mx.abs(coord) % period
            return mx.where(coord >= size, period - coord - 1, coord)
        return coord

    if order == 0:
        # Nearest-neighbor interpolation
        indices = [mx.clip(mx.round(coordinates[i]).astype(mx.int32), 0, inputs.shape[i] - 1)
                   for i in range(ndim)]
        if fill_mode != "constant":
            indices = [_apply_fill_mode(
                mx.round(coordinates[i]).astype(mx.int32), inputs.shape[i]
            ) for i in range(ndim)]
        result = inputs
        for i in range(ndim):
            result = mx.take(result, indices[i], axis=i)
            # Collapse the axis to get proper indexing
            idx = [0 if j != i else slice(None) for j in range(ndim)]
        # Use advanced indexing
        result = inputs[tuple(indices)]
        if fill_mode == "constant":
            mask = mx.ones(coordinates.shape[1:], dtype=mx.bool_)
            for i in range(ndim):
                idx = mx.round(coordinates[i]).astype(mx.int32)
                mask = mx.logical_and(mask, mx.logical_and(idx >= 0, idx < inputs.shape[i]))
            result = mx.where(mask, result, mx.array(fill_value, dtype=compute_dtype))
    else:
        # Bilinear interpolation
        coords_floor = [mx.floor(coordinates[i]).astype(mx.int32) for i in range(ndim)]
        coords_ceil = [c + 1 for c in coords_floor]
        fracs = [coordinates[i] - mx.floor(coordinates[i]) for i in range(ndim)]

        if fill_mode == "constant":
            # Clip for safe indexing, apply mask later
            def safe_idx(c, size):
                return mx.clip(c, 0, size - 1)
        else:
            def safe_idx(c, size):
                return _apply_fill_mode(c, size)

        if ndim == 2:
            y0 = safe_idx(coords_floor[0], inputs.shape[0])
            y1 = safe_idx(coords_ceil[0], inputs.shape[0])
            x0 = safe_idx(coords_floor[1], inputs.shape[1])
            x1 = safe_idx(coords_ceil[1], inputs.shape[1])
            fy, fx = fracs[0], fracs[1]

            v00 = inputs[y0, x0]
            v01 = inputs[y0, x1]
            v10 = inputs[y1, x0]
            v11 = inputs[y1, x1]

            result = (v00 * (1 - fy) * (1 - fx) + v01 * (1 - fy) * fx
                      + v10 * fy * (1 - fx) + v11 * fy * fx)

            if fill_mode == "constant":
                mask = (
                    (coords_floor[0] >= 0) & (coords_ceil[0] < inputs.shape[0])
                    & (coords_floor[1] >= 0) & (coords_ceil[1] < inputs.shape[1])
                )
                result = mx.where(mask, result, mx.array(fill_value, dtype=compute_dtype))
        elif ndim == 3:
            z0 = safe_idx(coords_floor[0], inputs.shape[0])
            z1 = safe_idx(coords_ceil[0], inputs.shape[0])
            y0 = safe_idx(coords_floor[1], inputs.shape[1])
            y1 = safe_idx(coords_ceil[1], inputs.shape[1])
            x0 = safe_idx(coords_floor[2], inputs.shape[2])
            x1 = safe_idx(coords_ceil[2], inputs.shape[2])
            fz, fy, fx = fracs[0], fracs[1], fracs[2]

            v000 = inputs[z0, y0, x0]
            v001 = inputs[z0, y0, x1]
            v010 = inputs[z0, y1, x0]
            v011 = inputs[z0, y1, x1]
            v100 = inputs[z1, y0, x0]
            v101 = inputs[z1, y0, x1]
            v110 = inputs[z1, y1, x0]
            v111 = inputs[z1, y1, x1]

            result = (
                v000 * (1 - fz) * (1 - fy) * (1 - fx)
                + v001 * (1 - fz) * (1 - fy) * fx
                + v010 * (1 - fz) * fy * (1 - fx)
                + v011 * (1 - fz) * fy * fx
                + v100 * fz * (1 - fy) * (1 - fx)
                + v101 * fz * (1 - fy) * fx
                + v110 * fz * fy * (1 - fx)
                + v111 * fz * fy * fx
            )

            if fill_mode == "constant":
                mask = (
                    (coords_floor[0] >= 0) & (coords_ceil[0] < inputs.shape[0])
                    & (coords_floor[1] >= 0) & (coords_ceil[1] < inputs.shape[1])
                    & (coords_floor[2] >= 0) & (coords_ceil[2] < inputs.shape[2])
                )
                result = mx.where(mask, result, mx.array(fill_value, dtype=compute_dtype))
        else:
            raise ValueError(
                f"map_coordinates only supports 2D and 3D inputs, got {ndim}D"
            )

    return result.astype(input_dtype)


# ---------------------------------------------------------------------------
# Gaussian blur
# ---------------------------------------------------------------------------


def gaussian_blur(
    images, kernel_size=(3, 3), sigma=(1.0, 1.0), data_format=None
):
    def _create_gaussian_kernel(kernel_size, sigma, num_channels, dtype):
        def _get_gaussian_kernel1d(size, sigma):
            x = mx.arange(int(size)) - (int(size) - 1) / 2.0
            kernel1d = mx.exp(-0.5 * (x / sigma) ** 2)
            return kernel1d / mx.sum(kernel1d)

        def _get_gaussian_kernel2d(size, sigma):
            kernel1d_x = _get_gaussian_kernel1d(size[0], sigma[0])
            kernel1d_y = _get_gaussian_kernel1d(size[1], sigma[1])
            # outer product
            return kernel1d_y[:, None] * kernel1d_x[None, :]

        kernel = _get_gaussian_kernel2d(kernel_size, sigma)
        return kernel.astype(dtype)

    images = convert_to_tensor(images)
    data_format = backend.standardize_data_format(data_format)
    input_dtype = backend.standardize_dtype(images.dtype)
    compute_dtype = backend.result_type(input_dtype, "float32")
    images = images.astype(compute_dtype)

    if len(images.shape) not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )

    need_squeeze = False
    if len(images.shape) == 3:
        images = mx.expand_dims(images, axis=0)
        need_squeeze = True

    if data_format == "channels_first":
        images = mx.transpose(images, (0, 2, 3, 1))

    batch_size, height, width, num_channels = images.shape

    kernel = _create_gaussian_kernel(
        kernel_size, sigma, num_channels, compute_dtype
    )

    kernel_h, kernel_w = kernel.shape[0], kernel.shape[1]
    pad_h = (kernel_h - 1) // 2
    pad_h_after = kernel_h - 1 - pad_h
    pad_w = (kernel_w - 1) // 2
    pad_w_after = kernel_w - 1 - pad_w

    # Pad the spatial dimensions
    images = mx.pad(
        images,
        [(0, 0), (pad_h, pad_h_after), (pad_w, pad_w_after), (0, 0)],
    )

    # Use depthwise conv2d: weight shape (C_out, KH, KW, C_in/groups)
    # For depthwise: groups=C, C_out=C, C_in/groups=1
    # kernel shape: (KH, KW) -> (C, KH, KW, 1)
    weight = mx.broadcast_to(
        kernel[None, :, :, None], (num_channels, kernel_h, kernel_w, 1)
    )
    blurred = mx.conv2d(images, weight, groups=num_channels)

    if data_format == "channels_first":
        blurred = mx.transpose(blurred, (0, 3, 1, 2))
    if need_squeeze:
        blurred = mx.squeeze(blurred, axis=0)
    return blurred.astype(input_dtype)


# ---------------------------------------------------------------------------
# Elastic transform
# ---------------------------------------------------------------------------


def elastic_transform(
    images,
    alpha=20.0,
    sigma=5.0,
    interpolation="bilinear",
    fill_mode="reflect",
    fill_value=0.0,
    seed=None,
    data_format=None,
):
    data_format = backend.standardize_data_format(data_format)
    if interpolation not in AFFINE_TRANSFORM_INTERPOLATIONS.keys():
        raise ValueError(
            "Invalid value for argument `interpolation`. Expected of one "
            f"{set(AFFINE_TRANSFORM_INTERPOLATIONS.keys())}. Received: "
            f"interpolation={interpolation}"
        )
    if fill_mode not in AFFINE_TRANSFORM_FILL_MODES:
        raise ValueError(
            "Invalid value for argument `fill_mode`. Expected of one "
            f"{AFFINE_TRANSFORM_FILL_MODES}. Received: "
            f"fill_mode={fill_mode}"
        )
    if len(images.shape) not in (3, 4):
        raise ValueError(
            "Invalid images rank: expected rank 3 (single image) "
            "or rank 4 (batch of images). Received input with shape: "
            f"images.shape={images.shape}"
        )

    images = convert_to_tensor(images)
    input_dtype = images.dtype

    alpha_t = convert_to_tensor(alpha, dtype=backend.standardize_dtype(input_dtype))
    sigma_t = convert_to_tensor(sigma, dtype=backend.standardize_dtype(input_dtype))
    sigma_val = float(sigma)

    kernel_size = (int(6 * sigma_val) | 1, int(6 * sigma_val) | 1)

    need_squeeze = False
    if len(images.shape) == 3:
        images = mx.expand_dims(images, axis=0)
        need_squeeze = True

    if data_format == "channels_last":
        batch_size, height, width, channels = images.shape
        channel_axis = -1
    else:
        batch_size, channels, height, width = images.shape
        channel_axis = 1

    seed_val = draw_seed(seed)
    key = mx.random.key(int(seed_val))
    k1, k2 = mx.random.split(key)
    dx = mx.random.normal(
        shape=(batch_size, height, width), key=k1
    ).astype(mx.float32) * sigma_t
    dy = mx.random.normal(
        shape=(batch_size, height, width), key=k2
    ).astype(mx.float32) * sigma_t

    dx = gaussian_blur(
        mx.expand_dims(dx, axis=channel_axis),
        kernel_size=kernel_size,
        sigma=(sigma_val, sigma_val),
        data_format=data_format,
    )
    dy = gaussian_blur(
        mx.expand_dims(dy, axis=channel_axis),
        kernel_size=kernel_size,
        sigma=(sigma_val, sigma_val),
        data_format=data_format,
    )

    dx = mx.squeeze(dx)
    dy = mx.squeeze(dy)

    x_grid, y_grid = mx.meshgrid(
        mx.arange(width).astype(mx.float32),
        mx.arange(height).astype(mx.float32),
    )
    x = mx.expand_dims(x_grid, 0)
    y = mx.expand_dims(y_grid, 0)

    distorted_x = x + alpha_t * dx
    distorted_y = y + alpha_t * dy

    transformed_images = mx.zeros_like(images)

    if data_format == "channels_last":
        results = []
        for b in range(batch_size):
            ch_results = []
            for i in range(channels):
                mapped = map_coordinates(
                    images[b, ..., i],
                    mx.stack([distorted_y[b], distorted_x[b]], axis=0),
                    order=AFFINE_TRANSFORM_INTERPOLATIONS[interpolation],
                    fill_mode=fill_mode,
                    fill_value=fill_value,
                )
                ch_results.append(mapped)
            results.append(mx.stack(ch_results, axis=-1))
        transformed_images = mx.stack(results, axis=0)
    else:
        results = []
        for b in range(batch_size):
            ch_results = []
            for i in range(channels):
                mapped = map_coordinates(
                    images[b, i, ...],
                    mx.stack([distorted_y[b], distorted_x[b]], axis=0),
                    order=AFFINE_TRANSFORM_INTERPOLATIONS[interpolation],
                    fill_mode=fill_mode,
                    fill_value=fill_value,
                )
                ch_results.append(mapped)
            results.append(mx.stack(ch_results, axis=0))
        transformed_images = mx.stack(results, axis=0)

    if need_squeeze:
        transformed_images = mx.squeeze(transformed_images, axis=0)
    transformed_images = transformed_images.astype(input_dtype)

    return transformed_images


# ---------------------------------------------------------------------------
# Scale and translate
# ---------------------------------------------------------------------------


def scale_and_translate(
    images,
    output_shape,
    scale,
    translation,
    spatial_dims,
    method,
    antialias=True,
):
    if method not in SCALE_AND_TRANSLATE_METHODS:
        raise ValueError(
            "Invalid value for argument `method`. Expected of one "
            f"{SCALE_AND_TRANSLATE_METHODS}. Received: method={method}"
        )
    if method in ("linear", "bilinear", "trilinear", "triangle"):
        method = "linear"
    elif method in ("cubic", "bicubic", "tricubic"):
        method = "cubic"

    images = convert_to_tensor(images)
    scale_list = [float(s) for s in scale]
    translation_list = [float(t) for t in translation]
    kernel = _kernels[method]
    return _scale_and_translate(
        images,
        output_shape,
        spatial_dims,
        scale_list,
        translation_list,
        kernel,
        antialias,
    )
