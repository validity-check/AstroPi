import io
import warnings

import numpy as np

from astro_pi_replay.picamera.exc import PiCameraDeprecated, PiCameraValueError


def raw_resolution(resolution, splitter=False):
    """
    Round a (width, height) tuple up to the nearest multiple of 32 horizontally
    and 16 vertically (as this is what the Pi's camera module does for
    unencoded output).
    """
    width, height = resolution
    if splitter:
        fwidth = (width + 15) & ~15
    else:
        fwidth = (width + 31) & ~31
    fheight = (height + 15) & ~15
    return fwidth, fheight


def bytes_to_rgb(data, resolution):
    """
    Converts a bytes objects containing RGB/BGR data to a `numpy`_ array.
    """
    width, height = resolution
    fwidth, fheight = raw_resolution(resolution)
    # Workaround: output from the video splitter is rounded to 16x16 instead
    # of 32x16 (but only for RGB, and only when a resizer is not used)
    if len(data) != (fwidth * fheight * 3):
        fwidth, fheight = raw_resolution(resolution, splitter=True)
        if len(data) != (fwidth * fheight * 3):
            raise PiCameraValueError(
                "Incorrect buffer length for resolution %dx%d" % (width, height)
            )
    # Crop to the actual resolution
    return np.frombuffer(data, dtype=np.uint8).reshape((fheight, fwidth, 3))[
        :height, :width, :
    ]


def bytes_to_yuv(data, resolution):
    """
    Converts a bytes object containing YUV data to a `numpy`_ array.
    """
    width, height = resolution
    fwidth, fheight = raw_resolution(resolution)
    y_len = fwidth * fheight
    uv_len = (fwidth // 2) * (fheight // 2)
    if len(data) != (y_len + 2 * uv_len):
        raise PiCameraValueError(
            "Incorrect buffer length for resolution %dx%d" % (width, height)
        )
    # Separate out the Y, U, and V values from the array
    a = np.frombuffer(data, dtype=np.uint8)
    Y = a[:y_len].reshape((fheight, fwidth))
    Uq = a[y_len:-uv_len].reshape((fheight // 2, fwidth // 2))
    Vq = a[-uv_len:].reshape((fheight // 2, fwidth // 2))
    # Reshape the values into two dimensions, and double the size of the
    # U and V values (which only have quarter resolution in YUV4:2:0)
    U = np.empty_like(Y)
    V = np.empty_like(Y)
    U[0::2, 0::2] = Uq
    U[0::2, 1::2] = Uq
    U[1::2, 0::2] = Uq
    U[1::2, 1::2] = Uq
    V[0::2, 0::2] = Vq
    V[0::2, 1::2] = Vq
    V[1::2, 0::2] = Vq
    V[1::2, 1::2] = Vq
    # Stack the channels together and crop to the actual resolution
    return np.dstack((Y, U, V))[:height, :width]


# class BroadcomRawHeader(ct.Structure):
#     _fields_ = [
#         ('name',          ct.c_char * 32),
#         ('width',         ct.c_uint16),
#         ('height',        ct.c_uint16),
#         ('padding_right', ct.c_uint16),
#         ('padding_down',  ct.c_uint16),
#         ('dummy',         ct.c_uint32 * 6),
#         ('transform',     ct.c_uint16),
#         ('format',        ct.c_uint16),
#         ('bayer_order',   ct.c_uint8),
#         ('bayer_format',  ct.c_uint8),
#         ]


class PiArrayOutput(io.BytesIO):
    """
    Base class for capture arrays.

    This class extends :class:`io.BytesIO` with a `numpy`_ array which is
    intended to be filled when :meth:`~io.IOBase.flush` is called (i.e. at the
    end of capture).

    .. attribute:: array

        After :meth:`~io.IOBase.flush` is called, this attribute contains the
        frame's data as a multi-dimensional `numpy`_ array. This is typically
        organized with the dimensions ``(rows, columns, plane)``. Hence, an
        RGB image with dimensions *x* and *y* would produce an array with shape
        ``(y, x, 3)``.
    """

    def __init__(self, camera, size=None):
        super(PiArrayOutput, self).__init__()
        self.camera = camera
        self.size = size
        self.array = None

    def close(self):
        super(PiArrayOutput, self).close()
        self.array = None

    def truncate(self, size=None):
        """
        Resize the stream to the given size in bytes (or the current position
        if size is not specified). This resizing can extend or reduce the
        current file size.  The new file size is returned.

        In prior versions of picamera, truncation also changed the position of
        the stream (because prior versions of these stream classes were
        non-seekable). This functionality is now deprecated; scripts should
        use :meth:`~io.IOBase.seek` and :meth:`truncate` as one would with
        regular :class:`~io.BytesIO` instances.
        """
        if size is not None:
            warnings.warn(
                PiCameraDeprecated(
                    "This method changes the position of the stream to the "
                    "truncated length; this is deprecated functionality and "
                    "you should not rely on it (seek before or after truncate "
                    "to ensure position is consistent)"
                )
            )
        super(PiArrayOutput, self).truncate(size)
        if size is not None:
            self.seek(size)


class PiRGBArray(PiArrayOutput):
    """
    Produces a 3-dimensional RGB array from an RGB capture.

    This custom output class can be used to easily obtain a 3-dimensional numpy
    array, organized (rows, columns, colors), from an unencoded RGB capture.
    The array is accessed via the :attr:`~PiArrayOutput.array` attribute. For
    example::

        import picamera
        import picamera.array

        with picamera.PiCamera() as camera:
            with picamera.array.PiRGBArray(camera) as output:
                camera.capture(output, 'rgb')
                print('Captured %dx%d image' % (
                        output.array.shape[1], output.array.shape[0]))

    You can re-use the output to produce multiple arrays by emptying it with
    ``truncate(0)`` between captures::

        import picamera
        import picamera.array

        with picamera.PiCamera() as camera:
            with picamera.array.PiRGBArray(camera) as output:
                camera.resolution = (1280, 720)
                camera.capture(output, 'rgb')
                print('Captured %dx%d image' % (
                        output.array.shape[1], output.array.shape[0]))
                output.truncate(0)
                camera.resolution = (640, 480)
                camera.capture(output, 'rgb')
                print('Captured %dx%d image' % (
                        output.array.shape[1], output.array.shape[0]))

    If you are using the GPU resizer when capturing (with the *resize*
    parameter of the various :meth:`~PiCamera.capture` methods), specify the
    resized resolution as the optional *size* parameter when constructing the
    array output::

        import picamera
        import picamera.array

        with picamera.PiCamera() as camera:
            camera.resolution = (1280, 720)
            with picamera.array.PiRGBArray(camera, size=(640, 360)) as output:
                camera.capture(output, 'rgb', resize=(640, 360))
                print('Captured %dx%d image' % (
                        output.array.shape[1], output.array.shape[0]))
    """

    def flush(self):
        super(PiRGBArray, self).flush()
        self.array = bytes_to_rgb(self.getvalue(), self.size or self.camera.resolution)


class PiYUVArray(PiArrayOutput):
    """
    Produces 3-dimensional YUV & RGB arrays from a YUV capture.

    This custom output class can be used to easily obtain a 3-dimensional numpy
    array, organized (rows, columns, channel), from an unencoded YUV capture.
    The array is accessed via the :attr:`~PiArrayOutput.array` attribute. For
    example::

        import picamera
        import picamera.array

        with picamera.PiCamera() as camera:
            with picamera.array.PiYUVArray(camera) as output:
                camera.capture(output, 'yuv')
                print('Captured %dx%d image' % (
                        output.array.shape[1], output.array.shape[0]))

    The :attr:`rgb_array` attribute can be queried for the equivalent RGB
    array (conversion is performed using the `ITU-R BT.601`_ matrix)::

        import picamera
        import picamera.array

        with picamera.PiCamera() as camera:
            with picamera.array.PiYUVArray(camera) as output:
                camera.resolution = (1280, 720)
                camera.capture(output, 'yuv')
                print(output.array.shape)
                print(output.rgb_array.shape)

    If you are using the GPU resizer when capturing (with the *resize*
    parameter of the various :meth:`~picamera.PiCamera.capture` methods),
    specify the resized resolution as the optional *size* parameter when
    constructing the array output::

        import picamera
        import picamera.array

        with picamera.PiCamera() as camera:
            camera.resolution = (1280, 720)
            with picamera.array.PiYUVArray(camera, size=(640, 360)) as output:
                camera.capture(output, 'yuv', resize=(640, 360))
                print('Captured %dx%d image' % (
                        output.array.shape[1], output.array.shape[0]))

    .. _ITU-R BT.601: https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion
    """

    def __init__(self, camera, size=None):
        super(PiYUVArray, self).__init__(camera, size)
        self._rgb = None

    def flush(self):
        super(PiYUVArray, self).flush()
        self.array = bytes_to_yuv(self.getvalue(), self.size or self.camera.resolution)
        self._rgb = None

    @property
    def rgb_array(self):
        if self._rgb is None:
            # Apply the standard biases
            YUV = self.array.astype(float)
            YUV[:, :, 0] = YUV[:, :, 0] - 16  # Offset Y by 16
            YUV[:, :, 1:] = YUV[:, :, 1:] - 128  # Offset UV by 128
            # YUV conversion matrix from ITU-R BT.601 version (SDTV)
            #              Y       U       V
            M = np.array(
                [
                    [1.164, 0.000, 1.596],  # R
                    [1.164, -0.392, -0.813],  # G
                    [1.164, 2.017, 0.000],
                ]
            )  # B
            # Calculate the dot product with the matrix to produce RGB output,
            # clamp the results to byte range and convert to bytes
            self._rgb = YUV.dot(M.T).clip(0, 255).astype(np.uint8)
        return self._rgb


# class PiBayerArray(PiArrayOutput):
#     """
#     Produces a 3-dimensional RGB array from raw Bayer data.

#     This custom output class is intended to be used with the
#     :meth:`~picamera.PiCamera.capture` method, with the *bayer* parameter set
#     to ``True``, to include raw Bayer data in the JPEG output.  The class
#     strips out the raw data, and constructs a numpy array from it.  The
#     resulting data is accessed via the :attr:`~PiArrayOutput.array` attribute::

#         import picamera
#         import picamera.array

#         with picamera.PiCamera() as camera:
#             with picamera.array.PiBayerArray(camera) as output:
#                 camera.capture(output, 'jpeg', bayer=True)
#                 print(output.array.shape)

#     The *output_dims* parameter specifies whether the resulting array is
#     three-dimensional (the default, or when *output_dims* is 3), or
#     two-dimensional (when *output_dims* is 2). The three-dimensional data is
#     already separated into the three color planes, whilst the two-dimensional
#     variant is not (in which case you need to know the Bayer ordering to
#     accurately deal with the results).

#     .. note::

#         Bayer data is *usually* full resolution, so the resulting array usually
#         has the shape (1944, 2592, 3) with the V1 module, or (2464, 3280, 3)
#         with the V2 module (if two-dimensional output is requested the
#         3-layered color dimension is omitted). If the camera's
#         :attr:`~picamera.PiCamera.sensor_mode` has been forced to something
#         other than 0, then the output will be the native size for the requested
#         sensor mode.

#         This also implies that the optional *size* parameter (for specifying a
#         resizer resolution) is not available with this array class.

#     As the sensor records 10-bit values, the array uses the unsigned 16-bit
#     integer data type.

#     By default, `de-mosaicing`_ is **not** performed; if the resulting array is
#     viewed it will therefore appear dark and too green (due to the green bias
#     in the `Bayer pattern`_). A trivial weighted-average demosaicing algorithm
#     is provided in the :meth:`demosaic` method::

#         import picamera
#         import picamera.array

#         with picamera.PiCamera() as camera:
#             with picamera.array.PiBayerArray(camera) as output:
#                 camera.capture(output, 'jpeg', bayer=True)
#                 print(output.demosaic().shape)

#     Viewing the result of the de-mosaiced data will look more normal but still
#     considerably worse quality than the regular camera output (as none of the
#     other usual post-processing steps like auto-exposure, white-balance,
#     vignette compensation, and smoothing have been performed).

#     .. versionchanged:: 1.13
#         This class now supports the V2 module properly, and handles flipped
#         images, and forced sensor modes correctly.

#     .. _de-mosaicing: https://en.wikipedia.org/wiki/Demosaicing
#     .. _Bayer pattern: https://en.wikipedia.org/wiki/Bayer_filter
#     """
#     BAYER_OFFSETS = {
#         # RGGB
#         0: ((0, 0), (1, 0), (0, 1), (1, 1)),
#         # GBRG
#         1: ((1, 0), (0, 0), (1, 1), (0, 1)),
#         # BGGR
#         2: ((1, 1), (0, 1), (1, 0), (0, 0)),
#         # GRBG
#         3: ((0, 1), (1, 1), (0, 0), (1, 0)),
#         }

#     def __init__(self, camera, output_dims=3):
#         super(PiBayerArray, self).__init__(camera, size=None)
#         if not (2 <= output_dims <= 3):
#             raise PiCameraValueError('output_dims must be 2 or 3')
#         self._demo = None
#         self._header = None
#         self._output_dims = output_dims

#     @property
#     def output_dims(self):
#         return self._output_dims

#     def _to_3d(self, array):
#         array_3d = np.zeros(array.shape + (3,), dtype=array.dtype)
#         (
#             (ry, rx), (gy, gx), (Gy, Gx), (by, bx)
#             ) = PiBayerArray.BAYER_OFFSETS[self._header.bayer_order]
#         array_3d[ry::2, rx::2, 0] = array[ry::2, rx::2] # Red
#         array_3d[gy::2, gx::2, 1] = array[gy::2, gx::2] # Green
#         array_3d[Gy::2, Gx::2, 1] = array[Gy::2, Gx::2] # Green
#         array_3d[by::2, bx::2, 2] = array[by::2, bx::2] # Blue
#         return array_3d

#     def flush(self):
#         super(PiBayerArray, self).flush()
#         self._demo = None
#         offset = {
#             'OV5647': {
#                 0: 6404096,
#                 1: 2717696,
#                 2: 6404096,
#                 3: 6404096,
#                 4: 1625600,
#                 5: 1233920,
#                 6: 445440,
#                 7: 445440,
#                 },
#             'IMX219': {
#                 0: 10270208,
#                 1: 2678784,
#                 2: 10270208,
#                 3: 10270208,
#                 4: 2628608,
#                 5: 1963008,
#                 6: 1233920,
#                 7: 445440,
#                 },
#             }[self.camera.revision.upper()][self.camera.sensor_mode]
#         data = self.getvalue()[-offset:]
#         if data[:4] != b'BRCM':
#             raise PiCameraValueError('Unable to locate Bayer data at end of buffer')
#         # Extract header (with bayer order and other interesting bits), which
#         # is 176 bytes from start of bayer data, and pixel data which 32768
#         # bytes from start of bayer data
#         self._header = BroadcomRawHeader.from_buffer_copy(
#             data[176:176 + ct.sizeof(BroadcomRawHeader)])
#         data = np.frombuffer(data, dtype=np.uint8, offset=32768)
#         # Reshape and crop the data. The crop's width is multiplied by 5/4 to
#         # deal with the packed 10-bit format; the shape's width is calculated
#         # in a similar fashion but with padding included (which involves
#         # several additional padding steps)
#         crop = mo.PiResolution(
#             self._header.width * 5 // 4,
#             self._header.height)
#         shape = mo.PiResolution(
#             (((self._header.width + self._header.padding_right) * 5) + 3) // 4,
#             (self._header.height + self._header.padding_down)
#             ).pad()
#         data = data.reshape((shape.height, shape.width))[:crop.height, :crop.width]
#         # Unpack 10-bit values; every 5 bytes contains the high 8-bits of 4
#         # values followed by the low 2-bits of 4 values packed into the fifth
#         # byte
#         data = data.astype(np.uint16) << 2
#         for byte in range(4):
#             data[:, byte::5] |= ((data[:, 4::5] >> (byte * 2)) & 3)
#         self.array = np.zeros(
#             (data.shape[0], data.shape[1] * 4 // 5), dtype=np.uint16)
#         for i in range(4):
#             self.array[:, i::4] = data[:, i::5]
#         if self.output_dims == 3:
#             self.array = self._to_3d(self.array)

#     def demosaic(self):
#         """
#         Perform a rudimentary `de-mosaic`_ of ``self.array``, returning the
#         result as a new array. The result of the demosaic is *always* three
#         dimensional, with the last dimension being the color planes (see
#         *output_dims* parameter on the constructor).

#         .. _de-mosaic: https://en.wikipedia.org/wiki/Demosaicing
#         """
#         if self._demo is None:
#             # Construct 3D representation of Bayer data (if necessary)
#             if self.output_dims == 2:
#                 array_3d = self._to_3d(self.array)
#             else:
#                 array_3d = self.array
#             # Construct representation of the bayer pattern
#             bayer = np.zeros(array_3d.shape, dtype=np.uint8)
#             (
#                 (ry, rx), (gy, gx), (Gy, Gx), (by, bx)
#                 ) = PiBayerArray.BAYER_OFFSETS[self._header.bayer_order]
#             bayer[ry::2, rx::2, 0] = 1 # Red
#             bayer[gy::2, gx::2, 1] = 1 # Green
#             bayer[Gy::2, Gx::2, 1] = 1 # Green
#             bayer[by::2, bx::2, 2] = 1 # Blue
#             # Allocate output array with same shape as data and set up some
#             # constants to represent the weighted average window
#             window = (3, 3)
#             borders = (window[0] - 1, window[1] - 1)
#             border = (borders[0] // 2, borders[1] // 2)
#             # Pad out the data and the bayer pattern (np.pad is faster but
#             # unavailable on the version of numpy shipped with Raspbian at the
#             # time of writing)
#             rgb = np.zeros((
#                 array_3d.shape[0] + borders[0],
#                 array_3d.shape[1] + borders[1],
#                 array_3d.shape[2]), dtype=array_3d.dtype)
#             rgb[
#                 border[0]:rgb.shape[0] - border[0],
#                 border[1]:rgb.shape[1] - border[1],
#                 :] = array_3d
#             bayer_pad = np.zeros((
#                 array_3d.shape[0] + borders[0],
#                 array_3d.shape[1] + borders[1],
#                 array_3d.shape[2]), dtype=bayer.dtype)
#             bayer_pad[
#                 border[0]:bayer_pad.shape[0] - border[0],
#                 border[1]:bayer_pad.shape[1] - border[1],
#                 :] = bayer
#             bayer = bayer_pad
#             # For each plane in the RGB data, construct a view over the plane
#             # of 3x3 matrices. Then do the same for the bayer array and use
#             # Einstein summation to get the weighted average
#             self._demo = np.empty(array_3d.shape, dtype=array_3d.dtype)
#             for plane in range(3):
#                 p = rgb[..., plane]
#                 b = bayer[..., plane]
#                 pview = as_strided(p, shape=(
#                     p.shape[0] - borders[0],
#                     p.shape[1] - borders[1]) + window, strides=p.strides * 2)
#                 bview = as_strided(b, shape=(
#                     b.shape[0] - borders[0],
#                     b.shape[1] - borders[1]) + window, strides=b.strides * 2)
#                 psum = np.einsum('ijkl->ij', pview)
#                 bsum = np.einsum('ijkl->ij', bview)
#                 self._demo[..., plane] = psum // bsum
#         return self._demo


# class PiMotionArray(PiArrayOutput):

#     def __init__(self,
#                  camera,
#                  size=None):
#         super.__init__(camera,size)

#     def flush(self):
#         pass

# class PiAnalysisOutput(io.IOBase):
#     pass
# class PiRGBAnalysis(PiAnalysisOutput):
#     pass
# class PiYUVAnalysis(PiAnalysisOutput):
#     pass
# class PiMotionAnalysis(PiAnalysisOutput):
#     pass
# class PiArrayTransform(mo.MMALPythonComponent):
#     pass
