import logging
from threading import Event, Lock
from typing import Optional

import astro_pi_replay.picamera.mmalobj as mo
from astro_pi_replay.picamera.abstract_camera import PiCamera
from astro_pi_replay.picamera.frames import PiVideoFrame, PiVideoFrameType
from astro_pi_replay.picamera.mmalobj import MMALVideoPort

logger = logging.getLogger(__name__)


class PiEncoder:
    def __init__(
        self,
        parent: PiCamera,
        camera_port: MMALVideoPort,
        input_port: MMALVideoPort,
        format: str,
        resize: Optional[tuple[int, int]],
        **options,
    ) -> None:
        self.parent: PiCamera = parent
        self.camera_port: MMALVideoPort = camera_port
        self.input_port: MMALVideoPort = input_port
        self.format: str = format
        self.resize: Optional[tuple[int, int]] = resize
        self.output_port: MMALVideoPort = MMALVideoPort(1)
        self.outputs_lock: Lock = Lock()
        self.outputs: dict = {}  # TODO type
        self.event: Event = Event()

    @property
    def active(self):
        """
        Returns ``True`` if the MMAL encoder exists and is enabled.
        """
        try:
            return bool(self.output_port.enabled)
        except AttributeError:
            # output_port can be None; avoid a (demonstrated) race condition
            # by catching AttributeError
            return False

    def close(self):
        """
        Finalizes the encoder and deallocates all structures.

        This method is called by the camera prior to destroying the encoder (or
        more precisely, letting it go out of scope to permit the garbage
        collector to destroy it at some future time). The method destroys all
        components that the various create methods constructed and resets their
        attributes.
        """
        self.stop()
        # if self.encoder:
        #     self.encoder.disconnect()
        # if self.resizer:
        #     self.resizer.disconnect()
        # if self.encoder:
        #     self.encoder.close()
        #     self.encoder = None
        # if self.resizer:
        #     self.resizer.close()
        #     self.resizer = None
        # self.output_port = None

    def start(self, output):
        self._open_output(output)

    def stop(self):
        """
        Stops the encoder, regardless of whether it's finished.

        This method is called by the camera to terminate the execution of the
        encoder. Typically, this is used with video to stop the recording, but
        can potentially be called in the middle of image capture to terminate
        the capture.
        """
        # NOTE: The active test below is necessary to prevent attempting to
        # re-enter the parent lock in the case the encoder is being torn down
        # by an error in the constructor
        if self.active:
            if self.parent and self.camera_port:
                with self.parent._encoders_lock:
                    self.parent._stop_capture(self.camera_port)
            self.output_port.disable()
        self.event.set()
        self._close_output()

    def _close_output(self, key=PiVideoFrameType.frame):
        """
        _close_output(key=PiVideoFrameType.frame)

        Closes the output associated with *key* in :attr:`outputs`.

        Closes the output object associated with the specified *key*, and
        removes it from the :attr:`outputs` dictionary (if we didn't open the
        object then we attempt to flush it instead).
        """
        with self.outputs_lock:
            try:
                (output, opened) = self.outputs.pop(key)
            except KeyError:
                pass
            else:
                mo.close_stream(output, opened)

    def _open_output(self, output, key=PiVideoFrameType.frame):
        """
        _open_output(output, key=PiVideoFrameType.frame)

        Opens *output* and associates it with *key* in :attr:`outputs`.

        If *output* is a string, this method opens it as a filename and keeps
        track of the fact that the encoder was the one to open it (which
        implies that :meth:`_close_output` should eventually close it).
        Otherwise, if *output* has a ``write`` method it is assumed to be a
        file-like object and it is used verbatim. If *output* is neither a
        string, nor an object with a ``write`` method it is assumed to be a
        writeable object supporting the buffer protocol (this is wrapped in
        a :class:`BufferIO` stream to simplify writing).

        The opened output is added to the :attr:`outputs` dictionary with the
        specified *key*.
        """
        with self.outputs_lock:
            self.outputs[key] = mo.open_stream(output)


#     def _callback(self, port: MMALPort, buf: MMALBuffer):
#         pass

#     def _callback_write(self, buf, key=PiVideoFrameType.frame):
#         pass

#     def _close_output(self, key=PiVideoFrameType.frame):
#         pass

#     def _create_encoder(self, format):
#         pass

#     def _create_resizer(self, width, height):
#         pass

#     def _open_output(self, output, key=PiVideoFrameType.frame):
#         pass

#     def close(self) -> None:
#         pass

#     @property
#     def encoder(self) -> Optional[MMALComponent]:
#         pass

#     @property
#     def exception(self) -> Optional[Exception]:
#         pass

#     @property
#     def event(self) -> Event:
#         return self._event

#     @property
#     def outputs(self) -> dict[str, tuple[BinaryIO, bool]]:
#         return {}

#     @property
#     def outputs_lock(self):
#         return self._outputs_lock

#     @property
#     def pool(self) -> None:
#         # TODO this hsould be a pointer...
#         pass

#     @property
#     def resizer(self) -> Optional[MMALResizer]:
#         pass

#     def start(self, output) -> None:
#         pass

#     def stop(self) -> None:
#         pass

#     def wait(self, timeout: Optional[int] = None) -> None:
#         pass


class PiVideoEncoder(PiEncoder):
    def __init__(
        self,
        parent: PiCamera,
        camera_port: MMALVideoPort,
        input_port: MMALVideoPort,
        format: str,
        resize: Optional[tuple[int, int]],
        **options,
    ) -> None:
        super().__init__(parent, camera_port, input_port, format, resize, **options)
        # Fake frame
        self.frame: Optional[PiVideoFrame] = PiVideoFrame(
            0, PiVideoFrameType.frame, 0, 0, 0, 0, False
        )


# class PiImageEncoder(PiEncoder):
#
#    @property
#    def encoder_type(self):
#        # alias of picamera.mmalobj.MMALImageEncoder
#        pass
#
#    def _create_encoder(self,
#                        format: str,
#                        quality: int=85,
#                        thumbnail: tuple[int,int,int]=(64,48,35),
#                        restart: int=0):
#        pass
