import atexit
from typing import Any, Callable, Protocol


class SupportsClose(Protocol):
    def close(self):
        pass


class SupportsBackgroundTaskTeardown(SupportsClose):
    _teardown_registered: bool
    _teardown: Callable[[], None]

    def close(self):
        super().close()
        self._teardown()
        if self._teardown_registered:
            atexit.unregister(self._teardown)
            self._teardown_registered = False

    def _register_background_proc(
        self, background_proc_spawner: Callable[[], Any], teardown: Callable[[], None]
    ) -> Any:
        if not self._teardown_registered:
            atexit.register(teardown)
            self._teardown_registered = True
            self._teardown = teardown
        return background_proc_spawner()
