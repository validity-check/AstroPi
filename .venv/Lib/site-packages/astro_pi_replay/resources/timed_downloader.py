from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, TypeVar

from astro_pi_replay.resources.downloader import Downloader

T = TypeVar("T")


class TimedDownloader(Downloader):
    """
    Downloader with timed fetch* methods, in order to keep
    track of how much time is spent waiting for the network.
    """

    def __init__(self, update_network_time: Callable[[float], None]) -> None:
        """
        update_network_time: function that receives the latest
            network time to add to the running total.
        """
        super().__init__()
        self.update_network_time: Callable[[float], None] = update_network_time

    def _monitor(self, func: Callable[[], T]) -> T:
        start: datetime = datetime.now()
        path: T = func()
        network_time: float = (datetime.now() - start).total_seconds()
        self.update_network_time(network_time)
        return path

    def fetch_metadata(
        self, sequence_id: str, destination: Optional[Path] = None
    ) -> Path:
        fetch = super().fetch_metadata
        return self._monitor(lambda: fetch(sequence_id, destination))

    def fetch_sequence_file(self, file_path: Path) -> Path:
        fetch = super().fetch_sequence_file
        return self._monitor(lambda: fetch(file_path))
