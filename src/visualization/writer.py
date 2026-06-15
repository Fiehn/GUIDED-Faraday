import os
from matplotlib import pyplot as plt
 
class Writer:
    """Lightweight logger that persists scalars, strings, and images to disk.
    
    Only pending (not-yet-dumped) data lives in memory.
    Getters read from disk + merge pending, so memory stays bounded.
    """

    def __init__(self, log_dir: str, auto_dump_threshold: int = 10000):
        """Initialize writer and in-memory pending buffers.

        Args:
            log_dir: Base directory where logs are stored. A `logs/` subfolder is
                created and used as the final write target.
            auto_dump_threshold: Number of pending entries across all buffers that
                triggers automatic flush to disk. Set `<= 0` to disable auto-dump.
        """
        self.log_dir = log_dir
        self.auto_dump_threshold = auto_dump_threshold
        os.makedirs(log_dir, exist_ok=True)
        # Move log_dir to a folder named "logs" inside original log_dir if it doesn't already exist
        if not os.path.isdir(os.path.join(log_dir, "logs")):
            os.makedirs(os.path.join(log_dir, "logs"), exist_ok=True)
            self.log_dir = os.path.join(log_dir, "logs")

        # Only pending data lives in memory
        self._pending_scalar: dict[str, list[tuple[int, float]]] = {}
        self._pending_string: dict[str, list[tuple[int, str]]] = {}
        self._pending_image: dict[str, list[tuple[int, plt.Figure]]] = {}

    # ------------------------------------------------------------------
    # Public getters (read disk + merge pending)
    # ------------------------------------------------------------------
    def get_scalars(self, name: str) -> list[tuple[int, float]]:
        """Read scalar series for a tag.

        Combines persisted entries from disk with currently pending in-memory
        entries so callers always get the latest view.

        Args:
            name: Scalar tag name.

        Returns:
            List of `(step, value)` scalar entries.
        """
        safe = self._safe_filename(name)
        path = os.path.join(self.log_dir, f"{safe}.csv")
        entries = []
        if os.path.isfile(path):
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(",", 1)
                    entries.append((int(parts[0]), float(parts[1])))
        entries.extend(self._pending_scalar.get(name, []))
        return entries

    def get_strings(self, name: str) -> list[tuple[int, str]]:
        """Read string series for a tag.

        Combines persisted entries from disk with currently pending in-memory
        entries so callers always get the latest view.

        Args:
            name: String tag name.

        Returns:
            List of `(step, text)` entries.
        """
        safe = self._safe_filename(name)
        path = os.path.join(self.log_dir, f"{safe}.txt")
        entries = []
        if os.path.isfile(path):
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(":", 1)
                    entries.append((int(parts[0]), parts[1].strip()))
        entries.extend(self._pending_string.get(name, []))
        return entries

    def get_images(self, name: str) -> list[tuple[int, plt.Figure]]:
        """Return pending image figures for a tag.

        Note:
            Saved PNGs are not reloaded into `Figure` objects, so this method only
            returns images still present in pending buffers.

        Args:
            name: Image tag name.

        Returns:
            List of `(step, figure)` tuples currently in memory.
        """
        return list(self._pending_image.get(name, []))

    def list_tags(self) -> dict[str, list[str]]:
        """List known scalar and string tags.

        Tags are discovered from both pending buffers and on-disk files.

        Returns:
            Dictionary with keys `scalars` and `strings`, each containing sorted
            tag names.
        """
        scalar_tags = set(self._pending_scalar.keys())
        string_tags = set(self._pending_string.keys())
        if os.path.isdir(self.log_dir):
            for f in os.listdir(self.log_dir):
                if f.endswith(".csv"):
                    scalar_tags.add(f[:-4])
                elif f.endswith(".txt"):
                    string_tags.add(f[:-4])
        return {"scalars": sorted(scalar_tags), "strings": sorted(string_tags)}

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------
    def _pending_count(self) -> int:
        """Count total pending entries across scalar/string/image buffers.

        Returns:
            Total number of buffered entries.
        """
        return (sum(len(v) for v in self._pending_scalar.values())
                + sum(len(v) for v in self._pending_string.values())
                + sum(len(v) for v in self._pending_image.values()))

    def _auto_dump(self):
        """Flush pending buffers to disk when threshold is reached."""
        if self.auto_dump_threshold > 0 and self._pending_count() >= self.auto_dump_threshold:
            self._dump()

    @staticmethod
    def _safe_filename(name: str) -> str:
        """Convert tag names like 'Train/loss_step' to safe filenames."""
        return name.replace("/", "_").replace("\\", "_")

    # ------------------------------------------------------------------
    # Add new data (goes into pending buffers only)
    # ------------------------------------------------------------------
    def add_scalar(self, name: str, value: float, step: int):
        """Append a scalar entry to pending buffer.

        Args:
            name: Scalar tag name.
            value: Scalar value.
            step: Training/evaluation step.
        """
        self._pending_scalar.setdefault(name, []).append((step, value))
        self._auto_dump()

    def add_string(self, name: str, value: str, step: int):
        """Append a string entry to pending buffer.

        Args:
            name: String tag name.
            value: Text value.
            step: Training/evaluation step.
        """
        self._pending_string.setdefault(name, []).append((step, value))
        self._auto_dump()

    def add_image(self, name: str, image: plt.Figure, step: int):
        """Append an image figure to pending buffer.

        Args:
            name: Image tag name.
            image: Matplotlib figure object.
            step: Training/evaluation step.
        """
        self._pending_image.setdefault(name, []).append((step, image))
        self._auto_dump()

    # TensorBoard API compatibility
    def add_text(self, name: str, value: str, step: int):
        """TensorBoard-compatible alias for `add_string`.

        Args:
            name: Text tag name.
            value: Text value.
            step: Training/evaluation step.
        """
        self.add_string(name, value, step)

    def add_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int):
        """TensorBoard-compatible helper to log multiple scalars under one prefix.

        Args:
            main_tag: Prefix for grouped scalar tags.
            tag_scalar_dict: Mapping `{sub_tag: scalar_value}`.
            step: Training/evaluation step.
        """
        for sub_tag, value in tag_scalar_dict.items():
            self.add_scalar(f"{main_tag}/{sub_tag}", value, step)

    # ------------------------------------------------------------------
    # Dump pending data to disk, then free memory
    # ------------------------------------------------------------------
    def _dump(self):
        """Persist all pending buffers to disk and clear in-memory state.

        Scalars are appended to `.csv`, strings to `.txt`, and figures are saved
        as PNG files with step suffixes.
        """
        os.makedirs(self.log_dir, exist_ok=True)

        for name, scalars in self._pending_scalar.items():
            safe = self._safe_filename(name)
            with open(os.path.join(self.log_dir, f"{safe}.csv"), "a") as f:
                for step, value in scalars:
                    f.write(f"{step},{value}\n")

        for name, strings in self._pending_string.items():
            safe = self._safe_filename(name)
            with open(os.path.join(self.log_dir, f"{safe}.txt"), "a") as f:
                for step, value in strings:
                    f.write(f"{step}: {value}\n")

        for name, images in self._pending_image.items():
            safe = self._safe_filename(name)
            for step, image in images:
                image.savefig(os.path.join(self.log_dir, f"{safe}_step{step}.png"))
                plt.close(image)

        self._pending_scalar.clear()
        self._pending_string.clear()
        self._pending_image.clear()

    def close(self):
        """Flush pending data to disk and finalize writer state."""
        self._dump()