import os
from matplotlib import pyplot as plt

class Writer:
    """Lightweight logger that persists scalars, strings, and images to disk.
    
    Only pending (not-yet-dumped) data lives in memory.
    Getters read from disk + merge pending, so memory stays bounded.
    """

    def __init__(self, log_dir: str, auto_dump_threshold: int = 10000):
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
        """Read persisted scalars from disk and merge with pending."""
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
        """Read persisted strings from disk and merge with pending."""
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
        """Images only available from pending (figures can't be reloaded)."""
        return list(self._pending_image.get(name, []))

    def list_tags(self) -> dict[str, list[str]]:
        """List all available tag names by type."""
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
        return (sum(len(v) for v in self._pending_scalar.values())
                + sum(len(v) for v in self._pending_string.values())
                + sum(len(v) for v in self._pending_image.values()))

    def _auto_dump(self):
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
        self._pending_scalar.setdefault(name, []).append((step, value))
        self._auto_dump()

    def add_string(self, name: str, value: str, step: int):
        self._pending_string.setdefault(name, []).append((step, value))
        self._auto_dump()

    def add_image(self, name: str, image: plt.Figure, step: int):
        self._pending_image.setdefault(name, []).append((step, image))
        self._auto_dump()

    # TensorBoard API compatibility
    def add_text(self, name: str, value: str, step: int):
        self.add_string(name, value, step)

    def add_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int):
        for sub_tag, value in tag_scalar_dict.items():
            self.add_scalar(f"{main_tag}/{sub_tag}", value, step)

    # ------------------------------------------------------------------
    # Dump pending data to disk, then free memory
    # ------------------------------------------------------------------
    def _dump(self):
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
        self._dump()