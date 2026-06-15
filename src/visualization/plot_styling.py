
"""Shared plotting style primitives used across visualization modules."""

# ============================================================================
# Global Plot Style (centralized)
# ============================================================================
class PlotStyle:
    """Container for consistent plotting defaults across the project.

    The class centralizes color palettes, line defaults, and figure sizing so
    all visualization modules can follow the same visual language.
    """
    
    def __init__(self,
                 figsize: tuple = (18, 6),
                 palette_colors: list = None,
                 color_original: str = "#111111",
                 marker: str = "o",
                 linestyle: str = "-",
                 linewidth: float = 2.0):
        """Initialize plotting defaults.

        Args:
            figsize: Default matplotlib figure size as ``(width, height)``.
            palette_colors: Optional list of hex colors for synthetic/sample series.
            color_original: Color used for original/ground-truth series.
            marker: Default marker style for plotted points.
            linestyle: Default line style for plotted series.
            linewidth: Default line width.
        """
        self.figsize = figsize
        self.palette_colors = palette_colors or [
            "#5f88b0",  # soft blue
            "#c6ad8f",  # soft beige
            "#7ea06b",  # soft green
            "#8fb7cf",  # light blue
            "#d9a066",  # light orange
        ]
        self.color_original = color_original
        self.marker = marker
        self.linestyle = linestyle
        self.linewidth = linewidth
     
    def get_color(self, index: int) -> str:
        """Return a palette color using cyclic indexing.

        Args:
            index: Palette index (can exceed palette length).

        Returns:
            Hex color string from ``palette_colors``.
        """
        return self.palette_colors[index % len(self.palette_colors)]
    
    def get_figsize(self, height_multiplier: float = 1.0) -> tuple:
        """Return figure size, optionally scaled in height.

        Args:
            height_multiplier: Factor multiplied onto the base height.

        Returns:
            Tuple ``(width, adjusted_height)``.
        """
        if height_multiplier == 1.0:
            return self.figsize
        return (self.figsize[0], self.figsize[1] * height_multiplier)
    
    def get_day_names(self, short: bool = True) -> list:
        """Return weekday names used in conditioning/labels.

        Args:
            short: When ``True``, returns abbreviated names.

        Returns:
            List of weekday names.
        """
        if short:
            return ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        return ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    def get_color_original(self) -> str:
        """Return the configured color for original series.

        Returns:
            Hex color string for original data.
        """
        return self.color_original

    @property
    def colors(self) -> list:
        """Expose the configured synthetic/sample palette.

        Returns:
            List of hex color strings.
        """
        return self.palette_colors
    
    def to_dict(self) -> dict:
        """Export style parameters as a plain dictionary.

        Returns:
            Dictionary containing all style attributes.
        """
        return {
            'figsize': self.figsize,
            'palette_colors': self.palette_colors,
            'color_original': self.color_original,
            'marker': self.marker,
            'linestyle': self.linestyle,
            'linewidth': self.linewidth,
        }