
# ============================================================================
# Global Plot Style (centralized)
# ============================================================================
class PlotStyle:
    """Centralized plot styling configuration."""
    
    def __init__(self,
                 figsize: tuple = (18, 6),
                 palette_colors: list = None,
                 color_original: str = "#111111",
                 marker: str = "o",
                 linestyle: str = "-",
                 linewidth: float = 2.0):
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
        """Get a color from the palette with cycling."""
        return self.palette_colors[index % len(self.palette_colors)]
    
    def get_figsize(self, height_multiplier: float = 1.0) -> tuple:
        """Get figsize, optionally with height multiplier for subplots."""
        if height_multiplier == 1.0:
            return self.figsize
        return (self.figsize[0], self.figsize[1] * height_multiplier)
    
    def get_day_names(self, short: bool = True) -> list:
        """Get day names for conditioning."""
        if short:
            return ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        return ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    def get_color_original(self) -> str:
        """Get the original color."""
        return self.color_original

    @property
    def colors(self) -> list:
        """Get all palette colors."""
        return self.palette_colors
    
    def to_dict(self) -> dict:
        """Export style as dict for easy reference."""
        return {
            'figsize': self.figsize,
            'palette_colors': self.palette_colors,
            'color_original': self.color_original,
            'marker': self.marker,
            'linestyle': self.linestyle,
            'linewidth': self.linewidth,
        }