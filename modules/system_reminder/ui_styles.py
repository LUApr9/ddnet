from PyQt5.QtGui import QColor

def darken_color(hex_color: str, factor: float) -> str:
    c = QColor(hex_color)
    r = max(0, int(c.red() * factor))
    g = max(0, int(c.green() * factor))
    b = max(0, int(c.blue() * factor))
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

def button_style(color: str = "#1E90FF", radius: int = 8, text_color: str = "white") -> str:
    dark = darken_color(color, 0.85)
    return (
        f"""
        QPushButton {{
            background-color: {color};
            color: {text_color};
            border-radius: {radius}px;
            padding: 8px 28px;
            border: none;
        }}
        QPushButton:pressed {{ background-color: {dark}; }}
        QPushButton:disabled {{
            background-color: #e5ebf2;
            color: #9aa8b7;
            border: 1px solid #d2dbe5;
        }}
        """
    )
