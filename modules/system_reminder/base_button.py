from PyQt5.QtWidgets import QPushButton
from .ui_styles import button_style

class BaseButton(QPushButton):
    def __init__(self, text, parent=None, color="#1E90FF", radius=8):
        super().__init__(text, parent)
        self.setStyleSheet(button_style(color=color, radius=radius))
