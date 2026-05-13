from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtCore import Qt

class LossWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.loss = []

    def set_data(self, loss):
        self.loss = loss
        self.update()

    def paintEvent(self, event):
        if not self.loss:
            return

        painter = QPainter(self)
        pen = QPen(Qt.blue, 2)
        painter.setPen(pen)

        w, h = self.width(), self.height()

        max_loss = max(self.loss)
        min_loss = min(self.loss)

        def norm(x):
            return (x - min_loss) / (max_loss - min_loss + 1e-8)

        points = []
        for i, v in enumerate(self.loss):
            x = int(i / len(self.loss) * w)
            y = int(h - norm(v) * h)
            points.append((x, y))

        for i in range(len(points) - 1):
            painter.drawLine(*points[i], *points[i+1])