from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QHBoxLayout,
    QComboBox,
    QGridLayout,
    QDialog,
    QFileDialog,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QPlainTextEdit,
    QApplication,
    QMessageBox,
    QLineEdit,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap, QPainter, QPen, QColor, QFontMetrics
import os
import csv
import glob
import json
import subprocess
import sys
import hashlib
import shutil
from datetime import datetime
import numpy as np
import cv2
import torch
import matplotlib.cm as mpl_cm
from func.datasets_reader import single_read_matfile, single_read_npyfile
from model_train import determine_network
from param_config import dataset_name as DEFAULT_DATASET_NAME
from .base_button import BaseButton


PANEL_STYLE = (
    "QWidget {"
    "background:#ffffff;"
    "border:1px solid #d9e1ec;"
    "border-radius:12px;"
    "}"
)

COMBO_STYLE = (
    "QComboBox {"
    "background:#f8fbff;"
    "border:1px solid #bfd1e3;"
    "border-radius:8px;"
    "padding:4px 8px;"
    "min-height:28px;"
    "font-size:14px;"
    "color:#223142;"
    "}"
    "QComboBox:hover {border:1px solid #8aaed1;}"
    "QComboBox:focus {border:1px solid #2c7fb8;}"
    "QComboBox::drop-down {border:none; width:24px;}"
    "QComboBox QAbstractItemView {"
    "border:1px solid #bfd1e3;"
    "font-size:14px;"
    "selection-background-color:#dbeeff;"
    "selection-color:#1f2d3d;"
    "}"
    "QComboBox:disabled {"
    "background:#eef2f6;"
    "border:1px solid #d2dbe5;"
    "color:#8a98a8;"
    "}"
)


def _apply_combo_style(combo):
    combo.setStyleSheet(COMBO_STYLE)
    combo.setMinimumWidth(150)
    combo.setMaximumWidth(175)


class LossPanel(QWidget):
    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.train_loss = []
        self.val_loss = None
        self.third_loss = None
        self.smooth_window = None
        self.epochs = None
        self.x_title = "迭代次数"
        self.y_title = "损失"
        self.title = "Training Loss"
        self.train_label = "Train"
        self.val_label = "Val"
        self.third_label = "Model3"
        self.error_text = ""

    # ===== 外部调用接口 =====
    def set_data(self, train_loss, val_loss=None, third_loss=None, smooth=None, epochs=None,
                 x_title="迭代次数", y_title="损失", title="Training Loss",
                 train_label="Train", val_label="Val", third_label="Model3"):
        self.train_loss = np.array(train_loss)
        self.val_loss = np.array(val_loss) if val_loss is not None else None
        self.third_loss = np.array(third_loss) if third_loss is not None else None
        self.smooth_window = smooth
        self.epochs = epochs
        self.x_title = x_title
        self.y_title = y_title
        self.title = title
        self.train_label = train_label
        self.val_label = val_label
        self.third_label = third_label
        self.error_text = ""
        self.update()

    def set_error(self, text):
        self.train_loss = []
        self.val_loss = None
        self.third_loss = None
        self.error_text = text
        self.update()

    # ===== 平滑函数 =====
    def smooth(self, data, window):
        if window is None or window <= 1:
            return data
        return np.convolve(data, np.ones(window)/window, mode='same')

    # ===== 核心绘图 =====
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = max(self.width(), 1)
        h = max(self.height(), 1)
        base = min(w, h)
        scale = max(0.65, min(2.4, base / 700.0))

        title_size = max(10, int(13 * scale))
        axis_size = max(8, int(11 * scale))
        tick_size = max(7, int(9 * scale))
        error_size = max(9, int(11 * scale))

        if len(self.train_loss) < 2:
            painter.fillRect(self.rect(), QColor(250, 252, 255))
            painter.setPen(QColor(185, 74, 72))
            font = QFont()
            font.setPointSize(error_size)
            painter.setFont(font)
            msg = self.error_text if self.error_text else "暂无可显示数据"
            painter.drawText(self.rect(), Qt.AlignCenter, msg)
            return

        title_h_for_layout = max(24, int(30 * scale))
        legend_h_for_layout = max(12, int(16 * scale))
        title_to_legend_gap = max(10, int(12 * scale))
        legend_to_plot_gap = max(8, int(10 * scale))

        left_margin = max(56, int(88 * scale))
        right_margin = max(18, int(28 * scale))
        top_margin = max(36, int(52 * scale), max(8, int(10 * scale)) + title_h_for_layout + title_to_legend_gap + legend_h_for_layout + legend_to_plot_gap)
        bottom_margin = max(56, int(88 * scale))
        left_margin = min(left_margin, int(w * 0.30))
        right_margin = min(right_margin, int(w * 0.20))
        top_margin = min(top_margin, int(h * 0.25))
        bottom_margin = min(bottom_margin, int(h * 0.32))
        plot_w = w - left_margin - right_margin
        plot_h = h - top_margin - bottom_margin
        if plot_w <= 40 or plot_h <= 40:
            return

        # ===== 数据处理 =====
        train = self.smooth(self.train_loss, self.smooth_window)
        data_all = train

        if self.val_loss is not None:
            val = self.smooth(self.val_loss, self.smooth_window)
            data_all = np.concatenate([train, val])
        else:
            val = None

        if self.third_loss is not None:
            third = self.smooth(self.third_loss, self.smooth_window)
            data_all = np.concatenate([data_all, third])
        else:
            third = None

        min_v = float(np.min(data_all))
        max_v = float(np.max(data_all))
        eps = 1e-8

        def norm_y(v):
            return (v - min_v) / (max_v - min_v + eps)

        # ===== 背景 =====
        painter.fillRect(self.rect(), Qt.white)

        # ===== 网格 =====
        grid_pen = QPen(QColor(222, 228, 236), 1, Qt.DashLine)
        painter.setPen(grid_pen)

        grid_n = 5
        for i in range(grid_n + 1):
            y = top_margin + int(i / grid_n * plot_h)
            painter.drawLine(left_margin, y, w - right_margin, y)

        for i in range(grid_n + 1):
            x = left_margin + int(i / grid_n * plot_w)
            painter.drawLine(x, top_margin, x, h - bottom_margin)

        # ===== 坐标轴 =====
        axis_pen = QPen(QColor(60, 60, 60), max(1, int(2 * scale)))
        painter.setPen(axis_pen)

        # x轴
        painter.drawLine(left_margin, h - bottom_margin, w - right_margin, h - bottom_margin)
        # y轴
        painter.drawLine(left_margin, top_margin, left_margin, h - bottom_margin)

        # ===== 画曲线函数 =====
        def draw_curve(data, color):
            pen = QPen(color, max(1, int(2 * scale)))
            painter.setPen(pen)

            n = len(data)
            points = []

            for i, v in enumerate(data):
                x = left_margin + int(i / (n - 1) * plot_w)
                y = top_margin + int((1 - norm_y(v)) * plot_h)
                points.append((x, y))

            for i in range(len(points) - 1):
                painter.drawLine(points[i][0], points[i][1],
                                 points[i+1][0], points[i+1][1])

            step = max(n // 12, 1)
            dot = max(2, int(4 * scale))
            for i in range(0, n, step):
                painter.drawEllipse(points[i][0] - dot // 2, points[i][1] - dot // 2, dot, dot)

        # ===== 画 train =====
        train_color = QColor(0, 114, 178)      # 深蓝
        val_color = QColor(230, 159, 0)        # 橙色
        third_color = QColor(204, 121, 167)    # 洋红
        draw_curve(train, train_color)

        # ===== 画 val =====
        if val is not None:
            draw_curve(val, val_color)

        # ===== 画 third =====
        if third is not None:
            draw_curve(third, third_color)

        # ===== 标题 =====
        painter.setPen(QColor(35, 35, 35))
        font = QFont()
        font.setPointSize(title_size)
        font.setBold(True)
        painter.setFont(font)

        title_y = max(8, int(10 * scale))
        title_h = max(24, int(30 * scale))
        title_bottom = title_y + title_h
        # 标题水平居中于绘图区域（left_margin 到 left_margin+plot_w）
        title_x = left_margin
        title_w = plot_w
        painter.drawText(title_x, title_y, title_w, title_h, Qt.AlignCenter, self.title)

        # ===== 图例（右上角自适应，避免文字被裁切） =====
        font.setPointSize(max(8, int(10 * scale)))
        font.setBold(False)
        painter.setFont(font)
        fm = QFontMetrics(font)

        line_len = max(12, int(20 * scale))
        text_gap = max(6, int(8 * scale))
        item_gap = max(10, int(14 * scale))
        legend_items = [(train_color, self.train_label)]
        if val is not None:
            legend_items.append((val_color, self.val_label))
        if third is not None:
            legend_items.append((third_color, self.third_label))

        total_w = 0
        for _, label in legend_items:
            total_w += line_len + text_gap + fm.horizontalAdvance(label) + item_gap
        total_w = max(0, total_w - item_gap)

        legend_x = max(left_margin + 4, w - right_margin - total_w - 4)
        legend_y = max(
            title_bottom + max(10, int(12 * scale)),
            top_margin - max(8, int(10 * scale)),
        )
        legend_y = min(legend_y, top_margin - 2)

        cursor_x = legend_x
        for color, label in legend_items:
            painter.setPen(QPen(color, max(1, int(2 * scale))))
            painter.drawLine(cursor_x, legend_y, cursor_x + line_len, legend_y)
            cursor_x += line_len + text_gap
            painter.setPen(Qt.black)
            label_w = fm.horizontalAdvance(label)
            painter.drawText(cursor_x, legend_y + max(5, int(6 * scale)), label)
            cursor_x += label_w + item_gap

        # ===== 坐标刻度 =====
        # x 轴刻度：对齐 save_results，每 20 一个刻度，范围 [0, epochs]
        tick_font = QFont()
        tick_font.setPointSize(tick_size)
        painter.setFont(tick_font)

        if self.epochs is not None and self.epochs > 0:
            tick_epochs = list(range(0, int(self.epochs) + 1, 20))
            if int(self.epochs) not in tick_epochs:
                tick_epochs.append(int(self.epochs))
            tick_epochs = sorted(set(tick_epochs))
            for epoch in tick_epochs:
                x = left_margin + int(epoch / max(int(self.epochs), 1) * plot_w)
                tick_len = max(4, int(6 * scale))
                painter.drawLine(x, h - bottom_margin, x, h - bottom_margin + tick_len)
                text_w = max(28, int(44 * scale))
                text_h = max(14, int(18 * scale))
                painter.drawText(x - text_w // 2, h - bottom_margin + tick_len + int(2 * scale), text_w, text_h, Qt.AlignCenter, str(epoch))
        else:
            ticks = 5
            for i in range(ticks + 1):
                x = left_margin + int(i / ticks * plot_w)
                epoch = int(i / ticks * (len(train) - 1))
                tick_len = max(4, int(6 * scale))
                painter.drawLine(x, h - bottom_margin, x, h - bottom_margin + tick_len)
                text_w = max(28, int(44 * scale))
                text_h = max(14, int(18 * scale))
                painter.drawText(x - text_w // 2, h - bottom_margin + tick_len + int(2 * scale), text_w, text_h, Qt.AlignCenter, str(epoch))

        ticks = 5
        for i in range(ticks + 1):
            # y轴
            y = top_margin + int(i / ticks * plot_h)
            val_text = max_v - i / ticks * (max_v - min_v)
            tick_len = max(4, int(6 * scale))
            painter.drawLine(left_margin - tick_len, y, left_margin, y)
            text_h = max(14, int(18 * scale))
            painter.drawText(max(6, int(10 * scale)), y - text_h // 2, left_margin - max(14, int(18 * scale)), text_h, Qt.AlignRight | Qt.AlignVCenter, f"{val_text:.2f}")

        # x/y 轴标题
        font.setBold(False)
        font.setPointSize(axis_size)
        painter.setFont(font)
        # 轴标题放在刻度文本之外，避免遮挡
        painter.drawText(left_margin, h - max(35, int(45 * scale)), plot_w, max(30, int(35 * scale)), Qt.AlignCenter, self.x_title)
        painter.save()
        painter.translate(max(10, int(14 * scale)), top_margin + plot_h // 2)
        painter.rotate(-90)
        painter.drawText(-plot_h // 2, 0, plot_h, max(18, int(20 * scale)), Qt.AlignCenter, self.y_title)
        painter.restore()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def export_pixmap(self):
        pix = QPixmap(self.size())
        pix.fill(Qt.transparent)
        self.render(pix)
        return pix


class SystemReminderPage(QWidget):
    # Simple home page for system reminder with titles only
    train_requested = pyqtSignal()
    compare_requested = pyqtSignal()
    metric_requested = pyqtSignal()
    observe_requested = pyqtSignal()
    back_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(16)
        self.setStyleSheet("background:qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #f5f9ff, stop:1 #eef4fb);")
        self.title_label = QLabel("融合预训练策略的地震反演系统", self)
        self.title_label.setAlignment(Qt.AlignCenter)
        font = self.title_label.font()
        font.setPointSize(22)
        font.setBold(True)
        self.title_label.setFont(font)
        self.title_label.setStyleSheet("color:#0f3352; letter-spacing:1px;")
        layout.addWidget(self.title_label)

        bottom_widget = QWidget(self)
        bottom_widget.setStyleSheet("background:#ffffff; border:1px solid #d7e1ec; border-radius:14px;")
        bottom_layout = QHBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(28, 20, 28, 20)
        bottom_layout.setSpacing(18)

        self.train_btn = BaseButton("loss曲线", self, color="#2c7fb8", radius=10)
        self.compare_btn = BaseButton("速度模型对比", self, color="#1f9d8a", radius=10)
        self.metric_btn = BaseButton("指标对比", self, color="#d9822b", radius=10)
        #self.observe_btn = BaseButton("实验数据观测", self, color="#3b78a8", radius=10)

        bottom_layout.addStretch(1)  # 左边弹性空间
        bottom_layout.addWidget(self.train_btn)
        bottom_layout.addStretch(1)  # 按钮之间弹性空间
        bottom_layout.addWidget(self.metric_btn)
        bottom_layout.addStretch(1)  # 按钮之间弹性空间
        bottom_layout.addWidget(self.compare_btn)
        bottom_layout.addStretch(1)  # 右边弹性空间
        #bottom_layout.addWidget(self.observe_btn)
        layout.addWidget(bottom_widget)
        self.setLayout(layout)

        # Signals to host (placeholders for future wiring)
        self.train_btn.clicked.connect(self.train_requested)
        self.compare_btn.clicked.connect(self.compare_requested)
        self.metric_btn.clicked.connect(self.metric_requested)
        #self.observe_btn.clicked.connect(self.observe_requested)

    def update_content(self, text: str) -> None:
        pass


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


class ObservationView(QWidget):
    back_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self._dataset_name = DEFAULT_DATASET_NAME
        self._last_manual_file = ""
        self._last_export_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "results",
        )
        self._seismic_pixmap = None
        self._velocity_pixmap = None
        self._build_ui()

    def _available_datasets(self):
        data_root = os.path.join(self._project_root, "data")
        excluded = {"SEGSalt", "SEGSimulation"}
        if not os.path.isdir(data_root):
            return [self._dataset_name]
        names = []
        for name in sorted(os.listdir(data_root)):
            full = os.path.join(data_root, name)
            if os.path.isdir(full) and not name.startswith(".") and name not in excluded:
                names.append(name)
        return names or [self._dataset_name]

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(14)

        left_widget = QWidget(self)
        left_widget.setStyleSheet(PANEL_STYLE)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(18, 16, 18, 16)
        left_layout.setSpacing(10)

        self.mode_label = QLabel("数据来源", self)
        self.mode_label.setStyleSheet("color:#1a3550; font-size:13px; font-weight:600;")
        self.mode_combo = QComboBox(self)
        self.mode_combo.addItems(["自动读取", "手动文件"])
        _apply_combo_style(self.mode_combo)

        self.dataset_label = QLabel("数据集", self)
        self.dataset_label.setStyleSheet("color:#1a3550; font-size:13px; font-weight:600;")
        self.dataset_combo = QComboBox(self)
        self.dataset_combo.addItems(self._available_datasets())
        self.dataset_combo.setCurrentText(self._dataset_name)
        if self.dataset_combo.currentText() != self._dataset_name:
            self._dataset_name = self.dataset_combo.currentText()
        _apply_combo_style(self.dataset_combo)

        self.split_label = QLabel("数据分区", self)
        self.split_label.setStyleSheet("color:#1a3550; font-size:13px; font-weight:600;")
        self.split_combo = QComboBox(self)
        self.split_combo.addItems(["test", "train"])
        _apply_combo_style(self.split_combo)

        self.sample_id_label = QLabel("样本编号", self)
        self.sample_id_label.setStyleSheet("color:#1a3550; font-size:13px; font-weight:600;")
        self.sample_id_edit = QLineEdit(self)
        self.sample_id_edit.setPlaceholderText("留空使用默认值；支持 1615 或 1,2")
        self.sample_id_edit.setStyleSheet(
            "QLineEdit {"
            "background:#f8fbff;"
            "border:1px solid #bfd1e3;"
            "border-radius:8px;"
            "padding:5px 8px;"
            "font-size:13px;"
            "color:#223142;"
            "}"
            "QLineEdit:focus {border:1px solid #2c7fb8;}"
            "QLineEdit:disabled {"
            "background:#eef2f6;"
            "border:1px solid #d2dbe5;"
            "color:#8a98a8;"
            "}"
        )

        self.file_label = QLabel("手动文件", self)
        self.file_label.setStyleSheet("color:#1a3550; font-size:13px; font-weight:600;")
        self.file_btn = BaseButton("选择 .npy/.mat", self, color="#2c7fb8", radius=8)
        self.file_hint = QLabel("未选择文件", self)
        self.file_hint.setWordWrap(True)
        self.file_hint.setStyleSheet("color:#5b6f83; font-size:11px;")

        self.run_btn = BaseButton("加载并显示", self, color="#1f9d8a", radius=10)
        self.export_btn = BaseButton("下载图像", self, color="#20B2AA", radius=10)
        self.back_btn = BaseButton("返回", self, color="#60798f", radius=10)

        left_layout.addWidget(self.mode_label)
        left_layout.addWidget(self.mode_combo)
        left_layout.addWidget(self.dataset_label)
        left_layout.addWidget(self.dataset_combo)
        left_layout.addWidget(self.split_label)
        left_layout.addWidget(self.split_combo)
        left_layout.addWidget(self.sample_id_label)
        left_layout.addWidget(self.sample_id_edit)
        left_layout.addWidget(self.file_label)
        left_layout.addWidget(self.file_btn)
        left_layout.addWidget(self.file_hint)
        left_layout.addSpacing(6)
        left_layout.addWidget(self.run_btn)
        left_layout.addWidget(self.export_btn)
        left_layout.addWidget(self.back_btn)
        left_layout.addStretch(1)
        root.addWidget(left_widget, 1)

        right_widget = QWidget(self)
        right_widget.setStyleSheet(PANEL_STYLE)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(10, 10, 10, 10)

        self.seismic_label = ClickableLabel("地震图像")
        self.seismic_label.setAlignment(Qt.AlignCenter)
        self.seismic_label.setMinimumSize(420, 220)
        self.seismic_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.seismic_label.setStyleSheet(
            "border:1px solid #c7d7e6;"
            "border-radius:10px;"
            "background:qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #f8fbff, stop:1 #edf4fc);"
            "color:#2f3b4a;"
        )
        right_layout.addWidget(self.seismic_label, 1)

        self.velocity_label = ClickableLabel("速度模型")
        self.velocity_label.setAlignment(Qt.AlignCenter)
        self.velocity_label.setMinimumSize(420, 220)
        self.velocity_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.velocity_label.setStyleSheet(
            "border:1px solid #c7d7e6;"
            "border-radius:10px;"
            "background:qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #f8fbff, stop:1 #edf4fc);"
            "color:#2f3b4a;"
        )
        right_layout.addWidget(self.velocity_label, 1)

        self.run_log = QPlainTextEdit(self)
        self.run_log.setReadOnly(True)
        self.run_log.setPlaceholderText("数据加载与渲染日志")
        self.run_log.setMinimumHeight(110)
        self.run_log.setStyleSheet(
            "QPlainTextEdit {"
            "background:#0f1a26;"
            "color:#d7e7ff;"
            "border:1px solid #24384d;"
            "border-radius:8px;"
            "padding:8px;"
            "font-family:Consolas, 'Courier New', monospace;"
            "font-size:12px;"
            "}"
        )
        right_layout.addWidget(self.run_log)
        root.addWidget(right_widget, 2)

        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        self.dataset_combo.currentTextChanged.connect(self._on_dataset_changed)
        self.file_btn.clicked.connect(self._choose_manual_file)
        self.run_btn.clicked.connect(self._load_and_render)
        self.export_btn.clicked.connect(self._export_image)
        self.back_btn.clicked.connect(self.back_requested.emit)
        self.seismic_label.clicked.connect(lambda: self._show_zoom("seismic"))
        self.velocity_label.clicked.connect(lambda: self._show_zoom("velocity"))
        self._on_mode_changed(self.mode_combo.currentText())

    def _append_log(self, text):
        if not text:
            return
        self.run_log.appendPlainText(text)
        bar = self.run_log.verticalScrollBar()
        bar.setValue(bar.maximum())
        QApplication.processEvents()

    def _short_path_with_parents(self, path, parent_count=3):
        if not path:
            return "(未找到)"
        norm = os.path.normpath(path)
        parts = norm.split(os.sep)
        need = parent_count + 1
        if len(parts) <= need:
            return norm
        return "..." + os.sep + os.sep.join(parts[-need:])

    def _dataset_cfg(self, dataset_name):
        if dataset_name in ["SEGSalt", "SEGSimulation"]:
            return {"data_dim": [400, 301], "model_dim": [201, 301], "default_select_id": 1615}
        return {"data_dim": [1000, 70], "model_dim": [70, 70], "default_select_id": [1, 2]}

    def _dataset_dir(self, dataset_name):
        return os.path.join(self._project_root, "data", dataset_name) + os.sep

    def _on_dataset_changed(self, name):
        self._dataset_name = name

    def _on_mode_changed(self, mode_text):
        auto_mode = mode_text == "自动读取"
        self.dataset_combo.setEnabled(auto_mode)
        self.split_combo.setEnabled(auto_mode)
        self.sample_id_edit.setEnabled(auto_mode)
        self.file_btn.setEnabled(not auto_mode)
        self.file_hint.setEnabled(not auto_mode)

    def _choose_manual_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择实验数据文件",
            self._project_root,
            "Data (*.npy *.mat)",
        )
        if not file_path:
            return
        self._last_manual_file = file_path
        self.file_hint.setText(self._short_path_with_parents(file_path, parent_count=4))
        self._append_log(f"已选择手动文件: {file_path}")

    def _parse_sample_id(self, cfg):
        text = self.sample_id_edit.text().strip()
        if not text:
            return cfg["default_select_id"]
        if "," in text:
            parts = [p.strip() for p in text.split(",") if p.strip()]
            if not parts:
                return cfg["default_select_id"]
            return [int(p) for p in parts]
        return int(text)

    def _extract_2d_data(self, data):
        arr = np.asarray(data)
        if arr.size == 0:
            raise ValueError("数据为空")
        arr = np.squeeze(arr)
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        while arr.ndim > 2:
            arr = arr[0]
        if arr.ndim != 2:
            raise ValueError(f"不支持的数据维度: {arr.shape}")
        return arr.astype(np.float32)

    def _load_auto_data(self):
        dataset_name = self.dataset_combo.currentText()
        split = self.split_combo.currentText()
        cfg = self._dataset_cfg(dataset_name)
        select_id = self._parse_sample_id(cfg)
        data_dir = self._dataset_dir(dataset_name)
        self._append_log(f"自动读取: dataset={dataset_name}, split={split}, select_id={select_id}")
        self._append_log(f"数据路径: {data_dir}")

        if dataset_name in ["SEGSalt", "SEGSimulation"]:
            seismic_data, velocity_model, _ = single_read_matfile(
                data_dir,
                cfg["data_dim"],
                cfg["model_dim"],
                select_id,
                train_or_test=split,
            )
        else:
            seismic_data, velocity_model, _ = single_read_npyfile(
                data_dir,
                select_id,
                train_or_test=split,
            )
        return self._extract_2d_data(seismic_data), self._extract_2d_data(velocity_model), dataset_name

    def _load_manual_seismic(self):
        if not self._last_manual_file:
            raise FileNotFoundError("请先选择 .npy 或 .mat 文件")
        path = self._last_manual_file
        ext = os.path.splitext(path)[1].lower()
        self._append_log(f"手动读取: {path}")

        if ext == ".npy":
            data = np.load(path, allow_pickle=True)
            return self._extract_2d_data(data), os.path.splitext(os.path.basename(path))[0]
        if ext == ".mat":
            import scipy.io

            mat = scipy.io.loadmat(path)
            candidates = []
            for k, v in mat.items():
                if k.startswith("__"):
                    continue
                arr = np.asarray(v)
                if arr.size == 0:
                    continue
                if arr.ndim >= 2:
                    candidates.append((k, arr))
            if not candidates:
                raise ValueError(".mat 文件中未找到可用二维数组")
            key, arr = max(candidates, key=lambda x: np.asarray(x[1]).size)
            self._append_log(f"使用字段: {key} shape={arr.shape}")
            return self._extract_2d_data(arr), os.path.splitext(os.path.basename(path))[0]
        raise ValueError("仅支持 .npy/.mat 文件")

    def _build_openfwi_seismic_pixmap(self, seismic_data, title, is_colorbar=True):
        from PyQt5.QtGui import QImage

        data = cv2.resize(np.asarray(seismic_data, dtype=np.float32), dsize=(400, 301), interpolation=cv2.INTER_CUBIC)

        # Keep aspect ratio consistent with velocity compare page.
        width, height = 580, 620
        right_margin = 24
        bottom_margin = 78
        title_y = 14

        tick_font = QFont()
        tick_font.setPointSize(11)
        tick_fm = QFontMetrics(tick_font)
        tick_h = tick_fm.height()

        axis_font = QFont()
        axis_font.setPointSize(11)
        axis_fm = QFontMetrics(axis_font)

        y_ticks = np.linspace(0.0, 1.0, 5)
        y_tick_texts = [f"{t:.2f}" if i in (1, 2, 3) else f"{t:.1f}" for i, t in enumerate(y_ticks)]
        max_y_tick_w = max(tick_fm.horizontalAdvance(txt) for txt in y_tick_texts)

        y_title_band = axis_fm.height() + 18
        gap_ytitle_to_ticks = 8
        y_tick_band = max_y_tick_w + 14
        left_margin = max(96, 12 + y_title_band + gap_ytitle_to_ticks + y_tick_band + 10)

        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title_fm = QFontMetrics(title_font)

        cbar_h = 16 if is_colorbar else 0
        cbar_tick_h = tick_h + 4 if is_colorbar else 0
        cbar_tick_len = 4 if is_colorbar else 0
        cbar_tick_gap = 3 if is_colorbar else 0
        cbar_pad = cbar_tick_len + cbar_tick_gap + cbar_tick_h + 8 if is_colorbar else 0
        top_margin = max(72, title_y + title_fm.height() + 12 + cbar_h + cbar_pad)

        plot_w = width - left_margin - right_margin
        plot_h = height - top_margin - bottom_margin
        plot_y = top_margin
        cbar_y = title_y + title_fm.height() + 12

        image = QImage(width, height, QImage.Format_RGB32)
        image.fill(QColor(248, 251, 255))
        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing)

        norm = np.clip((data - (-18.0)) / (19.0 - (-18.0)), 0.0, 1.0)
        cmap = mpl_cm.get_cmap("seismic")
        rgb = (cmap(norm)[..., :3] * 255).astype(np.uint8)
        plot_img = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3, QImage.Format_RGB888).copy()

        painter.drawImage(
            left_margin,
            plot_y,
            plot_img.scaled(plot_w, plot_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation),
        )

        if is_colorbar:
            cbar_norm = np.linspace(0.0, 1.0, 512, dtype=np.float32).reshape(1, 512)
            cbar_rgb = (cmap(cbar_norm)[..., :3] * 255).astype(np.uint8)
            cbar_img = QImage(cbar_rgb.data, 512, 1, 512 * 3, QImage.Format_RGB888).copy()
            painter.drawImage(
                left_margin,
                cbar_y,
                cbar_img.scaled(plot_w, cbar_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation),
            )
            painter.setPen(QPen(QColor(68, 68, 68), 1))
            painter.drawRect(left_margin, cbar_y, plot_w, cbar_h)

            painter.setFont(tick_font)
            painter.setPen(QColor(45, 45, 45))
            cbar_ticks = np.linspace(-18, 19, 7)
            for i, val in enumerate(cbar_ticks):
                ratio = i / 6.0
                x = left_margin + int(ratio * (plot_w - 1))
                painter.drawLine(x, cbar_y + cbar_h, x, cbar_y + cbar_h + cbar_tick_len)
                painter.drawText(
                    x - 26,
                    cbar_y + cbar_h + cbar_tick_len + cbar_tick_gap,
                    52,
                    cbar_tick_h,
                    Qt.AlignCenter,
                    f"{val:.0f}",
                )

            plot_y = cbar_y + cbar_h + cbar_pad

        painter.setPen(QPen(QColor(68, 68, 68), 2))
        painter.drawRect(left_margin, plot_y, plot_w, plot_h)

        painter.setFont(tick_font)
        painter.setPen(QColor(45, 45, 45))
        x_ticks = np.linspace(0.0, 0.7, 5)

        for i, t in enumerate(x_ticks):
            x = left_margin + int((i / 4.0) * plot_w)
            painter.drawLine(x, plot_y + plot_h, x, plot_y + plot_h + 5)
            tick_txt = f"{t:.2f}" if i in (1, 2, 3) else f"{t:.1f}"
            tw = tick_fm.horizontalAdvance(tick_txt) + 8
            painter.drawText(x - tw // 2, plot_y + plot_h + 8, tw, tick_h + 4, Qt.AlignCenter, tick_txt)

        for i, t in enumerate(y_ticks):
            y = plot_y + int((i / 4.0) * plot_h)
            painter.drawLine(left_margin - 5, y, left_margin, y)
            tick_txt = f"{t:.2f}" if i in (1, 2, 3) else f"{t:.1f}"
            y_text_w = left_margin - y_title_band - gap_ytitle_to_ticks - 12
            y_text_h = tick_h + 4
            y_text_x = y_title_band + gap_ytitle_to_ticks
            painter.drawText(y_text_x, y - y_text_h // 2, y_text_w, y_text_h, Qt.AlignRight | Qt.AlignVCenter, tick_txt)

        painter.setFont(axis_font)
        painter.drawText(left_margin, plot_y + plot_h + 36, plot_w, axis_fm.height() + 8, Qt.AlignCenter, "宽度(千米)")
        painter.save()
        y_title_center_x = max(10, y_title_band // 2)
        painter.translate(y_title_center_x, plot_y + plot_h // 2)
        painter.rotate(-90)
        painter.drawText(-plot_h // 2, 0, plot_h, axis_fm.height() + 8, Qt.AlignCenter, "时间(秒)")
        painter.restore()

        painter.end()
        return QPixmap.fromImage(image)

    def _build_openfwi_velocity_pixmap(self, velocity_model, min_v, max_v, title, is_colorbar=True):
        from PyQt5.QtGui import QImage

        data = np.asarray(velocity_model, dtype=np.float32)
        if data.shape != (70, 70):
            y_idx = np.linspace(0, max(data.shape[0] - 1, 0), 70).astype(int)
            x_idx = np.linspace(0, max(data.shape[1] - 1, 0), 70).astype(int)
            data = data[y_idx][:, x_idx]

        # Keep aspect ratio consistent with velocity compare page.
        width, height = 580, 620
        right_margin = 24
        bottom_margin = 78
        title_y = 14

        tick_font = QFont()
        tick_font.setPointSize(11)
        tick_fm = QFontMetrics(tick_font)
        tick_h = tick_fm.height()

        axis_font = QFont()
        axis_font.setPointSize(11)
        axis_fm = QFontMetrics(axis_font)

        y_ticks = np.linspace(0.0, 0.7, 8)
        y_tick_texts = [f"{t:.1f}" for t in y_ticks]
        max_y_tick_w = max(tick_fm.horizontalAdvance(txt) for txt in y_tick_texts)

        y_title_band = axis_fm.height() + 18
        gap_ytitle_to_ticks = 8
        y_tick_band = max_y_tick_w + 14
        left_margin = max(96, 12 + y_title_band + gap_ytitle_to_ticks + y_tick_band + 10)

        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title_fm = QFontMetrics(title_font)

        cbar_h = 16 if is_colorbar else 0
        cbar_tick_h = tick_h + 4 if is_colorbar else 0
        cbar_tick_len = 4 if is_colorbar else 0
        cbar_tick_gap = 3 if is_colorbar else 0
        cbar_pad = cbar_tick_len + cbar_tick_gap + cbar_tick_h + 8 if is_colorbar else 0
        top_margin = max(72, title_y + title_fm.height() + 12 + cbar_h + cbar_pad)

        plot_w = width - left_margin - right_margin
        plot_h = height - top_margin - bottom_margin
        plot_y = top_margin
        cbar_y = title_y + title_fm.height() + 12

        image = QImage(width, height, QImage.Format_RGB32)
        image.fill(QColor(248, 251, 255))
        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing)

        rng = max(max_v - min_v, 1e-8)
        norm = np.clip((data - min_v) / rng, 0.0, 1.0)
        cmap = mpl_cm.get_cmap("viridis")
        rgb = (cmap(norm)[..., :3] * 255).astype(np.uint8)
        plot_img = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3, QImage.Format_RGB888).copy()

        painter.drawImage(
            left_margin,
            plot_y,
            plot_img.scaled(plot_w, plot_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation),
        )

        if is_colorbar:
            cbar_norm = np.linspace(0.0, 1.0, 512, dtype=np.float32).reshape(1, 512)
            cbar_rgb = (cmap(cbar_norm)[..., :3] * 255).astype(np.uint8)
            cbar_img = QImage(cbar_rgb.data, 512, 1, 512 * 3, QImage.Format_RGB888).copy()
            painter.drawImage(
                left_margin,
                cbar_y,
                cbar_img.scaled(plot_w, cbar_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation),
            )
            painter.setPen(QPen(QColor(68, 68, 68), 1))
            painter.drawRect(left_margin, cbar_y, plot_w, cbar_h)

            painter.setFont(tick_font)
            painter.setPen(QColor(45, 45, 45))
            cbar_ticks = np.linspace(min_v, max_v, 7)
            for i, val in enumerate(cbar_ticks):
                ratio = i / 6.0
                x = left_margin + int(ratio * (plot_w - 1))
                painter.drawLine(x, cbar_y + cbar_h, x, cbar_y + cbar_h + cbar_tick_len)
                painter.drawText(
                    x - 45,  # 修改点 1：起始位置向左挪 (从 -26 改为 -45)
                    cbar_y + cbar_h + cbar_tick_len + cbar_tick_gap,
                    90,  # 修改点 2：文字框总宽度加宽 (从 52 改为 90)
                    cbar_tick_h,
                    Qt.AlignCenter,  # 居中对齐
                    f"{val:.0f}",
                )

            plot_y = cbar_y + cbar_h + cbar_pad

        painter.setPen(QPen(QColor(68, 68, 68), 2))
        painter.drawRect(left_margin, plot_y, plot_w, plot_h)

        painter.setFont(tick_font)
        painter.setPen(QColor(45, 45, 45))
        x_ticks = np.linspace(0.0, 0.7, 8)

        for i, t in enumerate(x_ticks):
            x = left_margin + int((i / 7.0) * plot_w)
            painter.drawLine(x, plot_y + plot_h, x, plot_y + plot_h + 5)
            tick_txt = f"{t:.1f}"
            tw = tick_fm.horizontalAdvance(tick_txt) + 8
            painter.drawText(x - tw // 2, plot_y + plot_h + 8, tw, tick_h + 4, Qt.AlignCenter, tick_txt)

        for i, t in enumerate(y_ticks):
            y = plot_y + int((i / 7.0) * plot_h)
            painter.drawLine(left_margin - 5, y, left_margin, y)
            tick_txt = f"{t:.1f}"
            y_text_w = left_margin - y_title_band - gap_ytitle_to_ticks - 12
            y_text_h = tick_h + 4
            y_text_x = y_title_band + gap_ytitle_to_ticks
            painter.drawText(y_text_x, y - y_text_h // 2, y_text_w, y_text_h, Qt.AlignRight | Qt.AlignVCenter, tick_txt)

        painter.setFont(axis_font)
        painter.drawText(left_margin, plot_y + plot_h + 36, plot_w, axis_fm.height() + 8, Qt.AlignCenter, "宽度(千米)")
        painter.save()
        y_title_center_x = max(10, y_title_band // 2)
        painter.translate(y_title_center_x, plot_y + plot_h // 2)
        painter.rotate(-90)
        painter.drawText(-plot_h // 2, 0, plot_h, axis_fm.height() + 8, Qt.AlignCenter, "深度(千米)")
        painter.restore()

        painter.end()
        return QPixmap.fromImage(image)

    def _render_to_label(self, seismic_data, velocity_model, title):
        seismic_pix = self._build_openfwi_seismic_pixmap(seismic_data, f"{title} - Seismic", is_colorbar=True)
        self._seismic_pixmap = seismic_pix
        self.seismic_label.setPixmap(
            seismic_pix.scaled(
                self.seismic_label.width(),
                self.seismic_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

        if velocity_model is not None:
            min_v = float(np.min(velocity_model))
            max_v = float(np.max(velocity_model))
            velocity_pix = self._build_openfwi_velocity_pixmap(velocity_model, min_v, max_v, f"{title} - Velocity", is_colorbar=True)
            self._velocity_pixmap = velocity_pix
            self.velocity_label.setPixmap(
                velocity_pix.scaled(
                    self.velocity_label.width(),
                    self.velocity_label.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )
        else:
            self._velocity_pixmap = None
            self.velocity_label.clear()
            self.velocity_label.setText("速度模型不可用")

    def _load_and_render(self):
        self.run_log.clear()
        try:
            mode = self.mode_combo.currentText()
            if mode == "自动读取":
                seismic_data, velocity_data, source = self._load_auto_data()
                title = f"{source} Seismic"
            else:
                seismic_data, source = self._load_manual_seismic()
                velocity_data = None
                title = f"{source} Seismic"
                self._append_log("手动文件模式未提供速度模型，已仅显示地震图")
            self._render_to_label(seismic_data, velocity_data, title)
            self._append_log(f"地震图渲染完成: shape={seismic_data.shape}")
            if velocity_data is not None:
                self._append_log(f"速度图渲染完成: shape={velocity_data.shape}")
        except Exception as e:
            self._append_log(f"ERROR: {e}")
            QMessageBox.warning(self, "数据观测失败", str(e))

    def _export_image(self):
        if (self._seismic_pixmap is None or self._seismic_pixmap.isNull()) and (self._velocity_pixmap is None or self._velocity_pixmap.isNull()):
            return
        os.makedirs(self._last_export_dir, exist_ok=True)
        default_name = f"{self.dataset_combo.currentText()}_seismic_observation.png"
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存观测图像",
            os.path.join(self._last_export_dir, default_name),
            "PNG (*.png);;JPG (*.jpg *.jpeg);;BMP (*.bmp)",
        )
        if not save_path:
            return
        self._last_export_dir = os.path.dirname(save_path)
        base, ext = os.path.splitext(save_path)
        ext = ext if ext else ".png"
        if self._seismic_pixmap is not None and not self._seismic_pixmap.isNull():
            seismic_path = f"{base}_seismic{ext}"
            self._seismic_pixmap.save(seismic_path)
            self._append_log(f"地震图已保存: {seismic_path}")
        if self._velocity_pixmap is not None and not self._velocity_pixmap.isNull():
            velocity_path = f"{base}_velocity{ext}"
            self._velocity_pixmap.save(velocity_path)
            self._append_log(f"速度图已保存: {velocity_path}")

    def _show_zoom(self, target):
        pix = self._seismic_pixmap if target == "seismic" else self._velocity_pixmap
        if pix is None or pix.isNull():
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("实验数据观测 - 放大图" if target == "seismic" else "速度模型 - 放大图")
        dialog.resize(1100, 820)
        layout = QVBoxLayout(dialog)
        zoom_label = QLabel(dialog)
        zoom_label.setAlignment(Qt.AlignCenter)
        zoom_label.setStyleSheet("background:#0f131a;")
        layout.addWidget(zoom_label)
        zoom_label.setPixmap(
            pix.scaled(1040, 760, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        dialog.exec_()


class ClickableLabel(QLabel):
    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class TrainingView(QWidget):
    train_requested = pyqtSignal()
    back_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._last_export_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "results",
        )
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(14)
        left_widget = QWidget(self)
        left_widget.setStyleSheet(PANEL_STYLE)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(18, 16, 18, 16)
        left_layout.setSpacing(10)
        self.model1_label = QLabel("对比模型1", self)
        self.model1_label.setAlignment(Qt.AlignCenter)
        self.combo = QComboBox(self)
        self.combo.addItems(["Scratch", "Sup-Mig", "Self-Sup"])
        self.model2_label = QLabel("对比模型2", self)
        self.model2_label.setAlignment(Qt.AlignCenter)
        self.compare_model_combo = QComboBox(self)
        self.compare_model_combo.addItems(["Scratch", "Sup-Mig", "Self-Sup"])
        self.model3_label = QLabel("对比模型3", self)
        self.model3_label.setAlignment(Qt.AlignCenter)
        self.compare_model_combo_2 = QComboBox(self)
        self.compare_model_combo_2.addItems(["Scratch", "Sup-Mig", "Self-Sup"])
        _apply_combo_style(self.combo)
        _apply_combo_style(self.compare_model_combo)
        _apply_combo_style(self.compare_model_combo_2)
        self.model1_label.setStyleSheet("color:#1a3550; font-size:13px; font-weight:600;")
        self.model2_label.setStyleSheet("color:#1a3550; font-size:13px; font-weight:600;")
        self.model3_label.setStyleSheet("color:#1a3550; font-size:13px; font-weight:600;")
        self.start_btn = BaseButton("显示Loss页面", self, color="#2c7fb8", radius=10)
        self.back_btn = BaseButton("返回", self, color="#60798f", radius=10)

        selector_widget = QWidget(self)
        selector_layout = QGridLayout(selector_widget)
        selector_layout.setContentsMargins(6, 4, 6, 4)
        selector_layout.setHorizontalSpacing(8)
        selector_layout.setVerticalSpacing(6)
        selector_layout.addWidget(self.model1_label, 0, 0)
        selector_layout.addWidget(self.combo, 0, 1)
        selector_layout.addWidget(self.model2_label, 1, 0)
        selector_layout.addWidget(self.compare_model_combo, 1, 1)
        selector_layout.addWidget(self.model3_label, 2, 0)
        selector_layout.addWidget(self.compare_model_combo_2, 2, 1)

        left_layout.addWidget(selector_widget, alignment=Qt.AlignCenter)
        left_layout.addWidget(self.start_btn, alignment=Qt.AlignCenter)
        left_layout.addWidget(self.back_btn, alignment=Qt.AlignCenter)
        root.addWidget(left_widget, 1)

        right_widget = QWidget(self)
        right_widget.setStyleSheet(PANEL_STYLE)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(12, 12, 12, 12)
        self.loss_panel = LossPanel(self)
        self.loss_panel.setStyleSheet("background:#f8fbff; border:1px solid #dbe5f0; border-radius:10px;")
        right_layout.addWidget(self.loss_panel)
        root.addWidget(right_widget, 2)
        self.setLayout(root)
        self.start_btn.clicked.connect(self._on_show_loss)
        self.loss_panel.clicked.connect(self._show_loss_zoom)
        self.compare_model_combo_2.currentTextChanged.connect(lambda _: self.loss_panel.update())
        self.back_btn.clicked.connect(self.back_requested.emit)

    def _show_loss_zoom(self):
        if len(self.loss_panel.train_loss) < 2:
            return

        pix = self.loss_panel.export_pixmap()
        if pix.isNull():
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Loss - 放大图")
        dialog.resize(920, 640)

        layout = QVBoxLayout(dialog)
        zoom_label = QLabel(dialog)
        zoom_label.setAlignment(Qt.AlignCenter)
        zoom_label.setStyleSheet("background:#f3f7fb;")
        layout.addWidget(zoom_label)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        zoom_download_btn = BaseButton("下载Loss图", dialog, color="#20B2AA", radius=8)
        btn_row.addWidget(zoom_download_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        zoom_label.setPixmap(
            pix.scaled(
                880,
                600,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )
        zoom_download_btn.clicked.connect(lambda: self._download_loss_image(pixmap=pix, parent=dialog))
        dialog.exec_()

    def _download_loss_image(self, pixmap=None, parent=None):
        if len(self.loss_panel.train_loss) < 2:
            return

        os.makedirs(self._last_export_dir, exist_ok=True)
        dataset_name = self.combo.currentText()
        default_name = f"{dataset_name}_loss.png"
        save_path, _ = QFileDialog.getSaveFileName(
            parent or self,
            "保存Loss图",
            os.path.join(self._last_export_dir, default_name),
            "PNG (*.png);;JPG (*.jpg *.jpeg);;BMP (*.bmp)",
        )
        if not save_path:
            return

        self._last_export_dir = os.path.dirname(save_path)
        export_pix = pixmap if pixmap is not None else self.loss_panel.export_pixmap()
        export_pix.save(save_path)

    def _load_loss_by_path(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".npy":
            arr = np.load(file_path)
        elif ext == ".npz":
            data = np.load(file_path)
            arr = data[data.files[0]] if data.files else np.array([])
        elif ext == ".csv":
            arr = np.loadtxt(file_path, delimiter=",")
        else:
            arr = np.loadtxt(file_path)
        arr = np.asarray(arr).reshape(-1)
        if arr.size == 0:
            raise ValueError(f"读取文件内容为空: {file_path}")
        return arr

    def _find_existing_path(self, paths):
        for path in paths:
            if os.path.exists(path):
                return path
        return None

    def _method_root_dir(self, method_name):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        base = os.path.join(project_root, "results", "CurveFaultAResults")
        mapping = {
            "Scratch": os.path.join(base, "unused_pretrain", "DDNet70", "CurveFaultA"),
            "Sup-Mig": os.path.join(base, "used_pretrain", "DDNet70", "CurveFaultA"),
            "Self-Sup": os.path.join(base, "self_sup", "DDNet70", "CurveFaultA"),
        }
        return mapping.get(method_name)

    def _find_method_loss_path(self, method_name):
        root = self._method_root_dir(method_name)
        if not root or not os.path.isdir(root):
            return None

        matches = []
        for cur_root, _, files in os.walk(root):
            for name in files:
                lower = name.lower()
                if lower.endswith(".npy") and "[loss]" in lower and "curvefaulta" in lower:
                    full = os.path.join(cur_root, name)
                    matches.append((os.path.getmtime(full), full))

        if not matches:
            return None
        matches.sort(key=lambda x: x[0], reverse=True)
        return matches[0][1]

    def _on_show_loss(self):
        model1_method = self.combo.currentText()
        model2_method = self.compare_model_combo.currentText()
        model3_method = self.compare_model_combo_2.currentText()

        model1_loss_path = self._find_method_loss_path(model1_method)
        model2_loss_path = self._find_method_loss_path(model2_method)
        model3_loss_path = self._find_method_loss_path(model3_method)
        if not model1_loss_path or not model2_loss_path or not model3_loss_path:
            self.loss_panel.set_error(
                f"未找到三模型Loss文件: {model1_method}={bool(model1_loss_path)}, {model2_method}={bool(model2_loss_path)}, {model3_method}={bool(model3_loss_path)}"
            )
            return

        try:
            # 对齐 utils.save_results 的绘制逻辑：默认从 loss[1:] 开始绘制
            full_loss = self._load_loss_by_path(model1_loss_path)
            train_loss = full_loss[1:] if full_loss.size > 1 else full_loss

            val_loss_raw = self._load_loss_by_path(model2_loss_path)
            val_loss = val_loss_raw[1:] if val_loss_raw.size > 1 else val_loss_raw

            third_loss_raw = self._load_loss_by_path(model3_loss_path)
            third_loss = third_loss_raw[1:] if third_loss_raw.size > 1 else third_loss_raw

            min_len = min(len(train_loss), len(val_loss), len(third_loss))
            if min_len < 2:
                raise ValueError("对比 Loss 数据长度不足")
            train_loss = train_loss[:min_len]
            val_loss = val_loss[:min_len]
            third_loss = third_loss[:min_len]

            self.loss_panel.set_data(
                train_loss,
                val_loss,
                third_loss,
                smooth=None,
                epochs=min_len - 1,
                x_title="epoch",
                y_title="Loss",
                title="Loss对比图",
                train_label=model1_method,
                val_label=model2_method,
                third_label=model3_method,
            )
        except Exception as e:
            self.loss_panel.set_error(f"数据读取失败: {e}")


class MetricAnalysisView(QWidget):
    back_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self._metric_results_dir = os.path.join(self._project_root, "results", "CurveFaultAResults")
        self._cache_dir = os.path.join(self._metric_results_dir, "metric_cache")
        self._snapshot_cache_path = os.path.join(self._cache_dir, "latest_compare_batch_snapshot.json")
        self._memory_pair_cache = {}
        self._memory_snapshot = None
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(14)

        left_widget = QWidget(self)
        left_widget.setStyleSheet(PANEL_STYLE)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(18, 16, 18, 16)
        left_layout.setSpacing(10)
        self.model1_label = QLabel("对比模型1", self)
        self.model1_label.setAlignment(Qt.AlignCenter)
        self.combo = QComboBox(self)
        self.combo.addItems(["Scratch", "Sup-Mig", "Self-Sup"])
        self.model2_label = QLabel("对比模型2", self)
        self.model2_label.setAlignment(Qt.AlignCenter)
        self.compare_model_combo = QComboBox(self)
        self.compare_model_combo.addItems(["Scratch", "Sup-Mig", "Self-Sup"])
        self.model3_label = QLabel("对比模型3", self)
        self.model3_label.setAlignment(Qt.AlignCenter)
        self.compare_model_combo_2 = QComboBox(self)
        self.compare_model_combo_2.addItems(["Scratch", "Sup-Mig", "Self-Sup"])
        _apply_combo_style(self.combo)
        _apply_combo_style(self.compare_model_combo)
        _apply_combo_style(self.compare_model_combo_2)
        self.model1_label.setStyleSheet("color:#1a3550; font-size:13px; font-weight:600;")
        self.model2_label.setStyleSheet("color:#1a3550; font-size:13px; font-weight:600;")
        self.model3_label.setStyleSheet("color:#1a3550; font-size:13px; font-weight:600;")
        self.start_btn = BaseButton("重新计算并刷新", self, color="#1f9d8a", radius=10)
        self.open_btn = BaseButton("保存", self, color="#2c7fb8", radius=10)
        self.back_btn = BaseButton("返回", self, color="#60798f", radius=10)

        selector_widget = QWidget(self)
        selector_layout = QGridLayout(selector_widget)
        selector_layout.setContentsMargins(6, 4, 6, 4)
        selector_layout.setHorizontalSpacing(8)
        selector_layout.setVerticalSpacing(6)
        selector_layout.addWidget(self.model1_label, 0, 0)
        selector_layout.addWidget(self.combo, 0, 1)
        selector_layout.addWidget(self.model2_label, 1, 0)
        selector_layout.addWidget(self.compare_model_combo, 1, 1)
        selector_layout.addWidget(self.model3_label, 2, 0)
        selector_layout.addWidget(self.compare_model_combo_2, 2, 1)

        left_layout.addWidget(selector_widget, alignment=Qt.AlignCenter)
        left_layout.addWidget(self.start_btn, alignment=Qt.AlignCenter)
        left_layout.addWidget(self.open_btn, alignment=Qt.AlignCenter)
        left_layout.addWidget(self.back_btn, alignment=Qt.AlignCenter)
        root.addWidget(left_widget, 1)

        right_widget = QWidget(self)
        right_widget.setStyleSheet(PANEL_STYLE)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(12, 12, 12, 12)
        right_layout.setSpacing(10)
        self.result_table = QTableWidget(6, 4, self)
        self.result_table.setHorizontalHeaderLabels(["指标", "模型1", "模型2", "模型3"])
        self.result_table.verticalHeader().setVisible(False)
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.result_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.result_table.setSelectionMode(QTableWidget.NoSelection)
        self.result_table.setAlternatingRowColors(True)
        self.result_table.setStyleSheet(
            "QTableWidget {"
            "background:#fbfdff;"
            "alternate-background-color:#f1f6fb;"
            "gridline-color:#d7e2ee;"
            "border:1px solid #d7e2ee;"
            "border-radius:8px;"
            "color:#23364a;"
            "font-size:13px;"
            "}"
            "QHeaderView::section {"
            "background:#dfeaf6;"
            "color:#1f3347;"
            "padding:6px;"
            "border:none;"
            "border-bottom:1px solid #c8d7e8;"
            "font-weight:600;"
            "}"
        )

        metric_names = ["MSE", "MAE", "UQI", "LPIPS", "推理耗时(s)", "样本数"]
        for i, name in enumerate(metric_names):
            self.result_table.setItem(i, 0, QTableWidgetItem(name))
            self.result_table.setItem(i, 1, QTableWidgetItem("--"))
            self.result_table.setItem(i, 2, QTableWidgetItem("--"))
            self.result_table.setItem(i, 3, QTableWidgetItem("--"))

        self.table_hint = QLabel("请选择模型后点击“重新计算并刷新”开始运行测试", self)
        self.table_hint.setAlignment(Qt.AlignCenter)
        self.table_hint.setStyleSheet("color:#5b6f83; font-size:12px; padding:4px 0;")
        self.run_log = QPlainTextEdit(self)
        self.run_log.setReadOnly(True)
        self.run_log.setPlaceholderText("运行日志将在这里显示")
        self.run_log.setMinimumHeight(180)
        self.run_log.setStyleSheet(
            "QPlainTextEdit {"
            "background:#0f1a26;"
            "color:#d7e7ff;"
            "border:1px solid #24384d;"
            "border-radius:8px;"
            "padding:8px;"
            "font-family:Consolas, 'Courier New', monospace;"
            "font-size:12px;"
            "}"
        )
        right_layout.addWidget(self.result_table)
        right_layout.addWidget(self.table_hint)
        right_layout.addWidget(self.run_log)
        root.addWidget(right_widget, 2)

        self.setLayout(root)
        self.start_btn.clicked.connect(self._on_show_metrics)
        self.open_btn.clicked.connect(self._on_open_existing_file)
        self.combo.currentTextChanged.connect(self._on_model_selection_changed)
        self.compare_model_combo.currentTextChanged.connect(self._on_model_selection_changed)
        self.compare_model_combo_2.currentTextChanged.connect(self._on_model_selection_changed)
        self._refresh_table_headers()
        self._try_render_from_cache(prefer_disk=True)
        self.back_btn.clicked.connect(self.back_requested.emit)

    def _refresh_table_headers(self):
        self.result_table.setHorizontalHeaderLabels([
            "指标",
            self.combo.currentText(),
            self.compare_model_combo.currentText(),
            self.compare_model_combo_2.currentText(),
        ])

    def _method_to_alias_candidates(self, method_name):
        fixed_mapping = {
            "Scratch": ["Baseline-DDNet70-Scratch"],
            "Sup-Mig": ["Proposed-DDNet70-PretrainFinetune", "Sup-Mig"],
            "Self-Sup": ["Self-Sup", "SelfSup", "self_sup"],
        }
        return fixed_mapping.get(method_name, [method_name])

    def _pair_cache_key(self, model1_name, model2_name, model3_name):
        return f"CurveFaultA|{model1_name}|{model2_name}|{model3_name}"

    def _pair_cache_file(self, pair_key):
        digest = hashlib.md5(pair_key.encode("utf-8")).hexdigest()
        return os.path.join(self._cache_dir, f"pair_{digest}.json")

    def _csv_fingerprint(self, csv_path):
        stat = os.stat(csv_path)
        return {
            "name": os.path.basename(csv_path),
            "path": csv_path,
            "size": int(stat.st_size),
            "mtime": float(stat.st_mtime),
        }

    def _now_text(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _ensure_cache_dir(self):
        os.makedirs(self._cache_dir, exist_ok=True)

    def _build_snapshot_payload(self, rows, csv_path):
        rows_by_alias = {}
        for row in rows:
            alias = str(row.get("alias", "")).strip()
            if not alias:
                continue
            rows_by_alias[alias] = row
        return {
            "dataset": "CurveFaultA",
            "csv": self._csv_fingerprint(csv_path),
            "refreshed_at": self._now_text(),
            "rows_by_alias": rows_by_alias,
        }

    def _save_snapshot_cache(self, snapshot):
        self._ensure_cache_dir()
        with open(self._snapshot_cache_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)

    def _load_snapshot_cache_from_disk(self):
        if not os.path.exists(self._snapshot_cache_path):
            return None
        with open(self._snapshot_cache_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else None

    def _build_pair_entry(self, snapshot, model1_name, model2_name, model3_name):
        rows = list(snapshot.get("rows_by_alias", {}).values())
        model1_row = self._pick_row_for_method(rows, model1_name)
        model2_row = self._pick_row_for_method(rows, model2_name)
        model3_row = self._pick_row_for_method(rows, model3_name)
        if model1_row is None or model2_row is None or model3_row is None:
            raise ValueError(
                f"缓存中未找到所选模型结果: {model1_name}={bool(model1_row)}, {model2_name}={bool(model2_row)}, {model3_name}={bool(model3_row)}"
            )

        return {
            "pair_key": self._pair_cache_key(model1_name, model2_name, model3_name),
            "dataset": "CurveFaultA",
            "model1": model1_name,
            "model2": model2_name,
            "model3": model3_name,
            "csv": snapshot.get("csv", {}),
            "refreshed_at": snapshot.get("refreshed_at", ""),
            "metrics": {
                "mse": [
                    self._format_metric(model1_row, "mse_mean"),
                    self._format_metric(model2_row, "mse_mean"),
                    self._format_metric(model3_row, "mse_mean"),
                ],
                "mae": [
                    self._format_metric(model1_row, "mae_mean"),
                    self._format_metric(model2_row, "mae_mean"),
                    self._format_metric(model3_row, "mae_mean"),
                ],
                "uqi": [
                    self._format_metric(model1_row, "uqi_mean"),
                    self._format_metric(model2_row, "uqi_mean"),
                    self._format_metric(model3_row, "uqi_mean"),
                ],
                "lpips": [
                    self._format_metric(model1_row, "lpips_mean"),
                    self._format_metric(model2_row, "lpips_mean"),
                    self._format_metric(model3_row, "lpips_mean"),
                ],
                "time": [
                    self._format_metric(model1_row, "per_sample_seconds"),
                    self._format_metric(model2_row, "per_sample_seconds"),
                    self._format_metric(model3_row, "per_sample_seconds"),
                ],
                "sample_count": [
                    self._format_metric(model1_row, "sample_count"),
                    self._format_metric(model2_row, "sample_count"),
                    self._format_metric(model3_row, "sample_count"),
                ],
            },
        }

    def _save_pair_cache(self, entry):
        pair_key = entry.get("pair_key")
        if not pair_key:
            return
        self._ensure_cache_dir()
        path = self._pair_cache_file(pair_key)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False, indent=2)

    def _download_csv_copy(self, src_csv_path):
        if not src_csv_path or not os.path.exists(src_csv_path):
            QMessageBox.information(self, "下载提示", "缓存关联的CSV文件不存在，无法下载。")
            return

        default_name = os.path.basename(src_csv_path)
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "下载缓存CSV",
            os.path.join(self._metric_results_dir, default_name),
            "CSV (*.csv)",
        )
        if not save_path:
            return
        if not save_path.lower().endswith(".csv"):
            save_path += ".csv"

        shutil.copy2(src_csv_path, save_path)
        QMessageBox.information(self, "下载完成", f"已保存到:\n{save_path}")

    def _prompt_download_if_cached(self):
        model1_name = self.combo.currentText()
        model2_name = self.compare_model_combo.currentText()
        model3_name = self.compare_model_combo_2.currentText()
        pair_key = self._pair_cache_key(model1_name, model2_name, model3_name)

        entry = self._memory_pair_cache.get(pair_key) or self._load_pair_cache_from_disk(pair_key)
        if not entry:
            return False

        csv_meta = entry.get("csv", {})
        csv_name = csv_meta.get("name", "未知CSV")
        self.table_hint.setText(f"检测到缓存结果: {csv_name}")
        choose = QMessageBox.question(
            self,
            "缓存提示",
            f"当前模型组合已有缓存结果（{csv_name}）。\n是否下载该缓存对应CSV文件？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if choose == QMessageBox.Yes:
            self._download_csv_copy(csv_meta.get("path"))
            return True
        return True

    def _load_pair_cache_from_disk(self, pair_key):
        path = self._pair_cache_file(pair_key)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else None

    def _load_metrics_from_csv_path(self, csv_path, source_tag="已有文件"):
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError("所选CSV为空")

        snapshot = self._build_snapshot_payload(rows, csv_path)
        self._memory_snapshot = snapshot
        self._save_snapshot_cache(snapshot)

        model1_name = self.combo.currentText()
        model2_name = self.compare_model_combo.currentText()
        model3_name = self.compare_model_combo_2.currentText()
        pair_entry = self._build_pair_entry(snapshot, model1_name, model2_name, model3_name)
        pair_key = pair_entry["pair_key"]
        self._memory_pair_cache[pair_key] = pair_entry
        self._save_pair_cache(pair_entry)
        self._apply_pair_entry_to_table(pair_entry, source_tag=source_tag)

    def _clear_metric_table_values(self):
        for row in range(self.result_table.rowCount()):
            self.result_table.setItem(row, 1, QTableWidgetItem("--"))
            self.result_table.setItem(row, 2, QTableWidgetItem("--"))
            self.result_table.setItem(row, 3, QTableWidgetItem("--"))

    def _on_open_existing_file(self):
        if self._prompt_download_if_cached():
            return

        default_dir = self._metric_results_dir if os.path.isdir(self._metric_results_dir) else self._project_root
        candidates = glob.glob(os.path.join(self._metric_results_dir, "[[]CompareBatch[]]CurveFaultA_*.csv"))
        if not candidates:
            has_cache = self._try_render_from_cache(prefer_disk=True)
            if not has_cache:
                self._clear_metric_table_values()
                QMessageBox.information(
                    self,
                    "文件提示",
                    "未找到可打开的对比文件，且当前无缓存数据。\n请点击“重新计算并刷新”生成结果。",
                )
                self.table_hint.setText("未找到可打开文件且缓存为空，请点击“重新计算并刷新”")
            else:
                QMessageBox.information(self, "文件提示", "未找到已有文件，已保留缓存结果。")
            return

        csv_path, _ = QFileDialog.getOpenFileName(
            self,
            "打开已有指标文件",
            default_dir,
            "CSV (*.csv)",
        )
        if not csv_path:
            return

        self.run_log.clear()
        self._append_run_log(f"打开已有文件: {csv_path}")
        try:
            self._load_metrics_from_csv_path(csv_path, source_tag="打开已有文件")
            self._append_run_log("已有文件加载完成")
        except Exception as e:
            self._append_run_log(f"ERROR: {e}")
            self.table_hint.setText(f"加载已有文件失败: {e}")

    def _apply_pair_entry_to_table(self, entry, source_tag):
        metrics = entry.get("metrics", {})
        self._set_metric_row(0, *self._normalize_metric_values(metrics.get("mse", [])))
        self._set_metric_row(1, *self._normalize_metric_values(metrics.get("mae", [])))
        self._set_metric_row(2, *self._normalize_metric_values(metrics.get("uqi", [])))
        self._set_metric_row(3, *self._normalize_metric_values(metrics.get("lpips", [])))
        self._set_metric_row(4, *self._normalize_metric_values(metrics.get("time", [])))
        self._set_metric_row(5, *self._normalize_metric_values(metrics.get("sample_count", [])))

        csv_meta = entry.get("csv", {})
        csv_name = csv_meta.get("name") or "未知CSV"
        refreshed_at = entry.get("refreshed_at", "未知时间")
        self.table_hint.setText(f"来源: {source_tag} | 数据: {csv_name} | 刷新: {refreshed_at}")

    def _on_model_selection_changed(self):
        self._refresh_table_headers()
        self._try_render_from_cache(prefer_disk=True)

    def _try_render_from_cache(self, prefer_disk=False):
        model1_name = self.combo.currentText()
        model2_name = self.compare_model_combo.currentText()
        model3_name = self.compare_model_combo_2.currentText()
        pair_key = self._pair_cache_key(model1_name, model2_name, model3_name)

        cached = self._memory_pair_cache.get(pair_key)
        if cached:
            self._apply_pair_entry_to_table(cached, source_tag="内存缓存(L1)")
            return True

        if prefer_disk:
            pair_disk = self._load_pair_cache_from_disk(pair_key)
            if pair_disk:
                self._memory_pair_cache[pair_key] = pair_disk
                self._apply_pair_entry_to_table(pair_disk, source_tag="磁盘缓存(L2)")
                return True

        snapshot = self._memory_snapshot
        if snapshot is None and prefer_disk:
            snapshot = self._load_snapshot_cache_from_disk()
            if snapshot:
                self._memory_snapshot = snapshot

        if snapshot:
            try:
                entry = self._build_pair_entry(snapshot, model1_name, model2_name, model3_name)
                self._memory_pair_cache[pair_key] = entry
                self._save_pair_cache(entry)
                self._apply_pair_entry_to_table(entry, source_tag="快照缓存")
                return True
            except Exception:
                pass

        self.table_hint.setText("当前组合暂无缓存，请点击“重新计算并刷新”")
        return False

    def _run_model_test(self):
        script_path = os.path.join(self._project_root, "model_test.py")
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"未找到测试脚本: {script_path}")

        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        proc = subprocess.Popen(
            [sys.executable, script_path],
            cwd=self._project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            bufsize=1,
        )
        log_tail = []
        if proc.stdout is not None:
            for line in proc.stdout:
                line = line.rstrip("\r\n")
                if line:
                    self._append_run_log(line)
                    log_tail.append(line)
                    if len(log_tail) > 12:
                        log_tail.pop(0)
        proc.wait()
        if proc.returncode != 0:
            msg = "\n".join(log_tail[-8:]) if log_tail else "未知错误"
            raise RuntimeError(f"model_test.py 运行失败 (code={proc.returncode})\n{msg}")

    def _append_run_log(self, text):
        if not text:
            return
        self.run_log.appendPlainText(text)
        bar = self.run_log.verticalScrollBar()
        bar.setValue(bar.maximum())
        QApplication.processEvents()

    def _find_latest_compare_batch_csv(self):
        pattern = os.path.join(self._metric_results_dir, "[[]CompareBatch[]]CurveFaultA_*.csv")
        candidates = glob.glob(pattern)
        if not candidates:
            return None
        candidates.sort(key=os.path.getmtime, reverse=True)
        return candidates[0]

    def _pick_row_for_method(self, rows, method_name):
        candidates = self._method_to_alias_candidates(method_name)
        rows_ok = [r for r in rows if str(r.get("status", "")).lower() == "ok"]
        search_rows = rows_ok if rows_ok else rows

        for alias in candidates:
            for row in search_rows:
                if row.get("alias") == alias:
                    return row

        tokens = [t.lower() for t in method_name.replace("-", " ").split() if t]
        for row in search_rows:
            alias = str(row.get("alias", "")).lower()
            if tokens and all(t in alias for t in tokens):
                return row
        return None

    def _normalize_metric_values(self, values):
        data = list(values) if isinstance(values, (list, tuple)) else []
        while len(data) < 3:
            data.append("--")
        return data[:3]

    def _set_metric_row(self, row_index, model1_val, model2_val, model3_val):
        self.result_table.setItem(row_index, 1, QTableWidgetItem(model1_val))
        self.result_table.setItem(row_index, 2, QTableWidgetItem(model2_val))
        self.result_table.setItem(row_index, 3, QTableWidgetItem(model3_val))

    def _format_metric(self, row, key):
        if row is None:
            return "--"
        value = row.get(key, "")
        if value in (None, ""):
            return "--"
        try:
            return f"{float(value):.6f}"
        except (TypeError, ValueError):
            return str(value)

    def _on_show_metrics(self):
        self._refresh_table_headers()
        self.table_hint.setText("正在运行 model_test.py 并刷新指标，请稍候...")
        self.run_log.clear()
        self._append_run_log(f"$ {sys.executable} model_test.py")
        self.start_btn.setEnabled(False)

        try:
            self._run_model_test()
            csv_path = self._find_latest_compare_batch_csv()
            if not csv_path:
                raise FileNotFoundError("未找到 [CompareBatch]CurveFaultA_*.csv 输出文件")

            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            if not rows:
                raise ValueError("对比结果文件为空")

            self._load_metrics_from_csv_path(csv_path, source_tag="实时计算")

            self._append_run_log("model_test.py 运行完成")
        except Exception as e:
            self._append_run_log(f"ERROR: {e}")
            self.table_hint.setText(f"指标刷新失败: {e}")
        finally:
            self.start_btn.setEnabled(True)

    def refresh_metrics(self):
        self._on_show_metrics()


class CompareView(QWidget):
    train_requested = pyqtSignal()
    back_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image_pixmaps = {}
        self._cached_compare = {}
        self._project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self._dataset_name = DEFAULT_DATASET_NAME
        self._model_runtime_cache = {}
        self._manual_model_paths = {"A": None, "B": None}
        self._last_export_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "results",
        )
        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.setSpacing(14)
        left_widget = QWidget(self)
        left_widget.setStyleSheet(PANEL_STYLE)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(18, 16, 18, 16)
        left_layout.setSpacing(12)

        self.dataset_label = QLabel(f"数据集：{self._dataset_name}", self)
        self.dataset_label.setAlignment(Qt.AlignCenter)
        self.dataset_label.setStyleSheet("color:#1a3550; font-size:13px; font-weight:600;")
        self.dataset_path_hint = QLabel("", self)
        self.dataset_path_hint.setWordWrap(True)
        self.dataset_path_hint.setStyleSheet("color:#5b6f83; font-size:11px;")

        self.strategy_a_label = QLabel("策略A", self)
        self.strategy_a_combo = QComboBox(self)
        self.strategy_a_combo.addItems(["Scratch", "Sup-Mig", "Self-Sup"])
        _apply_combo_style(self.strategy_a_combo)
        self.strategy_a_file_btn = BaseButton("选择模型文件", self, color="#2c7fb8", radius=8)
        self.strategy_a_path_hint = QLabel("", self)
        self.strategy_a_path_hint.setWordWrap(True)
        self.strategy_a_path_hint.setStyleSheet("color:#5b6f83; font-size:11px;")

        self.strategy_b_label = QLabel("策略B", self)
        self.strategy_b_combo = QComboBox(self)
        self.strategy_b_combo.addItems(["Sup-Mig", "Scratch", "Self-Sup"])
        _apply_combo_style(self.strategy_b_combo)
        self.strategy_b_file_btn = BaseButton("选择模型文件", self, color="#2c7fb8", radius=8)
        self.strategy_b_path_hint = QLabel("", self)
        self.strategy_b_path_hint.setWordWrap(True)
        self.strategy_b_path_hint.setStyleSheet("color:#5b6f83; font-size:11px;")
        self.strategy_a_label.setStyleSheet("color:#1a3550; font-size:13px; font-weight:600;")
        self.strategy_b_label.setStyleSheet("color:#1a3550; font-size:13px; font-weight:600;")

        self.start_btn = BaseButton("运行对比", self, color="#2c7fb8", radius=10)
        self.pd_btn = BaseButton("显示PD-A", self, color="#1f9d8a", radius=10)
        self.gt_btn = BaseButton("显示GT", self, color="#60798f", radius=10)
        self.pd_b_btn = BaseButton("显示PD-B", self, color="#d9822b", radius=10)
        self.back_btn = BaseButton("返回", self, color="#60798f", radius=10)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(self.gt_btn)
        btn_row.addSpacing(10)
        btn_row.addWidget(self.pd_btn)
        btn_row.addSpacing(10)
        btn_row.addWidget(self.pd_b_btn)
        btn_row.addStretch()

        left_layout.addWidget(self.dataset_label, alignment=Qt.AlignCenter)
        left_layout.addWidget(self.dataset_path_hint)
        left_layout.addWidget(self.strategy_a_label, alignment=Qt.AlignCenter)
        left_layout.addWidget(self.strategy_a_combo, alignment=Qt.AlignCenter)
        left_layout.addWidget(self.strategy_a_file_btn, alignment=Qt.AlignCenter)
        left_layout.addWidget(self.strategy_a_path_hint)
        left_layout.addWidget(self.strategy_b_label, alignment=Qt.AlignCenter)
        left_layout.addWidget(self.strategy_b_combo, alignment=Qt.AlignCenter)
        left_layout.addWidget(self.strategy_b_file_btn, alignment=Qt.AlignCenter)
        left_layout.addWidget(self.strategy_b_path_hint)
        left_layout.addWidget(self.start_btn, alignment=Qt.AlignCenter)
        left_layout.addLayout(btn_row)
        self.compare_hint = QLabel("请选择策略后，点击“运行对比”开始实验", self)
        self.compare_hint.setAlignment(Qt.AlignCenter)
        self.compare_hint.setStyleSheet("color:#5b6f83; font-size:12px; padding:4px 0;")
        left_layout.addWidget(self.compare_hint, alignment=Qt.AlignCenter)
        left_layout.addWidget(self.back_btn, alignment=Qt.AlignCenter)

        root.addWidget(left_widget)

        # 右侧
        right_widget = QWidget(self)
        right_widget.setStyleSheet(PANEL_STYLE)
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(10, 10, 10, 10)

        img_col = QVBoxLayout()
        img_col.setContentsMargins(12, 12, 12, 12)
        img_col.setSpacing(10)

        self.gt_label = ClickableLabel("GT")
        self.pd_a_label = ClickableLabel("PD-A")
        self.pd_b_label = ClickableLabel("PD-B")

        for lbl in [self.gt_label, self.pd_a_label, self.pd_b_label]:
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setMinimumSize(260, 200)
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            lbl.setStyleSheet(
                "border:1px solid #c7d7e6;"
                "border-radius:10px;"
                "background:qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #f8fbff, stop:1 #edf4fc);"
                "color:#2f3b4a;"
            )

        img_col.addWidget(self.gt_label, 1)
        img_col.addWidget(self.pd_a_label, 1)
        img_col.addWidget(self.pd_b_label, 1)
        right_layout.addLayout(img_col)

        self.run_log = QPlainTextEdit(self)
        self.run_log.setReadOnly(True)
        self.run_log.setPlaceholderText("终端运行信息将在这里显示")
        self.run_log.setMinimumHeight(150)
        self.run_log.setStyleSheet(
            "QPlainTextEdit {"
            "background:#0f1a26;"
            "color:#d7e7ff;"
            "border:1px solid #24384d;"
            "border-radius:8px;"
            "padding:8px;"
            "font-family:Consolas, 'Courier New', monospace;"
            "font-size:12px;"
            "}"
        )
        right_layout.addWidget(self.run_log)

        root.addWidget(right_widget, 2)

        self.start_btn.clicked.connect(self._run_strategy_compare)
        self.strategy_a_file_btn.clicked.connect(lambda: self._choose_model_file("A"))
        self.strategy_b_file_btn.clicked.connect(lambda: self._choose_model_file("B"))
        self.pd_btn.clicked.connect(self.show_pd)
        self.gt_btn.clicked.connect(self.show_gt)
        self.pd_b_btn.clicked.connect(self.show_pd_b)
        self.back_btn.clicked.connect(self.back_requested.emit)
        self.gt_label.clicked.connect(lambda: self._show_zoom(self.gt_label, "GT"))
        self.pd_a_label.clicked.connect(lambda: self._show_zoom(self.pd_a_label, "PD-A"))
        self.pd_b_label.clicked.connect(lambda: self._show_zoom(self.pd_b_label, "PD-B"))
        self.strategy_a_combo.currentTextChanged.connect(self._on_compare_selection_changed)
        self.strategy_b_combo.currentTextChanged.connect(self._on_compare_selection_changed)

        self._refresh_compare_path_hints()
        self._set_compare_placeholders()

    def show_pd(self):
        if not self._update_compare_images(focus="pd_a", allow_compute=False):
            self.compare_hint.setText("当前组合暂无结果，请先点击“运行对比”")
            return
        self.pd_a_label.setStyleSheet(
            "border:2px solid #1f9d8a;"
            "border-radius:10px;"
            "background:qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #f8fbff, stop:1 #edf4fc);"
            "color:#2f3b4a;"
        )
        self.gt_label.setStyleSheet(
            "border:1px solid #c7d7e6;"
            "border-radius:10px;"
            "background:qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #f8fbff, stop:1 #edf4fc);"
            "color:#2f3b4a;"
        )
        self.pd_b_label.setStyleSheet(
            "border:1px solid #c7d7e6;"
            "border-radius:10px;"
            "background:qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #f8fbff, stop:1 #edf4fc);"
            "color:#2f3b4a;"
        )

    def show_gt(self):
        if not self._update_compare_images(focus="gt", allow_compute=False):
            self.compare_hint.setText("当前组合暂无结果，请先点击“运行对比”")
            return
        self.gt_label.setStyleSheet(
            "border:2px solid #60798f;"
            "border-radius:10px;"
            "background:qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #f8fbff, stop:1 #edf4fc);"
            "color:#2f3b4a;"
        )
        self.pd_a_label.setStyleSheet(
            "border:1px solid #c7d7e6;"
            "border-radius:10px;"
            "background:qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #f8fbff, stop:1 #edf4fc);"
            "color:#2f3b4a;"
        )
        self.pd_b_label.setStyleSheet(
            "border:1px solid #c7d7e6;"
            "border-radius:10px;"
            "background:qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #f8fbff, stop:1 #edf4fc);"
            "color:#2f3b4a;"
        )

    def show_pd_b(self):
        if not self._update_compare_images(focus="pd_b", allow_compute=False):
            self.compare_hint.setText("当前组合暂无结果，请先点击“运行对比”")
            return
        self.pd_b_label.setStyleSheet(
            "border:2px solid #d9822b;"
            "border-radius:10px;"
            "background:qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #f8fbff, stop:1 #edf4fc);"
            "color:#2f3b4a;"
        )
        self.gt_label.setStyleSheet(
            "border:1px solid #c7d7e6;"
            "border-radius:10px;"
            "background:qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #f8fbff, stop:1 #edf4fc);"
            "color:#2f3b4a;"
        )
        self.pd_a_label.setStyleSheet(
            "border:1px solid #c7d7e6;"
            "border-radius:10px;"
            "background:qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #f8fbff, stop:1 #edf4fc);"
            "color:#2f3b4a;"
        )

    def show_image(self, label, img_array):
        import numpy as np
        from PyQt5.QtGui import QImage, QPixmap

        if img_array is None:
            return

        if img_array.dtype != np.uint8:
            img_array = (255 * (img_array - img_array.min()) / (np.ptp(img_array) + 1e-8)).astype(np.uint8)

        h, w = img_array.shape
        qimg = QImage(img_array.data, w, h, w, QImage.Format_Grayscale8)
        pix = QPixmap.fromImage(qimg)
        self._image_pixmaps[label] = pix
        label.setPixmap(pix.scaled(label.width(), label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _show_zoom(self, label, title):
        pix = self._image_pixmaps.get(label)
        if pix is None or pix.isNull():
            return

        dialog = QDialog(self)
        dialog.setWindowTitle(f"{title} - 放大图")
        dialog.resize(720, 720)

        layout = QVBoxLayout(dialog)
        zoom_label = QLabel(dialog)
        zoom_label.setAlignment(Qt.AlignCenter)
        zoom_label.setStyleSheet("background:#0f131a;")
        layout.addWidget(zoom_label)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        zoom_download_btn = BaseButton("下载图像", dialog, color="#20B2AA", radius=8)
        btn_row.addWidget(zoom_download_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        zoom_label.setPixmap(
            pix.scaled(
                680,
                680,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )
        zoom_download_btn.clicked.connect(lambda: self._download_compare_image(title, pix, dialog))
        dialog.exec_()

    def _download_compare_image(self, title, pixmap, parent=None):
        if pixmap is None or pixmap.isNull():
            return

        os.makedirs(self._last_export_dir, exist_ok=True)
        dataset_name = self._dataset_name
        default_name = f"{dataset_name}_{title.lower()}.png"
        save_path, _ = QFileDialog.getSaveFileName(
            parent or self,
            "保存图像",
            os.path.join(self._last_export_dir, default_name),
            "PNG (*.png);;JPG (*.jpg *.jpeg);;BMP (*.bmp)",
        )
        if not save_path:
            return

        self._last_export_dir = os.path.dirname(save_path)
        pixmap.save(save_path)

    def _on_show_loss(self):
        self._run_strategy_compare()

    def _current_compare_key(self):
        return (self._dataset_name, self.strategy_a_combo.currentText(), self.strategy_b_combo.currentText(), self._manual_model_paths.get("A"), self._manual_model_paths.get("B"))

    def _set_compare_placeholders(self):
        for label, name in [(self.gt_label, "GT"), (self.pd_a_label, "PD-A"), (self.pd_b_label, "PD-B")]:
            label.clear()
            label.setText(name)

    def _append_compare_log(self, text):
        if not text:
            return
        self.run_log.appendPlainText(text)
        bar = self.run_log.verticalScrollBar()
        bar.setValue(bar.maximum())
        QApplication.processEvents()

    def _short_path_with_parents(self, path, parent_count=3):
        if not path:
            return "(未找到)"
        norm = os.path.normpath(path)
        parts = norm.split(os.sep)
        need = parent_count + 1
        if len(parts) <= need:
            return norm
        return "..." + os.sep + os.sep.join(parts[-need:])

    def _refresh_compare_path_hints(self):
        dataset_name = self._dataset_name
        dataset_dir = self._dataset_dir(dataset_name)
        self.dataset_path_hint.setText(f"数据路径: {self._short_path_with_parents(dataset_dir, parent_count=3)}")

        path_a = self._manual_model_paths.get("A") or self._find_latest_model_path(dataset_name, self.strategy_a_combo.currentText())
        path_b = self._manual_model_paths.get("B") or self._find_latest_model_path(dataset_name, self.strategy_b_combo.currentText())
        suffix_a = "(手动)" if self._manual_model_paths.get("A") else ""
        suffix_b = "(手动)" if self._manual_model_paths.get("B") else ""
        self.strategy_a_path_hint.setText(f"策略A模型{suffix_a}: {self._short_path_with_parents(path_a, parent_count=3)}")
        self.strategy_b_path_hint.setText(f"策略B模型{suffix_b}: {self._short_path_with_parents(path_b, parent_count=3)}")

    def _choose_model_file(self, slot):
        start_dir = os.path.join(self._project_root, "results", f"{self._dataset_name}Results")
        if not os.path.isdir(start_dir):
            start_dir = self._project_root
        model_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择模型文件",
            start_dir,
            "Model (*.pkl)",
        )
        if not model_path:
            return
        self._manual_model_paths[slot] = model_path
        self._refresh_compare_path_hints()
        self._set_compare_placeholders()
        self.compare_hint.setText(f"已设置策略{slot}手动模型，点击“运行对比”开始实验")

    def _on_compare_selection_changed(self, *_):
        self._refresh_compare_path_hints()
        if self._update_compare_images(focus=None, allow_compute=False):
            self.compare_hint.setText("已加载缓存结果（未重新实验）")
            self.show_gt()
        else:
            self._set_compare_placeholders()
            self.compare_hint.setText("策略已更新，点击“运行对比”开始实验")

    def _dataset_cfg(self, dataset_name):
        if dataset_name in ["SEGSalt", "SEGSimulation"]:
            return {"data_dim": [400, 301], "model_dim": [201, 301], "default_select_id": 1615}
        return {"data_dim": [1000, 70], "model_dim": [70, 70], "default_select_id": [1, 2]}

    def _dataset_dir(self, dataset_name):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        return os.path.join(root, "data", dataset_name) + os.sep

    def _load_compare_sample(self, dataset_name, train_or_test="test"):
        cfg = self._dataset_cfg(dataset_name)
        select_id = cfg["default_select_id"]
        data_dir = self._dataset_dir(dataset_name)

        if dataset_name in ["SEGSalt", "SEGSimulation"]:
            seismic_data, velocity_model, _ = single_read_matfile(
                data_dir,
                cfg["data_dim"],
                cfg["model_dim"],
                select_id,
                train_or_test=train_or_test,
            )
            max_velocity, min_velocity = np.max(velocity_model), np.min(velocity_model)
            velocity_model_norm = np.asarray(velocity_model, dtype=np.float32)
        else:
            seismic_data, velocity_model, _ = single_read_npyfile(
                data_dir,
                select_id,
                train_or_test=train_or_test,
            )
            max_velocity, min_velocity = np.max(velocity_model), np.min(velocity_model)
            velocity_model_norm = (velocity_model - np.min(velocity_model)) / (np.max(velocity_model) - np.min(velocity_model) + 1e-8)

        velocity_model_norm = np.asarray(velocity_model_norm)
        if velocity_model_norm.size == 0:
            raise ValueError(f"读取文件内容为空: dataset={dataset_name}, mode={train_or_test}")

        return np.asarray(seismic_data, dtype=np.float32), velocity_model_norm.astype(np.float32), float(min_velocity), float(max_velocity)

    def _method_root_dir(self, dataset_name, method_name):
        base = os.path.join(self._project_root, "results", f"{dataset_name}Results")
        if dataset_name != "CurveFaultA":
            return None
        mapping = {
            "Scratch": os.path.join(base, "unused_pretrain", "DDNet70", "CurveFaultA"),
            "Sup-Mig": os.path.join(base, "used_pretrain", "DDNet70", "CurveFaultA"),
            "Self-Sup": os.path.join(base, "self_sup", "DDNet70", "CurveFaultA"),
        }
        return mapping.get(method_name)

    def _find_latest_model_path(self, dataset_name, method_name):
        root = self._method_root_dir(dataset_name, method_name)
        if not root or not os.path.isdir(root):
            return None

        matches = []
        for cur_root, _, files in os.walk(root):
            for name in files:
                if name.lower().endswith(".pkl"):
                    full = os.path.join(cur_root, name)
                    matches.append((os.path.getmtime(full), full))
        if not matches:
            return None
        matches.sort(key=lambda x: x[0], reverse=True)
        return matches[0][1]

    def _predict_velocity_by_method(self, dataset_name, seismic_data, method_name, manual_path=None):
        model_path = manual_path if (manual_path and os.path.exists(manual_path)) else self._find_latest_model_path(dataset_name, method_name)
        if not model_path:
            raise FileNotFoundError(f"未找到模型文件: dataset={dataset_name}, method={method_name}")

        cache_key = (model_path, int(os.path.getmtime(model_path)))
        if cache_key not in self._model_runtime_cache:
            model_net, device, _ = determine_network(model_path, model_type="DDNet70")
            self._model_runtime_cache[cache_key] = (model_net, device)

        model_net, device = self._model_runtime_cache[cache_key]
        cfg = self._dataset_cfg(dataset_name)
        seismic_tensor = torch.from_numpy(np.array([seismic_data])).float().to(device, non_blocking=device.type == "cuda")
        model_net.eval()
        with torch.no_grad():
            outputs, _ = model_net(seismic_tensor, cfg["model_dim"])
        predicted_vmod = outputs.cpu().detach().numpy()[0][0]
        predicted_vmod = np.where(predicted_vmod > 0.0, predicted_vmod, 0.0)
        return predicted_vmod.astype(np.float32), model_path

    def _build_openfwi_model_pixmap(self, model, min_v, max_v, title):
        from PyQt5.QtGui import QImage

        width, height = 580, 620

        # 先根据字体度量动态计算边距，避免不同机器字体/DPI下坐标文字被裁切。
        tick_font = QFont()
        tick_font.setPointSize(10)
        tick_fm = QFontMetrics(tick_font)

        axis_font = QFont()
        axis_font.setPointSize(11)
        axis_fm = QFontMetrics(axis_font)

        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title_fm = QFontMetrics(title_font)

        ticks = np.linspace(0.0, 0.7, 8)
        tick_texts = [f"{t:.1f}" for t in ticks]
        max_tick_w = max(tick_fm.horizontalAdvance(txt) for txt in tick_texts)
        tick_h = tick_fm.height()

        # 左边距 = 左留白 + y轴标题区 + 间距 + y刻度文本区 + 刻度线与间距
        y_title_band = axis_fm.height() + 16
        gap_ytitle_to_ticks = 10
        y_tick_band = max_tick_w + 14
        left_margin = max(92, 10 + y_title_band + gap_ytitle_to_ticks + y_tick_band + 12)

        # 底边距 = x刻度文本区 + 与轴标题间距 + x轴标题区 + 底部留白
        x_axis_h = axis_fm.height() + 10
        tick_block_h = tick_h + 4
        gap_tick_to_xlabel = 8
        bottom_padding = 10
        bottom_margin = max(122, 8 + tick_block_h + gap_tick_to_xlabel + x_axis_h + bottom_padding)

        title_y = 14
        title_h = title_fm.height() + 10
        title_to_cbar_gap = 12
        cbar_tick_len = 5
        cbar_tick_text_h = tick_h + 4
        top_margin = max(82, title_y + title_h + title_to_cbar_gap + cbar_tick_len + cbar_tick_text_h)
        right_margin = 36

        # 参考 pain_openfwi_velocity_model：顶部水平色标（size=3%, pad≈0.35）。
        cbar_pad = 14

        plot_w = width - left_margin - right_margin
        usable_h = height - top_margin - bottom_margin - cbar_pad
        plot_h = max(50, int(usable_h / 1.03))
        cbar_h = max(8, usable_h - plot_h)

        image = QImage(width, height, QImage.Format_RGB32)
        image.fill(Qt.white)
        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing)

        data = np.asarray(model, dtype=np.float32)
        if data.shape != (70, 70):
            y_idx = np.linspace(0, max(data.shape[0] - 1, 0), 70).astype(int)
            x_idx = np.linspace(0, max(data.shape[1] - 1, 0), 70).astype(int)
            data = data[y_idx][:, x_idx]

        rng = max(max_v - min_v, 1e-8)
        norm = np.clip((data - min_v) / rng, 0.0, 1.0)
        cmap = mpl_cm.get_cmap("viridis")
        rgb = (cmap(norm)[..., :3] * 255).astype(np.uint8)

        cbar_norm = np.linspace(0.0, 1.0, 512, dtype=np.float32).reshape(1, 512)
        cbar_rgb = (cmap(cbar_norm)[..., :3] * 255).astype(np.uint8)

        cbar_x = left_margin
        cbar_y = top_margin
        plot_y = cbar_y + cbar_h + cbar_pad

        plot_img = QImage(rgb.data, 70, 70, 70 * 3, QImage.Format_RGB888).copy()
        painter.drawImage(
            left_margin,
            plot_y,
            plot_img.scaled(plot_w, plot_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation),
        )

        cbar_img = QImage(cbar_rgb.data, 512, 1, 512 * 3, QImage.Format_RGB888).copy()
        painter.drawImage(
            cbar_x,
            cbar_y,
            cbar_img.scaled(plot_w, cbar_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation),
        )

        painter.setPen(QPen(QColor(68, 68, 68), 2))
        painter.drawRect(left_margin, plot_y, plot_w, plot_h)
        painter.drawRect(cbar_x, cbar_y, plot_w, cbar_h)

        painter.setFont(tick_font)
        painter.setPen(QColor(45, 45, 45))

        for t in ticks:
            ratio = t / 0.7
            x = left_margin + int(ratio * plot_w)
            y = plot_y + int(ratio * plot_h)
            tick_txt = f"{t:.1f}"
            painter.drawLine(x, plot_y + plot_h, x, plot_y + plot_h + 5)

            # x轴刻度文本宽度按字体实时计算，避免固定 32 像素导致截断。
            x_text_w = tick_fm.horizontalAdvance(tick_txt) + 8
            x_text_h = tick_block_h
            x_text_y = plot_y + plot_h + 8
            painter.drawText(x - x_text_w // 2, x_text_y, x_text_w, x_text_h, Qt.AlignCenter, tick_txt)

            painter.drawLine(left_margin - 5, y, left_margin, y)
            y_text_w = left_margin - y_title_band - gap_ytitle_to_ticks - 12
            y_text_h = tick_h + 4
            y_text_x = y_title_band + gap_ytitle_to_ticks
            painter.drawText(y_text_x, y - y_text_h // 2, y_text_w, y_text_h, Qt.AlignRight | Qt.AlignVCenter, tick_txt)

        cbar_ticks = np.linspace(min_v, max_v, 7)
        for i, val in enumerate(cbar_ticks):
            ratio = i / 6.0
            x = cbar_x + int(ratio * (plot_w - 1))
            painter.drawLine(x, cbar_y - 5, x, cbar_y)
            painter.drawText(
                x - 26,
                cbar_y - cbar_tick_text_h - 8,
                52,
                cbar_tick_text_h,
                Qt.AlignCenter,
                f"{val:.0f}",
            )

        painter.setFont(title_font)
        painter.drawText(left_margin, title_y, plot_w, title_h, Qt.AlignCenter, title)

        painter.setFont(axis_font)
        x_tick_text_top = plot_y + plot_h + 8
        x_tick_text_bottom = x_tick_text_top + tick_block_h
        x_label_y = x_tick_text_bottom + gap_tick_to_xlabel
        painter.drawText(left_margin, x_label_y, plot_w, x_axis_h, Qt.AlignCenter, "宽度(千米)")
        painter.save()
        y_title_center_x = max(10, y_title_band // 2)
        painter.translate(y_title_center_x, plot_y + plot_h // 2)
        painter.rotate(-90)
        painter.drawText(-plot_h // 2, 0, plot_h, axis_fm.height() + 8, Qt.AlignCenter, "深度(千米)")
        painter.restore()

        painter.end()
        return QPixmap.fromImage(image)

    def _render_velocity_to_label(self, label, model, min_v, max_v, title):
        pix = self._build_openfwi_model_pixmap(model, min_v, max_v, title)
        self._image_pixmaps[label] = pix
        label.setPixmap(
            pix.scaled(
                label.width(),
                label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def _load_compare_data(self, dataset_name, force_refresh=False):
        strategy_a = self.strategy_a_combo.currentText()
        strategy_b = self.strategy_b_combo.currentText()
        cache_key = self._current_compare_key()
        if (not force_refresh) and cache_key in self._cached_compare:
            return self._cached_compare[cache_key]

        try:
            seismic_data, gt_vm_norm, min_velocity, max_velocity = self._load_compare_sample(dataset_name, train_or_test="test")

            if dataset_name == "CurveFaultA":
                pd_a_norm, path_a = self._predict_velocity_by_method(dataset_name, seismic_data, strategy_a, manual_path=self._manual_model_paths.get("A"))
                pd_b_norm, path_b = self._predict_velocity_by_method(dataset_name, seismic_data, strategy_b, manual_path=self._manual_model_paths.get("B"))
                warning = f"A={os.path.basename(path_a)} | B={os.path.basename(path_b)}"
                self._append_compare_log(f"[Model-A] {self._short_path_with_parents(path_a, parent_count=3)}")
                self._append_compare_log(f"[Model-B] {self._short_path_with_parents(path_b, parent_count=3)}")
            else:
                pd_a_norm = np.copy(gt_vm_norm)
                pd_b_norm = np.copy(gt_vm_norm)
                warning = "当前仅CurveFaultA启用策略权重推理，已回退为GT展示"
                self._append_compare_log("当前数据集未启用策略推理，已回退为GT展示")

            if dataset_name in ["SEGSalt", "SEGSimulation"]:
                gt_show = gt_vm_norm
                pd_a_show = pd_a_norm
                pd_b_show = pd_b_norm
            else:
                gt_show = min_velocity + gt_vm_norm * (max_velocity - min_velocity)
                pd_a_show = min_velocity + pd_a_norm * (max_velocity - min_velocity)
                pd_b_show = min_velocity + pd_b_norm * (max_velocity - min_velocity)

            min_v = float(np.min(gt_show))
            max_v = float(np.max(gt_show))
        except Exception:
            rng = np.random.default_rng(abs(hash((dataset_name, strategy_a, strategy_b))) % 100000)
            gt_vm = np.clip(rng.normal(0.55, 0.2, (70, 70)), 0.0, 1.0).astype(np.float32)
            pd_a_show = np.clip(gt_vm + rng.normal(0.0, 0.06, (70, 70)), 0.0, 1.0).astype(np.float32)
            pd_b_show = np.clip(gt_vm + rng.normal(0.0, 0.08, (70, 70)), 0.0, 1.0).astype(np.float32)
            gt_show = gt_vm
            min_v = float(np.min(gt_show))
            max_v = float(np.max(gt_show))
            warning = "推理失败，已回退到模拟数据"

        payload = {
            "gt": gt_show,
            "pd_a": pd_a_show,
            "pd_b": pd_b_show,
            "min_v": min_v,
            "max_v": max_v,
            "warning": warning,
            "strategy_a": strategy_a,
            "strategy_b": strategy_b,
        }
        self._cached_compare[cache_key] = payload
        return payload



    def _update_compare_images(self, focus=None, force_refresh=False, allow_compute=False):
        dataset_name = self._dataset_name
        if allow_compute:
            payload = self._load_compare_data(dataset_name, force_refresh=force_refresh)
        else:
            payload = self._cached_compare.get(self._current_compare_key())
            if payload is None:
                return False

        if focus in (None, "gt"):
            self._render_velocity_to_label(
                self.gt_label,
                payload["gt"],
                payload["min_v"],
                payload["max_v"],
                "GT",
            )
        if focus in (None, "pd_a"):
            self._render_velocity_to_label(
                self.pd_a_label,
                payload["pd_a"],
                payload["min_v"],
                payload["max_v"],
                f"PD-A ({payload.get('strategy_a', 'A')})",  # 去掉数据集名
            )
        if focus in (None, "pd_b"):
            self._render_velocity_to_label(
                self.pd_b_label,
                payload["pd_b"],
                payload["min_v"],
                payload["max_v"],
                f"PD-B ({payload.get('strategy_b', 'B')})",
            )
        return True

    def _run_strategy_compare(self):
        try:
            self.run_log.clear()
            self._append_compare_log("开始运行策略对比...")
            self._append_compare_log(f"Dataset: {self._dataset_name}")
            self._append_compare_log(f"Strategy-A: {self.strategy_a_combo.currentText()}")
            self._append_compare_log(f"Strategy-B: {self.strategy_b_combo.currentText()}")
            self._update_compare_images(focus=None, force_refresh=True, allow_compute=True)
            payload = self._cached_compare.get(self._current_compare_key())
            if payload:
                self.compare_hint.setText(f"实验完成：{payload.get('warning', '')}")
                self._append_compare_log("策略对比完成")
            self.show_gt()
        except Exception as e:
            self.compare_hint.setText(f"实验失败：{e}")
            self._append_compare_log(f"ERROR: {e}")
            QMessageBox.warning(self, "对比失败", str(e))

    def _load_placeholder_image(self):
        base = os.path.dirname(__file__)
        path = os.path.join(base, 'resources', 'system_reminder.png')
        if os.path.exists(path):
            pix = QPixmap(path)
            if not pix.isNull():
                return pix
        pix = QPixmap(320, 240)
        pix.fill(Qt.white)
        painter = QPainter(pix)
        painter.setPen(Qt.black)
        font = QFont()
        font.setPointSize(14)
        painter.setFont(font)
        painter.drawText(pix.rect(), Qt.AlignCenter, "<system-reminder>")
        painter.end()
        return pix
