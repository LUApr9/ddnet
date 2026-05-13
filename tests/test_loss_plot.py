import numpy as np
from modules.system_reminder.loss_plot import build_plot_data


def test_build_plot_data_basic():
    train = [1.0, 2.0, 1.5, 1.2, 0.8]
    data = build_plot_data(train, smooth_window=None, w=100, h=50, margin=5)
    assert "train_points" in data
    assert isinstance(data["train_points"], list)
    assert len(data["train_points"]) == len(train)
    assert data["val_points"] is None


def test_with_val_and_smoothing():
    train = list(np.linspace(0, 1, 10))
    val = list(np.linspace(1, 0, 10))
    data = build_plot_data(train, val, smooth_window=2, w=120, h=60, margin=6)
    assert len(data["train_points"]) == len(train)
    assert len(data["val_points"]) == len(val)


test_build_plot_data_basic()
