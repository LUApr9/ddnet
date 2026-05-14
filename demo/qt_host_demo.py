"""
Minimal demo host app to showcase embedding SystemReminderContainer.
Modified for PyInstaller compatibility and low-memory environments.
Run with: python ddnet-main/demo/qt_host_demo.py
"""
import sys
import os
import faulthandler
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QTimer

# ========================================================
# 路径兼容性修改：解决 EXE 运行和源码运行的导入问题
# ========================================================
if getattr(sys, 'frozen', False):
    # 如果是打包后的 EXE 运行，获取 EXE 所在的目录
    base_dir = os.path.dirname(sys.executable)
else:
    # 如果是源码运行，获取当前 py 文件目录
    base_dir = os.path.dirname(os.path.abspath(__file__))

# 定位项目根目录 (即 ddnet-main 所在位置)
# 如果在根目录运行，project_root 就是 base_dir
# 如果在 demo 文件夹运行，project_root 是 base_dir 的上一级
project_root = base_dir if "demo" not in os.path.basename(base_dir) else os.path.dirname(base_dir)

# 将项目根目录加入 sys.path，确保可以 import modules
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ========================================================

try:
    # 此时可以正常导入，无论是在根目录执行还是在 demo 下执行
    from modules.system_reminder import SystemReminderContainer
except Exception as e:
    print(f"导入失败: {e}")
    print(f"当前搜索路径: {sys.path}")
    raise

def main():
    # 启用故障处理程序，方便在小内存环境下诊断由于资源耗尽导致的崩溃
    faulthandler.enable()

    app = QApplication(sys.argv)
    window = QMainWindow()

    # 设置导师要求的窗口标题
    window.setWindowTitle("22计科3班-路凯乐-20220344")

    container = SystemReminderContainer()
    window.setCentralWidget(container)

    # 注入上下文信息
    container.set_context({"host": "demo", "theme": "light"})
    # 初始化内容
    container.update_content("融合预训练策略的地震反演系统")

    window.resize(1000, 800)
    window.show()

    def update_later():
        container.update_content("更新：系统正在准备切换到新对比模型，界面将保持当前布局。")

    # 延迟 3 秒更新状态
    QTimer.singleShot(3000, update_later)

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()