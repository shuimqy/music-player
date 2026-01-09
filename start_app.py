import sys
import time
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QLabel,
    QProgressBar,
    QGraphicsDropShadowEffect,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor

# 尝试导入主程序
# 注意：确保你的主程序文件名为 main.py，且类名为 MusicPlayer
try:
    from main import MusicPlayer
except ImportError:
    print("错误：未找到 main.py 或 MusicPlayer 类。请确保文件在同一目录下。")
    sys.exit(1)


class SplashScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("系统加载中...")
        # 设置无边框窗口 + 顶层显示
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)  # 允许背景透明（用于圆角阴影）

        self.init_ui()
        self.resize(500, 320)
        self.center()

        # 模拟加载进度的计时器
        self.counter = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.progress)
        self.timer.start(35)  # 每35毫秒更新一次进度

    def init_ui(self):
        # 外部容器（用于绘制圆角和背景）
        self.container = QWidget(self)
        self.container.setObjectName("Container")
        self.container.setGeometry(10, 10, 480, 300)  # 留出边距给阴影

        # 布局
        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(40, 40, 40, 40)

        # 1. 主标题 (中文)
        self.title_label = QLabel("手势音乐播放器")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setObjectName("Title")

        # 2. 副标题 (系统版本)
        self.subtitle_label = QLabel("系统版本 V2.0")
        self.subtitle_label.setAlignment(Qt.AlignCenter)
        self.subtitle_label.setObjectName("Subtitle")

        # 3. 加载状态文字 (中文)
        self.loading_label = QLabel("正在初始化核心系统...")
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.loading_label.setObjectName("LoadingText")

        # 4. 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("ProgressBar")
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        layout.addWidget(self.title_label)
        layout.addWidget(self.subtitle_label)
        layout.addStretch()
        layout.addWidget(self.loading_label)
        layout.addSpacing(10)
        layout.addWidget(self.progress_bar)

        # --- 样式表 (CSS) ---
        # 这里特别指定了 "Microsoft YaHei" (微软雅黑) 以确保中文好看
        self.setStyleSheet("""
            QWidget#Container {
                background-color: #1e1e1e;
                border: 1px solid #333;
                border-radius: 15px;
            }
            QLabel#Title {
                color: #ffffff;
                font-family: "Microsoft YaHei", "Segoe UI"; 
                font-size: 28px;
                font-weight: bold;
                letter-spacing: 2px;
            }
            QLabel#Subtitle {
                color: #00bcd4; /* 青色 */
                font-family: "Microsoft YaHei", "Segoe UI";
                font-size: 14px;
                font-weight: bold;
                letter-spacing: 4px;
                margin-bottom: 20px;
            }
            QLabel#LoadingText {
                color: #888888;
                font-family: "Microsoft YaHei", "SimHei"; 
                font-size: 13px;
            }
            QProgressBar {
                background-color: #2d2d2d;
                border-radius: 3px;
                border: none;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 #00bcd4, stop:1 #00e5ff);
                border-radius: 3px;
            }
        """)

        # 添加阴影效果
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setXOffset(0)
        shadow.setYOffset(0)
        shadow.setColor(QColor(0, 0, 0, 150))
        self.container.setGraphicsEffect(shadow)

    def center(self):
        # 居中显示
        screen = QApplication.primaryScreen().geometry()
        size = self.geometry()
        self.move(
            (screen.width() - size.width()) // 2, (screen.height() - size.height()) // 2
        )

    def progress(self):
        # 模拟加载进度的逻辑
        self.counter += 1
        self.progress_bar.setValue(self.counter)

        # --- 汉化后的加载提示 ---
        if self.counter == 10:
            self.loading_label.setText("正在加载 MediaPipe 视觉框架...")
        elif self.counter == 30:
            self.loading_label.setText("正在初始化摄像头传感器...")
        elif self.counter == 50:
            self.loading_label.setText("正在校准手势识别算法...")
        elif self.counter == 70:
            self.loading_label.setText("正在挂载音频播放引擎...")
        elif self.counter == 90:
            self.loading_label.setText("正在启动用户界面...")

        # 加载完成
        if self.counter >= 100:
            self.timer.stop()
            self.launch_main_app()

    def launch_main_app(self):
        # 关闭启动页
        self.close()
        # 初始化并显示主窗口
        self.main_window = MusicPlayer()
        self.main_window.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 创建并显示启动页
    splash = SplashScreen()
    splash.show()

    sys.exit(app.exec())
