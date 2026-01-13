import sys
import json
import os
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QGroupBox,
    QFileDialog,
    QMessageBox,
    QLineEdit,
)
from PySide6.QtCore import Signal, Qt

# 默认配置
DEFAULT_CONFIG = {
    "music_folder": "",
    "gestures": {
        "拳头": "play",  # 播放
        "掌心": "pause",  # 暂停
        "V字手势": "next",  # 下一首
        "音量加": "vol_up",  # 音量+
        "音量减": "vol_down",  # 音量-
    },
}

CONFIG_FILE = "config.json"


class ConfigManager:
    """管理配置文件的读取和写入"""

    @staticmethod
    def load_config():
        if not os.path.exists(CONFIG_FILE):
            ConfigManager.save_config(DEFAULT_CONFIG)
            return DEFAULT_CONFIG
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                config = json.load(f)
                # 确保所有键都存在（合并默认值，防止旧配置文件缺项）
                for key, value in DEFAULT_CONFIG.items():
                    if key not in config:
                        config[key] = value
                return config
        except Exception as e:
            print(f"配置文件加载失败: {e}")
            return DEFAULT_CONFIG

    @staticmethod
    def save_config(config):
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"配置文件保存失败: {e}")


class SettingsWindow(QWidget):
    # 信号：配置已保存（通知主窗口刷新）
    config_saved = Signal(dict)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("系统设置")
        self.resize(500, 600)
        self.config = ConfigManager.load_config()
        self.init_ui()
        self.apply_styles()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)

        # 1. 音乐文件夹设置
        folder_group = QGroupBox(" 媒体资源路径 ")
        folder_layout = QVBoxLayout(folder_group)

        path_layout = QHBoxLayout()
        self.path_input = QLineEdit()
        self.path_input.setReadOnly(True)
        self.path_input.setText(self.config.get("music_folder", ""))
        self.path_input.setPlaceholderText("未设置文件夹...")

        self.browse_btn = QPushButton("浏览...")
        self.browse_btn.setFixedWidth(80)
        self.browse_btn.clicked.connect(self.browse_folder)

        path_layout.addWidget(self.path_input)
        path_layout.addWidget(self.browse_btn)
        folder_layout.addLayout(path_layout)
        layout.addWidget(folder_group)

        # 2. 手势映射设置
        gesture_group = QGroupBox(" 手势映射 ")
        self.gesture_layout = QVBoxLayout(gesture_group)
        self.gesture_layout.setSpacing(15)

        # 动作选项
        self.actions = {
            "播放": "play",
            "暂停": "pause",
            "下一首": "next",
            "音量 +": "vol_up",
            "音量 -": "vol_down",
            "无操作": "none",
        }
        # 反向映射用于回显
        self.actions_rev = {v: k for k, v in self.actions.items()}

        # 手势列表
        self.gesture_names = ["拳头", "掌心", "V字手势", "音量加", "音量减"]
        self.combos = {}

        for gesture in self.gesture_names:
            row = QHBoxLayout()
            label = QLabel(f"手势 [{gesture}]:")
            label.setFixedWidth(120)

            combo = QComboBox()
            combo.addItems(self.actions.keys())

            # 设置当前选中的值
            current_action_code = self.config["gestures"].get(gesture, "none")
            current_action_name = self.actions_rev.get(current_action_code, "无操作")
            combo.setCurrentText(current_action_name)

            self.combos[gesture] = combo

            row.addWidget(label)
            row.addWidget(combo)
            self.gesture_layout.addLayout(row)

        layout.addWidget(gesture_group)
        layout.addStretch()

        # 3. 底部按钮
        btn_layout = QHBoxLayout()
        self.save_btn = QPushButton("保存配置")
        self.save_btn.setFixedHeight(40)
        self.save_btn.clicked.connect(self.save_settings)

        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setFixedHeight(40)
        self.cancel_btn.clicked.connect(self.close)

        btn_layout.addWidget(self.save_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "选择音乐文件夹", self.path_input.text()
        )
        if folder:
            self.path_input.setText(folder)

    def save_settings(self):
        # 1. 保存路径
        new_config = {"music_folder": self.path_input.text(), "gestures": {}}

        # 2. 保存手势映射
        for gesture, combo in self.combos.items():
            action_name = combo.currentText()
            action_code = self.actions[action_name]
            new_config["gestures"][gesture] = action_code

        # 3. 写入文件
        ConfigManager.save_config(new_config)
        self.config = new_config

        # 4. 发送信号
        self.config_saved.emit(new_config)
        QMessageBox.information(self, "系统提示", "配置已成功更新并保存。")
        self.close()

    def apply_styles(self):
        # 复用主界面的 Dark Tech 风格
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                color: #f0f0f0;
                font-family: "Microsoft YaHei", "Segoe UI";
            }
            QGroupBox {
                border: 2px solid #333;
                border-radius: 8px;
                margin-top: 20px;
                font-weight: bold;
                color: #00bcd4;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QLineEdit {
                background-color: #252526;
                border: 1px solid #444;
                color: #ccc;
                padding: 5px;
                border-radius: 4px;
            }
            QComboBox {
                background-color: #333;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 5px;
                color: white;
            }
            QComboBox::drop-down {
                border: none;
            }
            QPushButton {
                background-color: #333333;
                border: 1px solid #444;
                border-radius: 6px;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #444444;
                border-color: #00bcd4;
            }
            QPushButton:pressed {
                background-color: #00bcd4;
                color: black;
            }
        """)


if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    win = SettingsWindow()
    win.show()
    sys.exit(app.exec())
