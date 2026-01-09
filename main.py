from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QListWidget,
    QProgressBar,
    QFrame,
    QStyle,
    QGroupBox,
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, QThread, Signal
import sys
import cv2
import pygame
import numpy as np
import os
import time
import mediapipe as mp
import math

# 导入设置模块
from settings import SettingsWindow, ConfigManager

# --- 样式表 (Dark Tech Theme) ---
STYLESHEET = """
QMainWindow {
    background-color: #1e1e1e;
    color: #f0f0f0;
}
QGroupBox {
    border: 2px solid #333;
    border-radius: 8px;
    margin-top: 10px;
    font-weight: bold;
    color: #00bcd4; 
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
}
QLabel {
    color: #e0e0e0;
    font-family: "Segoe UI", sans-serif;
    font-size: 14px;
}
QListWidget {
    background-color: #252526;
    border: 1px solid #333;
    border-radius: 5px;
    color: #cccccc;
    font-size: 13px;
    padding: 5px;
}
QListWidget::item {
    height: 30px;
    padding: 5px;
}
QListWidget::item:selected {
    background-color: #00bcd4;
    color: #000000;
    border-radius: 3px;
}
QListWidget::item:hover {
    background-color: #333;
}
QPushButton {
    background-color: #333333;
    border: 1px solid #444;
    border-radius: 6px;
    color: white;
    padding: 8px 15px;
    font-weight: bold;
    min-height: 20px;
}
QPushButton:hover {
    background-color: #444444;
    border-color: #00bcd4;
}
QPushButton:pressed {
    background-color: #00bcd4;
    color: black;
}
QProgressBar {
    border: 1px solid #444;
    border-radius: 5px;
    background-color: #252526;
    text-align: center;
    color: white;
    font-weight: bold;
}
QProgressBar::chunk {
    background-color: #00bcd4;
    border-radius: 4px;
}
"""


class GestureRecognitionThread(QThread):
    image_signal = Signal(np.ndarray, np.ndarray)
    gesture_signal = Signal(str)
    error_signal = Signal(str)

    def __init__(self):
        super().__init__()
        self.cap = None
        self.running = True
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self.prev_gesture = None
        self.gesture_count = 0
        self.gesture_cooldown = 0

    def run(self):
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.error_signal.emit("无法打开摄像头")
                return

            self.error_signal.emit("系统就绪 - 等待手势")
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    break

                if self.gesture_cooldown > 0:
                    self.gesture_cooldown -= 1

                processed_frame, gesture = self.process_and_recognize(frame)
                self.image_signal.emit(frame, processed_frame)

                if self.gesture_cooldown == 0:
                    if gesture:
                        if gesture == self.prev_gesture:
                            self.gesture_count += 1
                        else:
                            self.prev_gesture = gesture
                            self.gesture_count = 1

                        required_frames = 3
                        # 为快速反应的手势降低帧数要求
                        if gesture == "V字手势" or gesture in ["音量加", "音量减"]:
                            required_frames = 2

                        if self.gesture_count >= required_frames:
                            self.gesture_signal.emit(gesture)

                            # 独立冷却时间
                            if gesture == "V字手势":
                                self.gesture_cooldown = 20
                            elif gesture == "拳头":
                                self.gesture_cooldown = 10
                            elif gesture in ["音量加", "音量减"]:
                                self.gesture_cooldown = 5
                            else:
                                self.gesture_cooldown = 12

                            self.gesture_count = 0
                    else:
                        self.prev_gesture = None
                        self.gesture_count = 0

                time.sleep(0.04)

        except Exception as e:
            self.error_signal.emit(f"线程错误: {str(e)}")
        finally:
            if self.cap:
                self.cap.release()
            self.hands.close()

    def process_and_recognize(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        processed_frame = frame.copy()

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    processed_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
                gesture = self.recognize_gesture(hand_landmarks.landmark)
                return processed_frame, gesture

        return processed_frame, None

    def calculate_distance(self, p1, p2):
        return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

    def recognize_gesture(self, landmarks):
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]

        thumb_ip = landmarks[3].y
        index_pip = landmarks[6].y
        middle_pip = landmarks[10].y
        ring_pip = landmarks[14].y
        pinky_pip = landmarks[18].y

        thumb_up = thumb_tip.y < thumb_ip - 0.015
        index_up = index_tip.y < index_pip - 0.015
        middle_up = middle_tip.y < middle_pip - 0.015
        ring_up = ring_tip.y < ring_pip - 0.015
        pinky_up = pinky_tip.y < pinky_pip - 0.015

        thumb_bend = not thumb_up
        index_bend = not index_up
        middle_bend = not middle_up
        ring_bend = not ring_up
        pinky_bend = not pinky_up

        pinch_distance = self.calculate_distance(thumb_tip, index_tip)
        if pinch_distance < 0.05 and middle_up and ring_up and pinky_up:
            return "音量加"

        if thumb_up and pinky_up and index_bend and middle_bend and ring_bend:
            return "音量减"

        if thumb_bend and index_bend and middle_bend and ring_bend and pinky_bend:
            return "拳头"

        if thumb_up and index_up and middle_up and ring_up and pinky_up:
            if pinch_distance > 0.05:
                return "掌心"

        if index_up and middle_up and thumb_bend and ring_bend and pinky_bend:
            return "V字手势"

        return None

    def stop(self):
        self.running = False


class MusicPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setStyleSheet(STYLESHEET)

        # 加载配置
        self.config = ConfigManager.load_config()
        self.settings_window = None  # 延迟初始化

        self.init_ui()
        self.init_music()
        self.init_gesture_thread()
        self.last_handled_gesture = None

        # 启动时根据配置加载音乐
        self.load_music_from_config()

    def init_ui(self):
        self.setWindowTitle("智能手势音乐播放器 V2.1")
        self.resize(1100, 750)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # --- 左侧区域 ---
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # 视频区域
        video_group = QGroupBox(" 手势识别窗口 ")
        video_layout = QHBoxLayout(video_group)
        self.original_video = QLabel()
        self.original_video.setAlignment(Qt.AlignCenter)
        self.original_video.setStyleSheet(
            "background-color: black; border-radius: 4px;"
        )
        self.original_video.setMinimumSize(320, 240)
        self.processed_video = QLabel()
        self.processed_video.setAlignment(Qt.AlignCenter)
        self.processed_video.setStyleSheet(
            "background-color: black; border-radius: 4px;"
        )
        self.processed_video.setMinimumSize(320, 240)
        video_layout.addWidget(self.original_video)
        video_layout.addWidget(self.processed_video)
        left_layout.addWidget(video_group, stretch=3)

        # 状态区域
        info_group = QGroupBox(" 系统状态 ")
        info_layout = QVBoxLayout(info_group)
        self.status_label = QLabel("正在初始化系统...")
        self.status_label.setStyleSheet(
            "font-size: 16px; color: #00bcd4; font-weight: bold;"
        )
        self.gesture_label = QLabel("当前手势: 无")
        self.gesture_label.setStyleSheet("font-size: 18px; color: white;")
        info_layout.addWidget(self.status_label)
        info_layout.addWidget(self.gesture_label)
        left_layout.addWidget(info_group, stretch=1)

        # 控制栏
        control_group = QFrame()
        control_group.setStyleSheet("background-color: #2b2b2b; border-radius: 10px;")
        control_layout = QHBoxLayout(control_group)

        # 按钮：设置 (替代了原来的加载文件夹)
        self.settings_btn = QPushButton("系统设置")
        self.settings_btn.setIcon(self.style().standardIcon(QStyle.SP_ComputerIcon))
        self.settings_btn.setToolTip("打开设置面板")
        self.settings_btn.clicked.connect(self.open_settings)

        self.play_btn = QPushButton()
        self.play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_btn.clicked.connect(self.play_music)

        self.pause_btn = QPushButton()
        self.pause_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        self.pause_btn.clicked.connect(self.pause_music)

        self.next_btn = QPushButton()
        self.next_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaSkipForward))
        self.next_btn.clicked.connect(self.next_music)

        control_layout.addWidget(self.settings_btn)  # 新的设置按钮
        control_layout.addSpacing(20)
        control_layout.addWidget(self.play_btn)
        control_layout.addWidget(self.pause_btn)
        control_layout.addWidget(self.next_btn)

        control_layout.addSpacing(30)
        control_layout.addWidget(QLabel("音量"))
        self.volume_bar = QProgressBar()
        self.volume_bar.setRange(0, 100)
        self.volume_bar.setValue(50)
        self.volume_bar.setTextVisible(True)
        self.volume_bar.setFixedWidth(150)
        control_layout.addWidget(self.volume_bar)

        left_layout.addWidget(control_group)

        # --- 右侧播放列表 ---
        right_container = QGroupBox(" 播放列表 ")
        right_layout = QVBoxLayout(right_container)
        self.music_list_widget = QListWidget()
        self.music_list_widget.itemClicked.connect(self.select_music)
        self.music_list_widget.setFrameShape(QFrame.NoFrame)
        right_layout.addWidget(self.music_list_widget)

        main_layout.addWidget(left_container, stretch=7)
        main_layout.addWidget(right_container, stretch=3)

    def init_music(self):
        try:
            pygame.mixer.init()
            self.current_volume = 0.5
            pygame.mixer.music.set_volume(self.current_volume)
            self.status_label.setText("AUDIO SYSTEM: ONLINE")
            self.volume_bar.setValue(50)
        except pygame.error as e:
            self.status_label.setText(f"ERROR: {str(e)}")

        self.current_music = ""
        self.is_playing = False
        self.music_files = []
        self.current_index = -1
        self.paused_position = 0

    def open_settings(self):
        """打开设置窗口"""
        if self.settings_window is None:
            self.settings_window = SettingsWindow()
            self.settings_window.config_saved.connect(self.on_config_updated)

        # 每次打开都刷新一下配置显示（万一文件被外部改了）
        self.settings_window.show()
        self.settings_window.raise_()  # 窗口置顶

    def on_config_updated(self, new_config):
        """当设置窗口保存配置后的回调"""
        self.config = new_config
        self.load_music_from_config()
        self.status_label.setText("系统设置已更新")

    def load_music_from_config(self):
        """从配置中读取路径并加载"""
        folder = self.config.get("music_folder", "")
        if folder and os.path.exists(folder):
            self.load_music_from_folder(folder)
        else:
            self.status_label.setText("请点击[系统设置]选择音乐文件夹")

    def load_music_from_folder(self, folder_path):
        supported_formats = (".mp3", ".wav", ".ogg", ".flac", ".m4a")
        music_files = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(supported_formats):
                    music_files.append(os.path.join(root, file))

        if music_files:
            self.music_files = music_files
            self.update_music_list()
            self.current_index = 0
            self.status_label.setText(f"已加载: {len(music_files)} 首曲目")
            self.play_btn.setEnabled(True)
        else:
            self.status_label.setText("该文件夹未发现音乐文件")

    def update_music_list(self):
        self.music_list_widget.clear()
        for file in self.music_files:
            item = os.path.basename(file)
            self.music_list_widget.addItem(item)
            self.music_list_widget.item(self.music_list_widget.count() - 1).setIcon(
                self.style().standardIcon(QStyle.SP_MediaVolume)
            )

    def select_music(self, item):
        index = self.music_list_widget.row(item)
        if 0 <= index < len(self.music_files):
            self.current_index = index
            self.current_music = self.music_files[index]
            self.paused_position = 0
            if self.is_playing:
                self.play_music()

    def play_music(self):
        if not self.music_files:
            return
        if self.current_index == -1:
            self.current_index = 0
            self.current_music = self.music_files[0]
        try:
            pygame.mixer.music.load(self.current_music)
            pygame.mixer.music.set_volume(self.current_volume)
            if self.paused_position > 0:
                pygame.mixer.music.play(start=self.paused_position)
            else:
                pygame.mixer.music.play()
            self.is_playing = True
            self.status_label.setText(
                f"正在播放: {os.path.basename(self.current_music)}"
            )
            self.music_list_widget.setCurrentRow(self.current_index)
        except pygame.error as e:
            self.status_label.setText(f"ERROR: {str(e)}")

    def pause_music(self):
        if not self.music_files:
            return
        if self.is_playing:
            self.paused_position = pygame.mixer.music.get_pos() / 1000
            pygame.mixer.music.pause()
            self.is_playing = False
            self.status_label.setText("已暂停")
        else:
            self.play_music()

    def next_music(self):
        if not self.music_files:
            return
        self.paused_position = 0
        if self.current_index < len(self.music_files) - 1:
            self.current_index += 1
        else:
            self.current_index = 0
        self.current_music = self.music_files[self.current_index]
        self.play_music()

    def change_volume(self, change):
        new_volume = self.current_volume + change
        new_volume = max(0.0, min(1.0, new_volume))
        if new_volume != self.current_volume:
            self.current_volume = new_volume
            pygame.mixer.music.set_volume(self.current_volume)
            vol_percent = int(self.current_volume * 100)
            self.volume_bar.setValue(vol_percent)
            self.status_label.setText(f"音量调节: {vol_percent}%")

    def init_gesture_thread(self):
        self.gesture_thread = GestureRecognitionThread()
        self.gesture_thread.image_signal.connect(self.update_video)
        self.gesture_thread.gesture_signal.connect(self.handle_gesture)
        self.gesture_thread.error_signal.connect(self.update_status)
        self.gesture_thread.start()

    def update_video(self, original_frame, processed_frame):
        def set_frame(frame, label):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, c = rgb.shape
            qimg = QImage(rgb.data, w, h, w * c, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            scaled_pixmap = pixmap.scaled(
                label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            label.setPixmap(scaled_pixmap)

        set_frame(original_frame, self.original_video)
        set_frame(processed_frame, self.processed_video)

    def handle_gesture(self, gesture):
        self.gesture_label.setText(f"DETECTED: {gesture}")
        self.gesture_label.setStyleSheet(
            "font-size: 24px; color: #00bcd4; font-weight: bold;"
        )

        allow_repeat = gesture in ["V字手势", "音量加", "音量减"]
        if not allow_repeat and gesture == self.last_handled_gesture:
            return

        # --- 核心修改：使用配置文件中的映射来决定执行什么操作 ---
        gesture_map = self.config.get("gestures", {})
        action = gesture_map.get(gesture, "none")

        if action == "play":
            if not self.is_playing:
                self.play_music()
        elif action == "pause":
            self.pause_music()
        elif action == "next":
            self.next_music()
        elif action == "vol_up":
            self.change_volume(0.05)
        elif action == "vol_down":
            self.change_volume(-0.05)
        elif action == "none":
            pass  # 用户设置了无操作

        self.last_handled_gesture = gesture

    def update_status(self, message):
        self.status_label.setText(message)

    def closeEvent(self, event):
        if hasattr(self, "gesture_thread"):
            self.gesture_thread.stop()
            self.gesture_thread.wait(2000)
        pygame.mixer.quit()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MusicPlayer()
    window.show()
    sys.exit(app.exec())
