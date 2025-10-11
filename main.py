from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QFileDialog,
    QMessageBox,
    QListWidget,
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
        self.cooldown_frames = 15
        print("手势识别线程已初始化")

    def run(self):
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.error_signal.emit("无法打开摄像头，请检查设备")
                return

            self.error_signal.emit("摄像头打开成功，开始手势检测")
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    self.error_signal.emit("无法获取摄像头帧数据")
                    break

                if self.gesture_cooldown > 0:
                    self.gesture_cooldown -= 1

                processed_frame, gesture = self.process_and_recognize(frame)
                self.image_signal.emit(frame, processed_frame)

                if gesture and self.gesture_cooldown == 0:
                    if gesture == self.prev_gesture:
                        self.gesture_count += 1
                        if self.gesture_count >= 3:
                            self.gesture_signal.emit(gesture)
                            self.prev_gesture = None
                            self.gesture_count = 0
                            self.gesture_cooldown = self.cooldown_frames
                    else:
                        self.prev_gesture = gesture
                        self.gesture_count = 1

                time.sleep(0.05)

            self.error_signal.emit("手势识别线程正常结束")
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

        thumb_up = thumb_tip.y < thumb_ip - 0.02
        index_up = index_tip.y < index_pip - 0.02
        middle_up = middle_tip.y < middle_pip - 0.02
        ring_up = ring_tip.y < ring_pip - 0.02
        pinky_up = pinky_tip.y < pinky_pip - 0.02

        if (
            not thumb_up
            and not index_up
            and not middle_up
            and not ring_up
            and not pinky_up
        ):
            return "拳头"

        if thumb_up and index_up and middle_up and ring_up and pinky_up:
            return "掌心"

        thumb_index_dist = np.sqrt(
            (thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2
        )
        if thumb_index_dist < 0.05 and not middle_up and not ring_up and not pinky_up:
            return "OK 手势"

        return None

    def stop(self):
        self.running = False
        print("手势识别线程已停止")


class MusicPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.init_music()
        self.init_gesture_thread()
        self.last_handled_gesture = None
        self.auto_load_music()  # 启动时自动加载音乐

    def init_ui(self):
        self.setWindowTitle("MediaPipe手势控制音乐播放器")
        self.setGeometry(100, 100, 1000, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # 视频显示区域
        video_layout = QHBoxLayout()
        self.original_video = QLabel("原始视频")
        self.original_video.setAlignment(Qt.AlignCenter)
        self.original_video.setMinimumSize(400, 400)
        video_layout.addWidget(self.original_video)

        self.processed_video = QLabel("检测结果（带关键点）")
        self.processed_video.setAlignment(Qt.AlignCenter)
        self.processed_video.setMinimumSize(400, 400)
        video_layout.addWidget(self.processed_video)
        layout.addLayout(video_layout)

        # 状态与手势显示
        self.status_label = QLabel("系统状态: 初始化中...")
        layout.addWidget(self.status_label)
        self.gesture_label = QLabel("手势识别结果：等待识别...")
        layout.addWidget(self.gesture_label)

        # 音乐列表
        self.music_list_widget = QListWidget()
        self.music_list_widget.itemClicked.connect(self.select_music)
        layout.addWidget(QLabel("音乐列表:"))
        layout.addWidget(self.music_list_widget)

        # 控制按钮
        control_layout = QHBoxLayout()
        self.load_btn = QPushButton("加载音乐文件夹")
        self.load_btn.clicked.connect(self.load_music_folder)
        self.play_btn = QPushButton("播放")
        self.play_btn.clicked.connect(self.play_music)
        self.pause_btn = QPushButton("暂停")
        self.pause_btn.clicked.connect(self.pause_music)
        self.next_btn = QPushButton("下一首")
        self.next_btn.clicked.connect(self.next_music)
        control_layout.addWidget(self.load_btn)
        control_layout.addWidget(self.play_btn)
        control_layout.addWidget(self.pause_btn)
        control_layout.addWidget(self.next_btn)
        layout.addLayout(control_layout)

    def init_music(self):
        try:
            pygame.mixer.init()
            self.status_label.setText("系统状态: 音频系统初始化成功")
        except pygame.error as e:
            self.status_label.setText(f"系统状态: 音频初始化失败: {str(e)}")
            QMessageBox.warning(self, "音频错误", f"无法初始化音频系统: {str(e)}")

        self.current_music = ""
        self.is_playing = False
        self.music_files = []
        self.current_index = -1
        self.paused_position = 0  # 记录暂停时的播放位置（单位：秒）

    def auto_load_music(self):
        """启动时自动从默认路径加载音乐"""
        default_folders = [
            "music",  # 当前目录下的music文件夹（可修改为常用音乐路径）
            os.path.expanduser("~/Music"),  # 用户音乐文件夹（macOS/Linux）
            os.path.expanduser("~/Downloads"),  # 下载文件夹
        ]

        for folder in default_folders:
            if os.path.exists(folder):
                self.load_music_from_folder(folder)
                if self.music_files:
                    break

        if not self.music_files:
            self.status_label.setText("系统状态: 未找到音乐文件，请手动加载")
        else:
            self.status_label.setText(
                f"系统状态: 已自动加载 {len(self.music_files)} 首音乐"
            )

    def load_music_folder(self):
        """手动选择文件夹加载音乐"""
        folder_path = QFileDialog.getExistingDirectory(self, "选择音乐文件夹")
        if folder_path:
            self.load_music_from_folder(folder_path)

    def load_music_from_folder(self, folder_path):
        """从指定文件夹扫描并加载音乐文件"""
        supported_formats = (".mp3", ".wav", ".ogg", ".flac", ".m4a")
        music_files = []

        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(supported_formats):
                    music_files.append(os.path.join(root, file))

        if music_files:
            self.music_files = music_files
            self.update_music_list()
            self.current_index = 0  # 自动选中第一首
            self.play_btn.setEnabled(True)
            self.status_label.setText(f"系统状态: 已加载 {len(music_files)} 首音乐")
        else:
            self.status_label.setText(f"系统状态: 在 {folder_path} 中未找到音乐文件")

    def update_music_list(self):
        """更新音乐列表UI显示"""
        self.music_list_widget.clear()
        for file in self.music_files:
            self.music_list_widget.addItem(os.path.basename(file))

    def select_music(self, item):
        """从列表中选择音乐并切换"""
        index = self.music_list_widget.row(item)
        if 0 <= index < len(self.music_files):
            self.current_index = index
            self.current_music = self.music_files[index]
            self.paused_position = 0  # 切换音乐时重置暂停位置
            self.status_label.setText(
                f"系统状态: 已选择: {os.path.basename(self.current_music)}"
            )
            if self.is_playing:
                self.play_music()  # 若正在播放，立即切换

    def play_music(self):
        """播放/继续播放音乐（支持从暂停位置恢复）"""
        if not self.music_files:
            QMessageBox.information(self, "提示", "请先加载音乐文件")
            return

        # 自动选中第一首（若未选择）
        if self.current_index == -1 and self.music_files:
            self.current_index = 0
            self.current_music = self.music_files[0]

        try:
            pygame.mixer.music.load(self.current_music)
            if self.paused_position > 0:  # 从暂停位置继续
                pygame.mixer.music.play(start=self.paused_position)
            else:  # 从头播放
                pygame.mixer.music.play()
            self.is_playing = True
            self.play_btn.setText("播放中")
            self.status_label.setText(
                f"系统状态: 正在播放: {os.path.basename(self.current_music)}"
            )
            self.music_list_widget.setCurrentRow(self.current_index)
        except pygame.error as e:
            self.status_label.setText(f"系统状态: 播放失败: {str(e)}")
            QMessageBox.warning(self, "播放错误", f"无法播放音乐: {str(e)}")

    def pause_music(self):
        """暂停音乐并记录播放位置"""
        if not self.music_files:
            return

        if self.is_playing:
            # 获取当前播放位置（毫秒 → 转换为秒）
            self.paused_position = pygame.mixer.music.get_pos() / 1000
            pygame.mixer.music.pause()
            self.is_playing = False
            self.play_btn.setText("播放")
            self.status_label.setText(
                f"系统状态: 已暂停: {os.path.basename(self.current_music)}"
            )
        else:
            self.play_music()  # 继续播放时调用play_music（从暂停位置恢复）

    def next_music(self):
        """切换下一首音乐"""
        if not self.music_files:
            return

        self.paused_position = 0  # 切换歌曲时重置暂停位置
        if self.current_index < len(self.music_files) - 1:
            self.current_index += 1
        else:
            self.current_index = 0  # 循环播放

        self.current_music = self.music_files[self.current_index]
        if self.is_playing:
            self.play_music()
        else:
            self.status_label.setText(
                f"系统状态: 已选择下一首: {os.path.basename(self.current_music)}"
            )
            self.music_list_widget.setCurrentRow(self.current_index)

    def init_gesture_thread(self):
        """初始化手势识别线程"""
        self.gesture_thread = GestureRecognitionThread()
        self.gesture_thread.image_signal.connect(self.update_video)
        self.gesture_thread.gesture_signal.connect(self.handle_gesture)
        self.gesture_thread.error_signal.connect(self.update_status)
        self.gesture_thread.start()
        self.status_label.setText("系统状态: 正在初始化MediaPipe手势识别...")

    def update_video(self, original_frame, processed_frame):
        """更新视频显示"""
        try:
            rgb_original = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
            h, w, c = rgb_original.shape
            q_original = QImage(rgb_original.data, w, h, w * c, QImage.Format_RGB888)
            self.original_video.setPixmap(
                QPixmap.fromImage(q_original).scaled(
                    self.original_video.width(),
                    self.original_video.height(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

            rgb_processed = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            q_processed = QImage(rgb_processed.data, w, h, w * c, QImage.Format_RGB888)
            self.processed_video.setPixmap(
                QPixmap.fromImage(q_processed).scaled(
                    self.processed_video.width(),
                    self.processed_video.height(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        except Exception as e:
            self.update_status(f"视频更新错误: {str(e)}")

    def handle_gesture(self, gesture):
        """处理手势控制逻辑"""
        self.gesture_label.setText(f"手势识别结果：{gesture}")
        self.status_label.setText(f"系统状态: 识别到手势 - {gesture}")

        if gesture == self.last_handled_gesture:
            return  # 忽略重复手势

        # 手势与操作映射
        if gesture == "拳头" and not self.is_playing:
            self.play_music()
            self.last_handled_gesture = gesture
        elif gesture == "掌心":
            self.pause_music()
            self.last_handled_gesture = gesture
        elif gesture == "OK 手势":
            self.next_music()
            self.last_handled_gesture = gesture

    def update_status(self, message):
        """更新系统状态显示"""
        self.status_label.setText(f"系统状态: {message}")

    def closeEvent(self, event):
        """程序关闭时释放资源"""
        if hasattr(self, "gesture_thread") and self.gesture_thread.isRunning():
            self.gesture_thread.stop()
            self.gesture_thread.wait(5000)
        pygame.mixer.quit()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MusicPlayer()
    window.show()
    sys.exit(app.exec())
