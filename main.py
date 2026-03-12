"""
gui_app.py
Simple PyQt5 GUI with dual camera views and control buttons.

Features:
- Two camera windows displayed side-by-side
- Object detection and ArUco marker detection
- Capture button to get measurements
- Generate Dieline button to create the dieline
"""

import os
import sys
import cv2
import numpy as np
import torch
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtSvg import QSvgWidget
from object_detector import ObjectDetector  


def cv2_to_qimage(frame: np.ndarray) -> QtGui.QImage:
    """Convert OpenCV BGR frame to QImage."""
    if frame is None:
        return QtGui.QImage()
    if frame.ndim == 2:
        h, w = frame.shape
        bytes_per_line = w
        return QtGui.QImage(frame.data, w, h, bytes_per_line, QtGui.QImage.Format_Grayscale8).copy()
    h, w, ch = frame.shape
    bytes_per_line = ch * w
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888).copy()


class DielinePreviewWindow(QtWidgets.QMainWindow):
    def __init__(self, svg_path: str,
                 measured_dimensions: tuple,
                 adjusted_dimensions: tuple):
        super().__init__()
        self.svg_path = svg_path

        measured_width, measured_height, measured_length = measured_dimensions
        adj_length, adj_width, adj_height = adjusted_dimensions

        self.setWindowTitle('Carton Dieline Preview')
        self.setMinimumSize(1000, 640)
        self.resize(1200, 720)
        self.setStyleSheet('background-color: #f5f5f5;')

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        # ── Title bar ──────────────────────────────────────────────
        title = QtWidgets.QLabel('Carton Dieline Preview')
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet(
            'font-size: 20px; font-weight: bold; color: #1a1a2e;'
            'padding: 6px 0; border-bottom: 2px solid #ddd;'
        )
        root.addWidget(title)

        # ── Content row (left panel  +  preview area) ───────────────
        content_row = QtWidgets.QHBoxLayout()
        content_row.setSpacing(16)
        root.addLayout(content_row, 1)

        # LEFT – measurements card ──────────────────────────────────
        left_card = QtWidgets.QFrame()
        left_card.setFixedWidth(220)
        left_card.setStyleSheet(
            'QFrame { background: #ffffff; border: 1px solid #ddd;'
            ' border-radius: 10px; }'
        )
        left_layout = QtWidgets.QVBoxLayout(left_card)
        left_layout.setContentsMargins(18, 20, 18, 20)
        left_layout.setSpacing(0)

        card_title = QtWidgets.QLabel('Measurements')
        card_title.setAlignment(QtCore.Qt.AlignCenter)
        card_title.setStyleSheet(
            'font-size: 13px; font-weight: bold; color: #555;'
            'padding-bottom: 14px; border-bottom: 1px solid #eee;'
        )
        left_layout.addWidget(card_title)
        left_layout.addSpacing(16)

        def _dim_block(label: str, measured: float, adjusted: float) -> QtWidgets.QWidget:
            block = QtWidgets.QWidget()
            bl = QtWidgets.QVBoxLayout(block)
            bl.setContentsMargins(0, 0, 0, 0)
            bl.setSpacing(2)
            lbl = QtWidgets.QLabel(label)
            lbl.setStyleSheet('font-size: 11px; color: #888; font-weight: bold; text-transform: uppercase;')
            val = QtWidgets.QLabel(f'{measured:.1f} cm')
            val.setStyleSheet('font-size: 22px; font-weight: bold; color: #1a1a2e;')
            adj_lbl = QtWidgets.QLabel(f'+ 6 cm  →  {adjusted:.1f} cm')
            adj_lbl.setStyleSheet('font-size: 11px; color: #e07b00;')
            bl.addWidget(lbl)
            bl.addWidget(val)
            bl.addWidget(adj_lbl)
            return block

        left_layout.addWidget(_dim_block('Length', measured_length, adj_length))
        left_layout.addSpacing(18)
        left_layout.addWidget(_dim_block('Width', measured_width, adj_width))
        left_layout.addSpacing(18)
        left_layout.addWidget(_dim_block('Height', measured_height, adj_height))
        left_layout.addStretch()

        note = QtWidgets.QLabel('6 cm clearance applied\nto each dimension')
        note.setAlignment(QtCore.Qt.AlignCenter)
        note.setWordWrap(True)
        note.setStyleSheet(
            'font-size: 10px; color: #aaa; padding-top: 10px;'
            'border-top: 1px solid #eee;'
        )
        left_layout.addWidget(note)
        content_row.addWidget(left_card)

        # RIGHT – SVG preview ──────────────────────────────────────
        preview_frame = QtWidgets.QFrame()
        preview_frame.setStyleSheet(
            'QFrame { background: #ffffff; border: 3px solid #e8a000;'
            ' border-radius: 10px; }'
        )
        preview_layout = QtWidgets.QVBoxLayout(preview_frame)
        preview_layout.setContentsMargins(6, 6, 6, 6)

        self.svg_widget = QSvgWidget(svg_path)
        self.svg_widget.setStyleSheet('background: transparent;')
        preview_layout.addWidget(self.svg_widget)
        content_row.addWidget(preview_frame, 1)

        # ── Bottom bar ─────────────────────────────────────────────
        bottom_bar = QtWidgets.QHBoxLayout()
        bottom_bar.setSpacing(16)

        self.path_label = QtWidgets.QLabel(f'Temp file: {svg_path}')
        self.path_label.setStyleSheet('font-size: 10px; color: #aaa;')
        self.path_label.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred
        )
        bottom_bar.addWidget(self.path_label)

        save_btn = QtWidgets.QPushButton('  Save as SVG  ')
        save_btn.setFixedHeight(40)
        save_btn.setStyleSheet('''
            QPushButton {
                background: #e8a000; color: #fff;
                font-size: 14px; font-weight: bold;
                border: none; border-radius: 8px;
                padding: 0 24px;
            }
            QPushButton:hover  { background: #f5b300; }
            QPushButton:pressed { background: #c98a00; }
        ''')
        save_btn.clicked.connect(self._save_to_desktop)
        bottom_bar.addWidget(save_btn)

        root.addLayout(bottom_bar)

    # ─────────────────────────────────────────────────────────────
    def _save_to_desktop(self):
        """Save the generated SVG to the user's Desktop via file dialog (defaults to Desktop)."""
        desktop = os.path.join(os.path.expanduser('~'), 'Desktop')
        default_path = os.path.join(desktop, 'carton_dieline.svg')
        dest, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save Dieline as SVG', default_path,
            'SVG Files (*.svg);;All Files (*)'
        )
        if not dest:
            return
        try:
            import shutil
            shutil.copy2(self.svg_path, dest)
            self.path_label.setText(f'Saved: {dest}')
            QMessageBox.information(self, 'Saved', f'Dieline saved to:\n{dest}')
        except Exception as exc:
            QMessageBox.critical(self, 'Save Error', str(exc))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if os.path.exists(self.svg_path):
            self.svg_widget.load(self.svg_path)


class DualCameraApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Carton Measurement System')
        self.resize(1000, 600)

        self.cap1 = None
        self.cap2 = None
        self.detector1 = None
        self.detector2 = None

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        self.ARUCO_MARKER_SIZE_CM = 5.0

        self.width = None
        self.height = None
        self.length = None
        self.pixels_per_cm_cam1 = None
        self.pixels_per_cm_cam2 = None
        self.preview_window = None
        self.not_allowed_active = False

        self._setup_ui()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._update_frames)
        self.timer.start(30)

        self._init_cameras()
        self._init_detectors()

    def _setup_ui(self):
        """Setup the user interface."""
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)

        camera_layout = QtWidgets.QHBoxLayout()

        cam1_container = QtWidgets.QVBoxLayout()
        cam1_label = QtWidgets.QLabel('Camera 1 - Width & Height')
        cam1_label.setAlignment(QtCore.Qt.AlignCenter)
        cam1_label.setStyleSheet('font-size: 14px; font-weight: bold; padding: 5px;')
        cam1_container.addWidget(cam1_label)

        self.camera1_view = QtWidgets.QLabel()
        self.camera1_view.setFixedSize(450, 350)
        self.camera1_view.setStyleSheet('background: #2a2a2a; border: 3px solid #00ff00;')
        self.camera1_view.setAlignment(QtCore.Qt.AlignCenter)
        self.camera1_view.setText('Camera 1')
        cam1_container.addWidget(self.camera1_view)
        camera_layout.addLayout(cam1_container)

        cam2_container = QtWidgets.QVBoxLayout()
        cam2_label = QtWidgets.QLabel('Camera 2 - Length')
        cam2_label.setAlignment(QtCore.Qt.AlignCenter)
        cam2_label.setStyleSheet('font-size: 14px; font-weight: bold; padding: 5px;')
        cam2_container.addWidget(cam2_label)

        self.camera2_view = QtWidgets.QLabel()
        self.camera2_view.setFixedSize(450, 350)
        self.camera2_view.setStyleSheet('background: #2a2a2a; border: 3px solid #00ff00;')
        self.camera2_view.setAlignment(QtCore.Qt.AlignCenter)
        self.camera2_view.setText('Camera 2')
        cam2_container.addWidget(self.camera2_view)
        camera_layout.addLayout(cam2_container)

        main_layout.addLayout(camera_layout)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addStretch()

        self.capture_btn = QtWidgets.QPushButton('CAPTURE')
        self.capture_btn.setFixedSize(200, 50)
        self.capture_btn.setStyleSheet('''
            QPushButton {
                background: #4080ff;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border: 2px solid #2060dd;
                border-radius: 10px;
            }
            QPushButton:hover {
                background: #5090ff;
            }
            QPushButton:pressed {
                background: #3070ee;
            }
        ''')
        self.capture_btn.clicked.connect(self.capture_measurements)
        button_layout.addWidget(self.capture_btn)

        button_layout.addSpacing(30)

        self.generate_btn = QtWidgets.QPushButton('GENERATE DIELINE')
        self.generate_btn.setFixedSize(200, 50)
        self.generate_btn.setStyleSheet('''
            QPushButton {
                background: #4080ff;
                color: white;
                font-size: 16px;
                font-weight: bold;
                border: 2px solid #2060dd;
                border-radius: 10px;
            }
            QPushButton:hover {
                background: #5090ff;
            }
            QPushButton:pressed {
                background: #3070ee;
            }
            QPushButton:disabled {
                background: #666;
                border: 2px solid #555;
                color: #999;
            }
        ''')
        self.generate_btn.clicked.connect(self.generate_dieline)
        self.generate_btn.setEnabled(False)
        button_layout.addWidget(self.generate_btn)

        button_layout.addStretch()

        main_layout.addSpacing(20)
        main_layout.addLayout(button_layout)
        main_layout.addSpacing(20)

        self.status_label = QtWidgets.QLabel('Ready - Place ArUco markers in both camera views, then click CAPTURE')
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        self.status_label.setStyleSheet('font-size: 12px; color: #888; padding: 10px;')
        main_layout.addWidget(self.status_label)

    def _init_cameras(self):
        """Initialize both cameras."""
        try:
            self.cap1 = cv2.VideoCapture(0)
            self.cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            print('Camera 1 initialized')
        except Exception as e:
            print(f'Error opening Camera 1: {e}')

        try:
            self.cap2 = cv2.VideoCapture(1)
            self.cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            print('Camera 2 initialized')
        except Exception as e:
            print(f'Error opening Camera 2: {e}')

    def _init_detectors(self):
        """Initialize object detectors for both cameras."""
        try:
            self.detector1 = ObjectDetector(model_path='camera1_segmodel.pt', conf_threshold=0.50)
            if self.detector1.load_model():
                print('Camera 1 object detector loaded')
            else:
                self.detector1 = None
        except Exception as e:
            print(f'Error loading Camera 1 detector: {e}')
            self.detector1 = None

        try:
            self.detector2 = ObjectDetector(model_path='camera2_segmodel.pt', conf_threshold=0.50)
            if self.detector2.load_model():
                print('Camera 2 object detector loaded')
            else:
                self.detector2 = None
        except Exception as e:
            print(f'Error loading Camera 2 detector: {e}')
            self.detector2 = None

    def _update_frames(self):
        """Update both camera feeds with object detection and ArUco markers."""
        not_allowed_this_frame = False

        if self.cap1 is not None and self.cap1.isOpened():
            ret, frame = self.cap1.read()
            if ret:
                if self.detector1 is not None:
                    annotated_frame, detections = self.detector1.detect(frame, draw_boxes=True)
                    frame = annotated_frame
                    if any(d['class_name'] == 'Not allowed' for d in detections):
                        not_allowed_this_frame = True

                corners, ids, rejected = self.aruco_detector.detectMarkers(frame)
                if ids is not None and len(ids) > 0:
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                    marker_corners = corners[0][0]
                    top_width = np.linalg.norm(marker_corners[0] - marker_corners[1])
                    bottom_width = np.linalg.norm(marker_corners[3] - marker_corners[2])
                    marker_width_pixels = (top_width + bottom_width) / 2
                    self.pixels_per_cm_cam1 = marker_width_pixels / self.ARUCO_MARKER_SIZE_CM

                qimg = cv2_to_qimage(frame)
                pixmap = QtGui.QPixmap.fromImage(qimg).scaled(
                    self.camera1_view.size(),
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation,
                )
                self.camera1_view.setPixmap(pixmap)

        if self.cap2 is not None and self.cap2.isOpened():
            ret, frame = self.cap2.read()
            if ret:
                if self.detector2 is not None:
                    annotated_frame, detections = self.detector2.detect(frame, draw_boxes=True)
                    frame = annotated_frame
                    if any(d['class_name'] == 'Not allowed' for d in detections):
                        not_allowed_this_frame = True

                corners, ids, rejected = self.aruco_detector.detectMarkers(frame)
                if ids is not None and len(ids) > 0:
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                    marker_corners = corners[0][0]
                    top_width = np.linalg.norm(marker_corners[0] - marker_corners[1])
                    bottom_width = np.linalg.norm(marker_corners[3] - marker_corners[2])
                    marker_width_pixels = (top_width + bottom_width) / 2
                    self.pixels_per_cm_cam2 = marker_width_pixels / self.ARUCO_MARKER_SIZE_CM

                qimg = cv2_to_qimage(frame)
                pixmap = QtGui.QPixmap.fromImage(qimg).scaled(
                    self.camera2_view.size(),
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation,
                )
                self.camera2_view.setPixmap(pixmap)

        # Update button states when "Not allowed" detection status changes
        if not_allowed_this_frame != self.not_allowed_active:
            self.not_allowed_active = not_allowed_this_frame
            if not_allowed_this_frame:
                self.capture_btn.setEnabled(False)
                self.generate_btn.setEnabled(False)
                self.status_label.setText('WARNING: "Not allowed" object detected — capture disabled')
                self.status_label.setStyleSheet('font-size: 12px; color: #ff4444; font-weight: bold; padding: 10px;')
            else:
                self.capture_btn.setEnabled(True)
                self.generate_btn.setEnabled(
                    self.width is not None and self.height is not None and self.length is not None
                )
                self.status_label.setText('Ready - Place ArUco markers in both camera views, then click CAPTURE')
                self.status_label.setStyleSheet('font-size: 12px; color: #888; padding: 10px;')

    def capture_measurements(self):
        """Capture measurements from both cameras."""
        if self.cap1 is None or not self.cap1.isOpened():
            QMessageBox.warning(self, 'Error', 'Camera 1 is not available')
            return

        if self.cap2 is None or not self.cap2.isOpened():
            QMessageBox.warning(self, 'Error', 'Camera 2 is not available')
            return

        ret1, frame1 = self.cap1.read()
        if not ret1:
            QMessageBox.warning(self, 'Error', 'Failed to capture from Camera 1')
            return

        ret2, frame2 = self.cap2.read()
        if not ret2:
            QMessageBox.warning(self, 'Error', 'Failed to capture from Camera 2')
            return

        if self.pixels_per_cm_cam1 is None:
            QMessageBox.warning(self, 'Error', 'Camera 1 not calibrated. Please place ArUco marker in Camera 1 view.')
            return

        if self.pixels_per_cm_cam2 is None:
            QMessageBox.warning(self, 'Error', 'Camera 2 not calibrated. Please place ArUco marker in Camera 2 view.')
            return

        if self.detector1 is not None:
            _, detections1 = self.detector1.detect(frame1, draw_boxes=False)
            if len(detections1) > 0:
                bbox = detections1[0]['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                width_pixels = x2 - x1
                height_pixels = y2 - y1
                self.width = width_pixels / self.pixels_per_cm_cam1
                self.height = height_pixels / self.pixels_per_cm_cam1
            else:
                QMessageBox.warning(self, 'Error', 'No object detected in Camera 1')
                return
        else:
            QMessageBox.warning(self, 'Error', 'Object detector for Camera 1 not loaded')
            return

        if self.detector2 is not None:
            _, detections2 = self.detector2.detect(frame2, draw_boxes=False)
            if len(detections2) > 0:
                bbox = detections2[0]['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                length_pixels = x2 - x1
                self.length = length_pixels / self.pixels_per_cm_cam2
            else:
                QMessageBox.warning(self, 'Error', 'No object detected in Camera 2')
                return
        else:
            QMessageBox.warning(self, 'Error', 'Object detector for Camera 2 not loaded')
            return

        self.status_label.setText(
            f'Captured: Width={self.width:.1f}cm, Height={self.height:.1f}cm, Length={self.length:.1f}cm'
        )
        self.status_label.setStyleSheet('font-size: 12px; color: #00ff00; padding: 10px;')
        self.generate_btn.setEnabled(True)

        QMessageBox.information(
            self,
            'Measurements Captured',
            f'Width: {self.width:.1f} cm\nHeight: {self.height:.1f} cm\nLength: {self.length:.1f} cm'
        )

    def generate_dieline(self):
        """Generate dieline from captured measurements."""
        if self.width is None or self.height is None or self.length is None:
            QMessageBox.warning(self, 'Error', 'Please capture measurements first')
            return

        try:
            from dieline import adjust_dimensions, build_reference_svg

            adjusted_length, adjusted_width, adjusted_height = adjust_dimensions(
                self.length,
                self.width,
                self.height,
                clearance=6.0,
            )

            save_path = os.path.abspath('carton_dieline.svg')
            build_reference_svg(
                length=adjusted_length,
                width=adjusted_width,
                height=adjusted_height,
                filename=save_path,
            )

            self.preview_window = DielinePreviewWindow(
                svg_path=save_path,
                measured_dimensions=(self.width, self.height, self.length),
                adjusted_dimensions=(adjusted_length, adjusted_width, adjusted_height),
            )
            self.preview_window.show()
            self.preview_window.raise_()
            self.preview_window.activateWindow()

        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to generate dieline:\n{str(e)}')

    def closeEvent(self, event):
        """Clean up when closing the application."""
        if self.cap1 is not None:
            self.cap1.release()
        if self.cap2 is not None:
            self.cap2.release()
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = DualCameraApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
