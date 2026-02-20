"""
main.py
PyQt5 GUI and controller integrating Camera1, Camera2, and SegmentationModel.

Features:
- Live preview of Camera 1 and Camera 2
- Button to capture measurements (Width, Height, Length)
- Button to load segmentation model
- Button to run segmentation on Camera 1 frame
- Displays segmented object image and numeric measurements

Run: python main.py
"""

import sys
import os
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from camera1 import Camera1
from camera2 import Camera2
from dieline import generate_dieline


def cv2_to_qimage(frame: np.ndarray) -> QtGui.QImage:
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


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('CartonIQ - Live Measurement & Dieline Generator')
        self.resize(1200, 700)

        self.camera1 = Camera1(camera_index=0, marker_size_cm=5.0)
        self.camera2 = Camera2(camera_index=1, marker_size_cm=5.0)
        
        # Store captured measurements
        self.measured_width = None
        self.measured_height = None
        self.measured_length = None

        self._setup_ui()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._update_frames)
        self.timer.start(30)

    def _setup_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QGridLayout(central)

        # Single large camera preview (shows active camera)
        self.camera_label = QtWidgets.QLabel('Camera Preview')
        self.camera_label.setFixedSize(800, 600)
        self.camera_label.setStyleSheet('background: #222; border: 2px solid #555;')
        self.camera_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.camera_label, 0, 0)

        # Right-side controls
        controls = QtWidgets.QVBoxLayout()
        
        # Workflow instructions
        workflow_label = QtWidgets.QLabel('Sequential Workflow:\n1. Measure Width & Height (Camera 1)\n2. Measure Length (Camera 2)')
        workflow_label.setStyleSheet('font-size: 12px; color: #888; padding: 10px;')
        controls.addWidget(workflow_label)
        
        controls.addWidget(QtWidgets.QLabel('─' * 40))

        # Camera 1 controls
        cam1_group = QtWidgets.QLabel('Camera 1 - Width & Height:')
        cam1_group.setStyleSheet('font-weight: bold; font-size: 14px; margin-top: 10px;')
        controls.addWidget(cam1_group)
        
        self.start_cam1_btn = QtWidgets.QPushButton('▶ Start Camera 1')
        self.start_cam1_btn.clicked.connect(self.start_camera1)
        self.start_cam1_btn.setStyleSheet('background: #2a5; color: white; padding: 8px;')
        controls.addWidget(self.start_cam1_btn)

        self.capture_cam1_btn = QtWidgets.QPushButton('📸 Capture Width & Height')
        self.capture_cam1_btn.clicked.connect(self.capture_camera1)
        self.capture_cam1_btn.setEnabled(False)
        controls.addWidget(self.capture_cam1_btn)
        
        self.stop_cam1_btn = QtWidgets.QPushButton('■ Stop Camera 1')
        self.stop_cam1_btn.clicked.connect(self.stop_camera1)
        self.stop_cam1_btn.setEnabled(False)
        controls.addWidget(self.stop_cam1_btn)

        controls.addWidget(QtWidgets.QLabel('─' * 40))

        # Camera 2 controls
        cam2_group = QtWidgets.QLabel('Camera 2 - Length:')
        cam2_group.setStyleSheet('font-weight: bold; font-size: 14px; margin-top: 10px;')
        controls.addWidget(cam2_group)
        
        self.start_cam2_btn = QtWidgets.QPushButton('▶ Start Camera 2')
        self.start_cam2_btn.clicked.connect(self.start_camera2)
        self.start_cam2_btn.setStyleSheet('background: #25a; color: white; padding: 8px;')
        controls.addWidget(self.start_cam2_btn)

        self.capture_cam2_btn = QtWidgets.QPushButton('📸 Capture Length')
        self.capture_cam2_btn.clicked.connect(self.capture_camera2)
        self.capture_cam2_btn.setEnabled(False)
        controls.addWidget(self.capture_cam2_btn)
        
        self.stop_cam2_btn = QtWidgets.QPushButton('■ Stop Camera 2')
        self.stop_cam2_btn.clicked.connect(self.stop_camera2)
        self.stop_cam2_btn.setEnabled(False)
        controls.addWidget(self.stop_cam2_btn)

        controls.addWidget(QtWidgets.QLabel('─' * 40))

        # Dieline generation controls
        dieline_group = QtWidgets.QLabel('Dieline Generation:')
        dieline_group.setStyleSheet('font-weight: bold; font-size: 14px; margin-top: 10px;')
        controls.addWidget(dieline_group)
        
        self.generate_dieline_btn = QtWidgets.QPushButton('📐 Generate Dieline')
        self.generate_dieline_btn.clicked.connect(self.generate_dieline)
        self.generate_dieline_btn.setStyleSheet('background: #2a5; color: white; padding: 10px; font-weight: bold;')
        self.generate_dieline_btn.setEnabled(False)
        controls.addWidget(self.generate_dieline_btn)
        
        self.dieline_status = QtWidgets.QLabel('Status: Waiting for measurements...')
        self.dieline_status.setStyleSheet('font-size: 10px; color: #888; padding: 4px;')
        self.dieline_status.setWordWrap(True)
        controls.addWidget(self.dieline_status)

        controls.addWidget(QtWidgets.QLabel('─' * 40))

        # Measurement results
        results_group = QtWidgets.QLabel('Measurements:')
        results_group.setStyleSheet('font-weight: bold; font-size: 14px; margin-top: 10px;')
        controls.addWidget(results_group)
        
        self.width_label = QtWidgets.QLabel('Width: -')
        self.height_label = QtWidgets.QLabel('Height: -')
        self.length_label = QtWidgets.QLabel('Length: -')
        for lbl in (self.width_label, self.height_label, self.length_label):
            lbl.setStyleSheet('font-size: 16px; padding: 4px;')
            controls.addWidget(lbl)

        controls.addStretch()
        layout.addLayout(controls, 0, 1)

    def start_camera1(self):
        # Close camera 2 if open
        if self.camera2.is_opened:
            self.camera2.close()
            self.stop_cam2_btn.setEnabled(False)
            self.capture_cam2_btn.setEnabled(False)
        
        ok = self.camera1.open()
        if ok:
            self.start_cam1_btn.setEnabled(False)
            self.capture_cam1_btn.setEnabled(True)
            self.stop_cam1_btn.setEnabled(True)
        else:
            QMessageBox.warning(self, 'Camera 1', 'Failed to open Camera 1')

    def stop_camera1(self):
        self.camera1.close()
        self.start_cam1_btn.setEnabled(True)
        self.capture_cam1_btn.setEnabled(False)
        self.stop_cam1_btn.setEnabled(False)
        self.camera_label.setText('Camera Preview')

    def capture_camera1(self):
        if not self.camera1.is_opened:
            QMessageBox.warning(self, 'Camera 1', 'Camera 1 is not opened')
            return
        
        res = self.camera1.get_width_and_height(display=False)
        if res['width'] is not None:
            self.width_label.setText(f'Width: {res["width"]} cm')
            self.measured_width = res['width']
        else:
            self.width_label.setText('Width: -')
            self.measured_width = None
        if res['height'] is not None:
            self.height_label.setText(f'Height: {res["height"]} cm')
            self.measured_height = res['height']
        else:
            self.height_label.setText('Height: -')
            self.measured_height = None
        
        self._update_dieline_button_state()
        
        if res['success']:
            QMessageBox.information(self, 'Camera 1', f'Captured!\nWidth: {res["width"]} cm\nHeight: {res["height"]} cm')

    def start_camera2(self):
        # Close camera 1 if open
        if self.camera1.is_opened:
            self.camera1.close()
            self.stop_cam1_btn.setEnabled(False)
            self.capture_cam1_btn.setEnabled(False)
        
        ok = self.camera2.open()
        if ok:
            self.start_cam2_btn.setEnabled(False)
            self.capture_cam2_btn.setEnabled(True)
            self.stop_cam2_btn.setEnabled(True)
        else:
            QMessageBox.warning(self, 'Camera 2', 'Failed to open Camera 2')

    def stop_camera2(self):
        self.camera2.close()
        self.start_cam2_btn.setEnabled(True)
        self.capture_cam2_btn.setEnabled(False)
        self.stop_cam2_btn.setEnabled(False)
        self.camera_label.setText('Camera Preview')

    def capture_camera2(self):
        if not self.camera2.is_opened:
            QMessageBox.warning(self, 'Camera 2', 'Camera 2 is not opened')
            return
        
        res = self.camera2.get_length(display=False)
        if res['length'] is not None:
            self.length_label.setText(f'Length: {res["length"]} cm')
            self.measured_length = res['length']
        else:
            self.length_label.setText('Length: -')
            self.measured_length = None
        
        self._update_dieline_button_state()
        
        if res['success']:
            QMessageBox.information(self, 'Camera 2', f'Captured!\nLength: {res["length"]} cm')

    def _update_frames(self):
        # Show whichever camera is currently active
        if self.camera1.is_opened:
            frame = self.camera1.read_frame()
            if frame is not None:
                qimg = cv2_to_qimage(frame)
                pix = QtGui.QPixmap.fromImage(qimg).scaled(self.camera_label.size(), QtCore.Qt.KeepAspectRatio)
                self.camera_label.setPixmap(pix)
        elif self.camera2.is_opened:
            frame = self.camera2.read_frame()
            if frame is not None:
                qimg = cv2_to_qimage(frame)
                pix = QtGui.QPixmap.fromImage(qimg).scaled(self.camera_label.size(), QtCore.Qt.KeepAspectRatio)
                self.camera_label.setPixmap(pix)
    
    def _update_dieline_button_state(self):
        """Enable Generate Dieline button only when all measurements are available."""
        if self.measured_width is not None and self.measured_height is not None and self.measured_length is not None:
            self.generate_dieline_btn.setEnabled(True)
            self.dieline_status.setText(f'✓ Ready: W={self.measured_width}cm, H={self.measured_height}cm, L={self.measured_length}cm')
            self.dieline_status.setStyleSheet('font-size: 10px; color: #2a5; padding: 4px;')
        else:
            self.generate_dieline_btn.setEnabled(False)
            missing = []
            if self.measured_width is None:
                missing.append('Width')
            if self.measured_height is None:
                missing.append('Height')
            if self.measured_length is None:
                missing.append('Length')
            self.dieline_status.setText(f'Missing: {", ".join(missing)}')
            self.dieline_status.setStyleSheet('font-size: 10px; color: #888; padding: 4px;')
    
    def generate_dieline(self):
        """Generate dieline from captured measurements."""
        if self.measured_width is None or self.measured_height is None or self.measured_length is None:
            QMessageBox.warning(self, 'Dieline Generation', 
                              'Please capture all measurements first:\n- Width & Height (Camera 1)\n- Length (Camera 2)')
            return
        
        try:
            # Generate dieline with captured measurements
            print("\n" + "="*60)
            print("GENERATING DIELINE FROM LIVE MEASUREMENTS")
            print("="*60)
            
            save_path = 'carton_dieline.svg'
            fig = generate_dieline(
                width=self.measured_width,
                height=self.measured_height,
                length=self.measured_length,
                show=True,  # Display the dieline
                save_path=save_path
            )
            
            # Show success message
            msg = f"Dieline generated successfully!\n\n"
            msg += f"Measurements used:\n"
            msg += f"  Width:  {self.measured_width} cm\n"
            msg += f"  Height: {self.measured_height} cm\n"
            msg += f"  Length: {self.measured_length} cm\n\n"
            msg += f"Saved to: {save_path}"
            
            QMessageBox.information(self, 'Dieline Generated', msg)
            
            print("="*60)
            print("DIELINE GENERATION COMPLETE")
            print("="*60)
            
        except Exception as e:
            QMessageBox.critical(self, 'Dieline Generation Error', 
                               f'Failed to generate dieline:\n{str(e)}')
            print(f"Error generating dieline: {e}")

    def closeEvent(self, event):
        try:
            self.camera1.close()
            self.camera2.close()
        except Exception:
            pass
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
