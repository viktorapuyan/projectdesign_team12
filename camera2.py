"""
Camera 2 module: Measures object length using ArUco markers.

Exposes a Camera2 class with methods to open the camera, capture frames,
and compute the object's length (cm) using ArUco markers and utilities
in `utils.py`.
"""

import cv2
import numpy as np
from typing import Optional, Tuple

from utils import (
    detect_aruco_markers,
    find_marker_by_id,
    calculate_pixel_to_cm_scale,
    calculate_distance_between_markers,
    draw_aruco_detections,
)


class Camera2:
    """Camera2: measure length (cm) using ArUco markers."""

    def __init__(self, camera_index: int = 1, marker_size_cm: float = 5.0,
                 aruco_dict_type: int = cv2.aruco.DICT_5X5_50):
        self.camera_index = camera_index
        self.marker_size_cm = float(marker_size_cm)
        self.aruco_dict_type = aruco_dict_type
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_opened = False

    def open(self) -> bool:
        self.cap = cv2.VideoCapture(self.camera_index)
        self.is_opened = self.cap.isOpened()
        if self.is_opened:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return self.is_opened

    def read_frame(self) -> Optional[np.ndarray]:
        if not self.is_opened or self.cap is None:
            return None
        ok, frame = self.cap.read()
        return frame if ok else None

    def measure_length(self, frame: np.ndarray,
                       reference_marker_id: int = 0,
                       length_marker_ids: Optional[Tuple[int, int]] = None,
                       use_contour_detection: bool = True) -> dict:
        """
        Compute object's length in centimeters.

        Strategy:
        - Detect ArUco markers for scale reference
        - If use_contour_detection=True: automatically detect object contour and measure longest dimension
        - Otherwise: use multiple markers to measure distance between them

        Returns dict: {'length': float or None, 'scale': float or None, 'frame': annotated, 'success': bool}
        """
        out = {'length': None, 'width': None, 'height': None, 'scale': None, 'frame': frame.copy(), 'success': False}

        corners, ids, rejected = detect_aruco_markers(frame, self.aruco_dict_type)
        if ids is None or len(ids) == 0:
            cv2.putText(out['frame'], "No ArUco marker detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return out

        out['frame'] = draw_aruco_detections(out['frame'], corners, ids)
        
        # Show detected marker IDs for debugging
        ids_text = f"Detected IDs: {ids.flatten().tolist()}"
        cv2.putText(out['frame'], ids_text, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Find largest marker for scale reference
        ref_corners, side_pixels = self._find_largest_marker(corners)
        if ref_corners is None:
            cv2.putText(out['frame'], "Marker detection failed", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            return out

        scale = side_pixels / self.marker_size_cm
        out['scale'] = scale
        if scale is None or scale <= 0:
            return out
        
        # Show scale info
        cv2.putText(out['frame'], f"Scale: {scale:.1f} px/cm", (10, frame.shape[0] - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Use contour detection to automatically find and measure objects
        if use_contour_detection:
            contour = self._detect_object_contour(frame, ref_corners)
            if contour is not None:
                # Measure using minimum area rectangle and get longest dimension as length
                box, length_cm, width_cm, height_cm = self._measure_length_from_contour(contour, scale)
                
                # Draw the bounding box
                cv2.drawContours(out['frame'], [box], 0, (0, 200, 255), 2)
                
                out['length'] = round(length_cm, 2)
                out['width'] = round(width_cm, 2)
                out['height'] = round(height_cm, 2)
                out['success'] = True
                
                # Display measurement
                cv2.putText(out['frame'], f"Length: {out['length']} cm", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                cv2.putText(out['frame'], f"(W: {out['width']}, H: {out['height']})", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                return out
            else:
                cv2.putText(out['frame'], "Object not found - adjust lighting/position", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                return out

        # Fallback: multi-marker measurement (original approach)
        if len(ids) == 1:
            cv2.putText(out['frame'], "Place object near marker or add more markers", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            return out

        # Auto-select pair if not provided: choose the two markers with the largest distance
        if length_marker_ids is None:
            length_marker_ids = self._select_longest_pair(corners, ids)

        if length_marker_ids is not None and len(length_marker_ids) == 2:
            c1 = find_marker_by_id(corners, ids, length_marker_ids[0])
            c2 = find_marker_by_id(corners, ids, length_marker_ids[1])
            if c1 is not None and c2 is not None:
                length_cm = calculate_distance_between_markers(c1, c2, scale)
                out['length'] = round(length_cm, 2) if length_cm is not None else None
                out['success'] = out['length'] is not None

        if out['length'] is not None:
            cv2.putText(out['frame'], f"Length: {out['length']} cm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        return out

    def get_length(self, reference_marker_id: int = 0,
                   length_marker_ids: Optional[Tuple[int, int]] = None,
                   display: bool = False) -> dict:
        frame = self.read_frame()
        if frame is None:
            return {'length': None, 'scale': None, 'frame': None, 'success': False}
        res = self.measure_length(frame, reference_marker_id, length_marker_ids)
        if display and res['frame'] is not None:
            cv2.imshow('Camera 2 - Length', res['frame'])
            cv2.waitKey(1)
        return res

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
        try:
            cv2.destroyAllWindows()
        except:
            pass  # Ignore errors if OpenCV GUI support is unavailable
        self.is_opened = False

    def _select_longest_pair(self, corners, ids) -> Optional[Tuple[int, int]]:
        if ids is None or len(ids) == 0:
            return None
        ids_flat = ids.flatten()
        centers = []
        for i, c in enumerate(corners):
            pts = np.array(c).reshape(-1, 2)
            cx = float(pts[:, 0].mean())
            cy = float(pts[:, 1].mean())
            centers.append((int(ids_flat[i]), cx, cy))

        best_pair = None
        best_dist = 0.0
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                id1, x1, y1 = centers[i]
                id2, x2, y2 = centers[j]
                d = np.hypot(x1 - x2, y1 - y2)
                if d > best_dist:
                    best_dist = d
                    best_pair = (id1, id2)
        return best_pair
    
    def _find_largest_marker(self, corners):
        """Find the largest ArUco marker in the detected corners."""
        if corners is None or len(corners) == 0:
            return None, None
        
        best_idx = -1
        best_side = -1.0
        
        for idx, c in enumerate(corners):
            pts = c.reshape(-1, 2)
            # Calculate average side length
            avg = 0.0
            for i in range(4):
                avg += np.linalg.norm(pts[i] - pts[(i + 1) % 4])
            avg /= 4.0
            
            if avg > best_side:
                best_side = avg
                best_idx = idx
        
        return corners[best_idx].reshape(-1, 2), best_side
    
    def _detect_object_contour(self, frame: np.ndarray, marker_pts: np.ndarray, min_area_ratio: float = 0.001):
        """Detect the main object contour excluding the ArUco marker."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Mask out the ArUco marker area
        if marker_pts is not None:
            try:
                cv2.fillPoly(thresh, [marker_pts.astype(np.int32)], 0)
            except Exception:
                pass

        contours_info = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
        
        if not contours:
            return None

        img_area = frame.shape[0] * frame.shape[1]
        filtered = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter by minimum area and exclude very large contours
            if area < max(400, img_area * min_area_ratio):
                continue
            if area > img_area * 0.9:
                continue
            filtered.append((cnt, area))

        if not filtered:
            return None

        # Return the largest valid contour
        cnt, _ = max(filtered, key=lambda item: item[1])
        return cnt
    
    def _measure_length_from_contour(self, contour, pixels_per_cm: float):
        """Measure length (longest dimension), width, and height of the contour."""
        rect = cv2.minAreaRect(contour)
        (_, _), (w_px, h_px), angle = rect
        
        # Convert pixels to cm
        width_cm = w_px / pixels_per_cm
        height_cm = h_px / pixels_per_cm
        
        # Length is the longest dimension
        length_cm = max(width_cm, height_cm)
        
        # Get the bounding box points
        box = cv2.boxPoints(rect).astype(np.int32)
        
        return box, length_cm, width_cm, height_cm


if __name__ == '__main__':
    # quick manual test
    cam = Camera2(camera_index=1, marker_size_cm=5.0)
    if not cam.open():
        print('Cannot open camera2')
    else:
        print('Camera2 opened - press q to quit')
        try:
            while True:
                r = cam.get_length(display=True)
                if r['success']:
                    print('Length:', r['length'])
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cam.close()
