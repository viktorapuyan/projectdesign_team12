"""
Camera 1 module: Measures object width and height using ArUco markers.

This module exposes a `Camera1` class with methods to open the camera,
capture frames, and measure width/height in centimeters using ArUco
markers detected via `utils.py`.
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


class Camera1:
    """Camera1: measure width and height (cm) using ArUco markers."""

    def __init__(self, camera_index: int = 0, marker_size_cm: float = 5.0,
                 aruco_dict_type: int = cv2.aruco.DICT_5X5_50):
        """
        Args:
            camera_index: integer OpenCV camera index
            marker_size_cm: known physical size of the reference marker (cm)
            aruco_dict_type: predefined ArUco dictionary constant
        """
        self.camera_index = camera_index
        self.marker_size_cm = float(marker_size_cm)
        self.aruco_dict_type = aruco_dict_type
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_opened = False

    def open(self) -> bool:
        """Open the video capture device."""
        self.cap = cv2.VideoCapture(self.camera_index)
        self.is_opened = self.cap.isOpened()
        if self.is_opened:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return self.is_opened

    def read_frame(self) -> Optional[np.ndarray]:
        """Read a single frame from the camera (BGR) or return None."""
        if not self.is_opened or self.cap is None:
            return None
        ok, frame = self.cap.read()
        return frame if ok else None

    def measure_dimensions(self, frame: np.ndarray,
                           reference_marker_id: int = 0,
                           width_marker_ids: Optional[Tuple[int, int]] = None,
                           height_marker_ids: Optional[Tuple[int, int]] = None,
                           use_contour_detection: bool = True) -> dict:
        """
        Measure object width and height in cm.

        Strategy:
         - Detect ArUco markers for scale reference
         - If use_contour_detection=True: automatically detect object contour and measure
         - Otherwise: use multiple markers to measure distance between them

        Returns a dict with keys: width, height, scale, frame (annotated), success.
        """
        out = {'width': None, 'height': None, 'scale': None, 'frame': frame.copy(), 'success': False}

        corners, ids, rejected = detect_aruco_markers(frame, self.aruco_dict_type)
        if ids is None or len(ids) == 0:
            cv2.putText(out['frame'], "No markers detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return out

        out['frame'] = draw_aruco_detections(out['frame'], corners, ids)
        
        # Show detected marker IDs for debugging
        ids_text = f"Detected IDs: {ids.flatten().tolist()}"
        cv2.putText(out['frame'], ids_text, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Find largest marker for scale reference
        ref_corners, side_pixels = self._find_largest_marker(corners)
        if ref_corners is None:
            cv2.putText(out['frame'], f"Marker detection failed", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
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
                # Measure using minimum area rectangle
                box, width_cm, height_cm, angle = self._measure_contour(contour, scale)
                
                # Draw the bounding box
                cv2.drawContours(out['frame'], [box], 0, (0, 255, 0), 2)
                
                out['width'] = round(width_cm, 2)
                out['height'] = round(height_cm, 2)
                out['success'] = True
                
                # Display measurements
                cv2.putText(out['frame'], f"Width: {out['width']} cm", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(out['frame'], f"Height: {out['height']} cm", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(out['frame'], f"Angle: {angle:.1f}°", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
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

        # auto-select pairs if not provided
        if width_marker_ids is None or height_marker_ids is None:
            w_pair, h_pair = self._auto_select_marker_pairs(corners, ids)
            if width_marker_ids is None:
                width_marker_ids = w_pair
            if height_marker_ids is None:
                height_marker_ids = h_pair

        # measure width
        if width_marker_ids is not None and len(width_marker_ids) == 2:
            c1 = find_marker_by_id(corners, ids, width_marker_ids[0])
            c2 = find_marker_by_id(corners, ids, width_marker_ids[1])
            if c1 is not None and c2 is not None:
                w_cm = calculate_distance_between_markers(c1, c2, scale)
                out['width'] = round(w_cm, 2) if w_cm is not None else None

        # measure height
        if height_marker_ids is not None and len(height_marker_ids) == 2:
            c1 = find_marker_by_id(corners, ids, height_marker_ids[0])
            c2 = find_marker_by_id(corners, ids, height_marker_ids[1])
            if c1 is not None and c2 is not None:
                h_cm = calculate_distance_between_markers(c1, c2, scale)
                out['height'] = round(h_cm, 2) if h_cm is not None else None

        out['success'] = out['width'] is not None or out['height'] is not None

        if out['width'] is not None:
            cv2.putText(out['frame'], f"Width: {out['width']} cm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if out['height'] is not None:
            cv2.putText(out['frame'], f"Height: {out['height']} cm", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return out

    def get_width_and_height(self, reference_marker_id: int = 0,
                             width_marker_ids: Optional[Tuple[int, int]] = None,
                             height_marker_ids: Optional[Tuple[int, int]] = None,
                             display: bool = False) -> dict:
        """Capture a frame and return measurement dict from `measure_dimensions`."""
        frame = self.read_frame()
        if frame is None:
            return {'width': None, 'height': None, 'scale': None, 'frame': None, 'success': False}
        res = self.measure_dimensions(frame, reference_marker_id, width_marker_ids, height_marker_ids)
        if display and res['frame'] is not None:
            cv2.imshow('Camera 1 - Width & Height', res['frame'])
            cv2.waitKey(1)
        return res

    def close(self) -> None:
        """Release resources."""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        self.is_opened = False

    def _auto_select_marker_pairs(self, corners, ids):
        """Return (w1,w2),(h1,h2) selected from detected markers by center separations."""
        if ids is None or len(ids) == 0:
            return None, None
        ids_flat = ids.flatten()
        centers = []
        for i, c in enumerate(corners):
            pts = np.array(c).reshape(-1, 2)
            cx = float(pts[:, 0].mean())
            cy = float(pts[:, 1].mean())
            centers.append((int(ids_flat[i]), cx, cy))

        w_pair = None
        h_pair = None
        max_dx = 0
        max_dy = 0
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                id1, x1, y1 = centers[i]
                id2, x2, y2 = centers[j]
                dx = abs(x1 - x2)
                dy = abs(y1 - y2)
                if dx > max_dx:
                    max_dx = dx
                    w_pair = (id1, id2)
                if dy > max_dy:
                    max_dy = dy
                    h_pair = (id1, id2)
        return w_pair, h_pair
    
    def _get_marker_size_pixels(self, marker_corners: np.ndarray) -> float:
        """Get average side length of marker in pixels."""
        pts = np.array(marker_corners).reshape(-1, 2)
        d1 = np.linalg.norm(pts[0] - pts[1])
        d2 = np.linalg.norm(pts[1] - pts[2])
        return float((d1 + d2) / 2.0)
    
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
    
    def _measure_contour(self, contour, pixels_per_cm: float):
        """Measure width and height of the contour using minimum area rectangle."""
        rect = cv2.minAreaRect(contour)
        (center_x, center_y), (w_px, h_px), angle = rect
        
        # Convert pixels to cm
        width_cm = w_px / pixels_per_cm
        height_cm = h_px / pixels_per_cm
        
        # Get the bounding box points
        box = cv2.boxPoints(rect).astype(np.int32)
        
        # Determine which dimension is width vs height
        # Return longer dimension as width, shorter as height
        if width_cm > height_cm:
            actual_width = width_cm
            actual_height = height_cm
        else:
            actual_width = height_cm
            actual_height = width_cm
        
        return box, actual_width, actual_height, angle


def _test():
    cam = Camera1(0, marker_size_cm=5.0)
    if not cam.open():
        print('Cannot open camera')
        return
    print('Press q to quit')
    try:
        while True:
            res = cam.get_width_and_height(display=True)
            if res['success']:
                print('W:', res['width'], 'H:', res['height'])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cam.close()


if __name__ == '__main__':
    _test()
