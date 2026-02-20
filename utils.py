import cv2
import numpy as np
from typing import Optional, Tuple, List

# Type aliases
Corners = List[np.ndarray]  # list of marker corner arrays
Ids = Optional[np.ndarray]  # array of ids or None


def _get_detector(aruco_dict_type: int):
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    # Use new Detector API when available, otherwise fall back to detectMarkers
    try:
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        return detector
    except Exception:
        return (aruco_dict, cv2.aruco.DetectorParameters_create())


def detect_aruco_markers(frame: np.ndarray, aruco_dict_type: int = cv2.aruco.DICT_5X5_100) -> Tuple[Corners, Ids, List]:
    """
    Detect ArUco markers in a BGR frame.

    Returns:
        corners: list of corner arrays (each is Nx2 or 1x4x2 depending on API)
        ids: numpy array of ids (shape (N,1)) or None
        rejected: rejected candidates (may be empty)
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    try:
        detector = _get_detector(aruco_dict_type)
        if isinstance(detector, cv2.aruco.ArucoDetector):
            corners, ids, rejected = detector.detectMarkers(gray)
        else:
            aruco_dict, params = detector
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)
    except Exception:
        # fallback to basic detectMarkers
        aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        params = cv2.aruco.DetectorParameters_create()
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)

    return corners, ids, rejected


def find_marker_by_id(corners: Corners, ids: Ids, target_id: int) -> Optional[np.ndarray]:
    """
    Return the corners for a specific marker ID, or None if not found.
    """
    if ids is None or len(ids) == 0:
        return None
    ids_flat = ids.flatten()
    for i, mid in enumerate(ids_flat):
        if int(mid) == int(target_id):
            return corners[i]
    return None


def get_marker_center(marker_corners: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Compute the (x, y) center of a single marker's corners.
    Accepts corners shaped (4,2) or (1,4,2).
    """
    if marker_corners is None:
        return None
    pts = np.array(marker_corners).reshape(-1, 2)
    cx = float(pts[:, 0].mean())
    cy = float(pts[:, 1].mean())
    return cx, cy


def _marker_diagonal_pixels(marker_corners: np.ndarray) -> float:
    """
    Approximate marker size in pixels by averaging the two side lengths.
    """
    pts = np.array(marker_corners).reshape(-1, 2)
    d1 = np.linalg.norm(pts[0] - pts[1])
    d2 = np.linalg.norm(pts[1] - pts[2])
    return float((d1 + d2) / 2.0)


def calculate_pixel_to_cm_scale(marker_corners: np.ndarray, marker_size_cm: float) -> Optional[float]:
    """
    Compute pixels-per-cm using one detected marker's corners and known marker size (cm).

    Args:
        marker_corners: corners for a single marker (from detect_aruco_markers or find_marker_by_id)
        marker_size_cm: real-world side length of the marker in centimeters

    Returns:
        pixels_per_cm (float) or None if computation fails
    """
    if marker_corners is None:
        return None
    avg_pixels = _marker_diagonal_pixels(marker_corners)
    if avg_pixels <= 0 or marker_size_cm <= 0:
        return None
    return avg_pixels / float(marker_size_cm)


def calculate_distance_between_markers(corners1: np.ndarray, corners2: np.ndarray, pixels_per_cm: float) -> Optional[float]:
    """
    Calculate real-world distance (cm) between the centers of two markers.

    Args:
        corners1, corners2: marker corner arrays
        pixels_per_cm: scale produced by calculate_pixel_to_cm_scale

    Returns:
        distance in centimeters or None
    """
    if corners1 is None or corners2 is None or pixels_per_cm is None or pixels_per_cm == 0:
        return None
    c1 = np.array(get_marker_center(corners1))
    c2 = np.array(get_marker_center(corners2))
    px_dist = float(np.linalg.norm(c1 - c2))
    return px_dist / pixels_per_cm


def draw_aruco_detections(frame: np.ndarray, corners: Corners, ids: Ids) -> np.ndarray:
    """
    Draw detected markers and ids on a copy of the frame and return it.
    """
    out = frame.copy()
    if ids is not None and len(ids) > 0:
        try:
            cv2.aruco.drawDetectedMarkers(out, corners, ids)
        except Exception:
            # draw manually if drawDetectedMarkers not available
            for i, c in enumerate(corners):
                pts = np.array(c).reshape(-1, 2).astype(int)
                cv2.polylines(out, [pts], True, (0, 255, 0), 2)
                if ids is not None:
                    cv2.putText(out, f"id={int(ids.flatten()[i])}", (int(pts[0][0]), int(pts[0][1]) - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return out


def estimate_pose_distance(marker_corners: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray,
                           marker_size_cm: float) -> Optional[float]:
    """
    Estimate camera-to-marker distance using solvePnP/estimatePoseSingleMarkers.
    Returns distance in centimeters (z component of translation vector) or None.
    Requires camera calibration.

    Args:
        marker_corners: corners for a single marker
        camera_matrix: 3x3 camera matrix
        dist_coeffs: distortion coefficients
        marker_size_cm: marker size in centimeters
    """
    if marker_corners is None or camera_matrix is None:
        return None
    try:
        # convert marker size to meters for OpenCV pose functions
        marker_size_m = float(marker_size_cm) / 100.0
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers([marker_corners], marker_size_m, camera_matrix, dist_coeffs)
        if tvecs is None or len(tvecs) == 0:
            return None
        # z is in meters
        z_m = float(tvecs[0][0][2])
        return z_m * 100.0
    except Exception:
        return None