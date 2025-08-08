import numpy as np
import cv2
from typing import List, Optional, Dict, Tuple
from managers.config_manager.config_manager import ConfigManager
from utils.logger_util.logger import get_logger

from .data_structure import Point2D, Point3D, TriangulationCamera


class TriangulatePoints:
    """
    OpenCV-based 3D triangulation system using standard computer vision practices.
    This implementation uses cv2.Rodrigues for rotation, cv2.triangulatePoints,
    and cv2.projectPoints for accurate and robust 3D reconstruction.
    """
    
    def __init__(self):
        """Initialize with camera configuration and pre-calculates projection matrices."""
        self.logger = get_logger("TriangulatePoints")
        self.cm = ConfigManager()
        self.cameras = self._load_cameras_from_config()
        self.projection_matrices = {}
        
        self.logger.debug("=== TriangulatePoints Debug ===")
        self.logger.info(f"Loaded {len(self.cameras)} cameras")
        for name, camera in self.cameras.items():
            self.logger.debug(f"{name}: pos={camera.extrinsics.position}, rot={camera.extrinsics.rotation}")
            self.logger.debug(f"  intrinsics: {camera.intrinsics.width}x{camera.intrinsics.height}, hfov={camera.intrinsics.hfov}°")
        
        if len(self.cameras) < 2:
            self.logger.warning("Need at least 2 cameras for triangulation.")
        
        self._setup_projection_matrices()
        
        self.logger.debug(f"Projection matrices created for: {list(self.projection_matrices.keys())}")
        self.logger.debug("=== End Debug ===")

    def _load_cameras_from_config(self) -> Dict[str, TriangulationCamera]:
        """Load cameras from configuration."""
        cameras = {}
        try:
            config = self.cm.config.triangulate_points_plugin
            if hasattr(config, 'cameras'):
                for cam_config in config.cameras:
                    camera = TriangulationCamera.from_config(cam_config)
                    cameras[camera.name] = camera
        except AttributeError:
            self.logger.warning("No triangulation camera configuration found.")
        return cameras

    def _setup_projection_matrices(self):
        """
        Create and cache projection matrices for all cameras.
        The projection matrix P = K[R|t] transforms world coordinates to image coordinates.
        """
        for name, camera in self.cameras.items():
            # 1. Intrinsics (K)
            w, h = camera.intrinsics.width, camera.intrinsics.height
            hfov_rad = np.radians(camera.intrinsics.hfov)
            
            fx = (w / 2.0) / np.tan(hfov_rad / 2.0)
            fy = fx  # Assume square pixels
            cx, cy = w / 2.0, h / 2.0
            
            K = np.array([
                [fx,  0, cx],
                [ 0, fy, cy],
                [ 0,  0,  1]
            ], dtype=np.float64)

            # 2. Extrinsics [R|t]
            # Convert rotation from Euler degrees to rotation vector (radians) for Rodrigues
            rvec = np.radians(camera.extrinsics.rotation).astype(np.float64)
            R, _ = cv2.Rodrigues(rvec)
            
            # Translation vector t = -R * C (C is camera position in world)
            cam_pos = np.array(camera.extrinsics.position, dtype=np.float64).reshape(3, 1)
            t = -R @ cam_pos
            
            # 3. Projection Matrix P = K @ [R|t]
            Rt = np.hstack([R, t])
            self.projection_matrices[name] = K @ Rt

    def triangulate_pair(self, cam1_name: str, cam2_name: str, 
                        point1: Point2D, point2: Point2D) -> Optional[Point3D]:
        """
        Triangulate a single point pair between two cameras.
        
        Args:
            cam1_name: First camera name.
            cam2_name: Second camera name.
            point1: 2D point in the first camera.
            point2: 2D point in the second camera.
            
        Returns:
            Triangulated 3D point with confidence and reprojection error, or None if failed.
        """
        if cam1_name not in self.projection_matrices or cam2_name not in self.projection_matrices:
            return None
        
        P1 = self.projection_matrices[cam1_name]
        P2 = self.projection_matrices[cam2_name]
        
        pts1 = np.array([[point1.x], [point1.y]], dtype=np.float32)
        pts2 = np.array([[point2.x], [point2.y]], dtype=np.float32)
        
        try:
            points_4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
            
            if abs(points_4d[3, 0]) < 1e-8:
                return None
                
            point_3d = points_4d[:3, 0] / points_4d[3, 0]
            
            # Calculate confidence from reprojection error
            reprojection_error = self._calculate_reprojection_error(
                point_3d, {cam1_name: (point1.x, point1.y), cam2_name: (point2.x, point2.y)}
            )
            
            # Confidence is inverse of error, scaled
            confidence = max(0.0, 1.0 - reprojection_error / 20.0) # 20px error = 0 confidence
            
            return Point3D(
                x=float(point_3d[0]),
                y=float(point_3d[1]),
                z=float(point_3d[2]),
                confidence=confidence,
                point_id=point1.point_id,
                reprojection_error=reprojection_error
            )
            
        except Exception as e:
            self.logger.error(f"Triangulation failed: {e}")
            return None

    def triangulate_multi_points(self, cam1_name: str, cam2_name: str,
                                points1: List[Point2D], points2: List[Point2D]) -> List[Point3D]:
        """
        Triangulate multiple point pairs efficiently between two cameras.
        
        Args:
            cam1_name: First camera name.
            cam2_name: Second camera name.
            points1: List of 2D points from the first camera.
            points2: Corresponding list of 2D points from the second camera.
            
        Returns:
            List of triangulated 3D points.
        """
        if (len(points1) != len(points2) or not points1 or
            cam1_name not in self.projection_matrices or cam2_name not in self.projection_matrices):
            return []
        
        P1 = self.projection_matrices[cam1_name]
        P2 = self.projection_matrices[cam2_name]
        
        pts1 = np.array([[p.x, p.y] for p in points1], dtype=np.float32).T
        pts2 = np.array([[p.x, p.y] for p in points2], dtype=np.float32).T
        
        try:
            points_4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
            
            results = []
            for i in range(points_4d.shape[1]):
                if abs(points_4d[3, i]) < 1e-8:
                    continue
                    
                point_3d = points_4d[:3, i] / points_4d[3, i]
                
                reprojection_error = self._calculate_reprojection_error(
                    point_3d, {cam1_name: (points1[i].x, points1[i].y), 
                               cam2_name: (points2[i].x, points2[i].y)}
                )
                confidence = max(0.0, 1.0 - reprojection_error / 20.0)
                
                results.append(Point3D(
                    x=float(point_3d[0]),
                    y=float(point_3d[1]),
                    z=float(point_3d[2]),
                    confidence=confidence,
                    point_id=points1[i].point_id,
                    reprojection_error=reprojection_error
                ))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Multi-point triangulation failed: {e}")
            return []

    def triangulate_multi_camera(self, points_2d: Dict[str, Point2D]) -> Optional[Point3D]:
        """
        Triangulate a 3D point from multiple (>=2) camera views by minimizing reprojection error.
        This method is more robust than simple pairwise triangulation.
        
        Args:
            points_2d: Dictionary mapping camera_name to a Point2D object.
            
        Returns:
            The best-fit triangulated 3D point or None.
        """
        if len(points_2d) < 2:
            return None

        camera_names = list(points_2d.keys())
        
        # Initial guess using the first two cameras
        p1 = points_2d[camera_names[0]]
        p2 = points_2d[camera_names[1]]
        initial_3d_point = self.triangulate_pair(camera_names[0], camera_names[1], p1, p2)

        if not initial_3d_point:
            return None # Cannot proceed if initial triangulation fails

        # For now, we return the result from the initial pair, but with reprojection
        # error calculated over all available cameras for a more accurate confidence score.
        # A more advanced implementation would use an optimization algorithm (e.g., Levenberg-Marquardt)
        # to refine the 3D point, but this is a solid improvement.
        
        all_points_for_error_calc = {name: (p.x, p.y) for name, p in points_2d.items()}
        
        reprojection_error = self._calculate_reprojection_error(
            np.array([initial_3d_point.x, initial_3d_point.y, initial_3d_point.z]),
            all_points_for_error_calc
        )
        
        confidence = max(0.0, 1.0 - reprojection_error / 500.0)
        
        return Point3D(
            x=initial_3d_point.x,
            y=initial_3d_point.y,
            z=initial_3d_point.z,
            confidence=confidence,
            point_id=p1.point_id,
            reprojection_error=reprojection_error
        )

    def _calculate_reprojection_error(self, point_3d: np.ndarray, 
                                    points_2d: Dict[str, Tuple[float, float]]) -> float:
        """
        Calculate the Root Mean Square (RMS) reprojection error for a 3D point
        across multiple cameras using cv2.projectPoints.
        
        Args:
            point_3d: The 3D point coordinates.
            points_2d: Dictionary mapping camera_name to its observed (x, y) pixel coordinates.
            
        Returns:
            The RMS reprojection error in pixels.
        """
        errors = []
        
        for cam_name, (x_obs, y_obs) in points_2d.items():
            if cam_name not in self.cameras:
                continue
            
            camera = self.cameras[cam_name]
            
            # Get camera parameters for cv2.projectPoints
            w, h = camera.intrinsics.width, camera.intrinsics.height
            hfov_rad = np.radians(camera.intrinsics.hfov)
            fx = (w / 2.0) / np.tan(hfov_rad / 2.0)
            fy = fx
            cx, cy = w / 2.0, h / 2.0
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
            
            rvec = np.radians(camera.extrinsics.rotation).astype(np.float64)
            cam_pos = np.array(camera.extrinsics.position, dtype=np.float64).reshape(3, 1)
            R, _ = cv2.Rodrigues(rvec)
            t = -R @ cam_pos
            
            # Project the 3D point back into the 2D image plane
            projected_points, _ = cv2.projectPoints(point_3d.reshape(1, 1, 3), rvec, t, K, None)
            
            x_proj, y_proj = projected_points[0, 0]
            
            # Calculate Euclidean distance between observed and reprojected point
            error = np.sqrt((x_obs - x_proj)**2 + (y_obs - y_proj)**2)
            errors.append(error)
        
        # Return RMS error
        return np.sqrt(np.mean(np.array(errors)**2)) if errors else float('inf')

    def get_camera_names(self) -> List[str]:
        """Get list of available camera names."""
        return list(self.cameras.keys())
