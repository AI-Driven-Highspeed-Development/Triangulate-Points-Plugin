import numpy as np
from typing import List, Optional, Tuple
import cv2
from managers.config_manager.config_manager import ConfigManager

from .data_structure import (
    Camera, Point2D, Point3D, CorrespondenceSet, 
    TriangulationInput, TriangulationResult
)

class TriangulatePoints:
    """
    Main class for 3D point triangulation from multiple camera views.
    """
    
    def __init__(self):
        """
        Initialize triangulation engine.
        """
        self.cm = ConfigManager()
        self.min_cameras = self.cm.config.triangulate_points_plugin.triangulation_settings.min_cameras
        self.max_reprojection_error = self.cm.config.triangulate_points_plugin.triangulation_settings.max_reprojection_error
        self.confidence_threshold = self.cm.config.triangulate_points_plugin.triangulation_settings.confidence_threshold

        self.cameras = self._load_cameras_from_config()

    def _load_cameras_from_config(self) -> dict:
        """
        Load Camera objects from configuration manager.
        
        Returns:
            Dictionary mapping camera names to Camera objects
        """
        cameras = {}
        
        # Use dot notation to access triangulate points plugin config
        if hasattr(self.cm.config, 'triangulate_points_plugin'):
            plugin_config = self.cm.config.triangulate_points_plugin
            
            if hasattr(plugin_config, 'cameras'):
                for camera_config in plugin_config.cameras:
                    camera = Camera.from_config(camera_config)
                    cameras[camera.name] = camera
        
        return cameras

    def _create_camera_dict(self, cameras: List[Camera]) -> dict:
        return {cam.name: cam for cam in cameras}

    def _validate_triangulation_input(self, triangulation_input: TriangulationInput) -> Optional[str]:
        """
        Validate triangulation input and return error message if invalid.
        
        Args:
            triangulation_input: Input to validate
            
        Returns:
            Error message if invalid, None if valid
        """
        if not triangulation_input.validate():
            return "Invalid input data"
        
        if len(triangulation_input.cameras) < self.min_cameras:
            return f"At least {self.min_cameras} cameras required"
        
        return None

    def triangulate(self, triangulation_input: TriangulationInput) -> TriangulationResult:
        """
        Triangulate 3D points from 2D correspondences across multiple cameras.
        
        Args:
            triangulation_input: Input containing cameras and point correspondences
            
        Returns:
            TriangulationResult with 3D points and quality metrics
        """
        # Validate input
        error_message = self._validate_triangulation_input(triangulation_input)
        if error_message:
            return TriangulationResult(
                points_3d=[],
                success=False,
                error_message=error_message
            )
        
        triangulated_points, reprojection_errors = self._process_correspondence_sets(
            triangulation_input.correspondence_sets, 
            triangulation_input.cameras
        )
        
        avg_error = np.mean(reprojection_errors) if reprojection_errors else None
        
        return TriangulationResult(
            points_3d=triangulated_points,
            success=len(triangulated_points) > 0,
            error_message=None if triangulated_points else "No points could be triangulated",
            average_reprojection_error=avg_error
        )
    
    def _process_correspondence_sets(
        self, 
        correspondence_sets: List[CorrespondenceSet], 
        cameras: dict
    ) -> Tuple[List[Point3D], List[float]]:
        """
        Process correspondence sets to triangulate points and calculate errors.
        
        Args:
            correspondence_sets: List of correspondence sets to process
            cameras: Dictionary of camera objects
            
        Returns:
            Tuple of (triangulated_points, reprojection_errors)
        """
        triangulated_points = []
        reprojection_errors = []
        
        for corr_set in correspondence_sets:
            try:
                point_3d = self._triangulate_single_point(corr_set, cameras)
                if point_3d is not None:
                    # Calculate reprojection error
                    error = self._calculate_reprojection_error(point_3d, corr_set, cameras)
                    
                    if error <= self.max_reprojection_error:
                        point_3d.reprojection_error = error
                        triangulated_points.append(point_3d)
                        reprojection_errors.append(error)
                        
            except Exception as e:
                # Skip this point if triangulation fails
                continue
        
        return triangulated_points, reprojection_errors
    
    def _create_point3d_from_homogeneous(
        self,
        homogeneous_coords: np.ndarray,
        confidence: float,
        point_id: str
    ) -> Optional[Point3D]:
        """
        Create Point3D from homogeneous coordinates with validation.
        
        Args:
            homogeneous_coords: 4D homogeneous coordinates [x, y, z, w]
            confidence: Confidence value for the point
            point_id: Identifier for the point
            
        Returns:
            Point3D object or None if invalid coordinates
        """
        if len(homogeneous_coords) < 4 or homogeneous_coords[3] == 0:
            return None
            
        # Convert from homogeneous coordinates
        x = homogeneous_coords[0] / homogeneous_coords[3]
        y = homogeneous_coords[1] / homogeneous_coords[3]
        z = homogeneous_coords[2] / homogeneous_coords[3]
        
        return Point3D(
            x=float(x),
            y=float(y),
            z=float(z),
            confidence=confidence,
            point_id=point_id
        )
    
    def _calculate_average_confidence(self, points_2d: List[Point2D]) -> float:
        """
        Calculate average confidence from a list of 2D points.
        
        Args:
            points_2d: List of 2D points
            
        Returns:
            Average confidence value
        """
        return float(np.mean([p.confidence for p in points_2d]))
    
    def _triangulate_single_point(
        self, 
        corr_set: CorrespondenceSet, 
        cameras: dict
    ) -> Optional[Point3D]:
        """
        Triangulate a single 3D point from its 2D correspondences.
        
        Args:
            corr_set: Correspondence set containing 2D points across cameras
            cameras: Dictionary of camera objects
            
        Returns:
            Triangulated 3D point or None if triangulation fails
        """
        if len(corr_set.points_2d) < self.min_cameras:
            return None
        
        # Use OpenCV triangulation for robust results
        if len(corr_set.points_2d) == 2:
            return self._triangulate_two_view(corr_set, cameras)
        else:
            return self._triangulate_multi_view(corr_set, cameras)
    
    def _triangulate_two_view(
        self, 
        corr_set: CorrespondenceSet, 
        cameras: dict
    ) -> Optional[Point3D]:
        """
        Triangulate using two cameras with OpenCV.
        """
        if len(corr_set.points_2d) != 2:
            return None
        
        point1, point2 = corr_set.points_2d
        camera1 = cameras[point1.camera_name]
        camera2 = cameras[point2.camera_name]
        
        # Get projection matrices
        P1 = camera1.get_projection_matrix()
        P2 = camera2.get_projection_matrix()
        
        # Prepare points for OpenCV (must be arrays)
        pts1 = np.array([[point1.x, point1.y]], dtype=np.float32)
        pts2 = np.array([[point2.x, point2.y]], dtype=np.float32)
        
        # Triangulate using OpenCV
        points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        
        # Calculate confidence as average of input confidences
        confidence = self._calculate_average_confidence([point1, point2])
        
        # Convert from homogeneous coordinates and create Point3D
        return self._create_point3d_from_homogeneous(
            points_4d[:, 0], confidence, corr_set.point_id
        )
    
    def _triangulate_multi_view(
        self, 
        corr_set: CorrespondenceSet, 
        cameras: dict
    ) -> Optional[Point3D]:
        """
        Triangulate using multiple cameras with DLT (Direct Linear Transform).
        """
        if len(corr_set.points_2d) < 2:
            return None
        
        # Build the linear system Ax = 0 for DLT
        A = []
        
        for point_2d in corr_set.points_2d:
            camera = cameras[point_2d.camera_name]
            P = camera.get_projection_matrix()
            
            x, y = point_2d.x, point_2d.y
            
            # Each point contributes 2 equations
            A.append(x * P[2, :] - P[0, :])  # x*P[2] - P[0] = 0
            A.append(y * P[2, :] - P[1, :])  # y*P[2] - P[1] = 0
        
        A = np.array(A)
        
        # Solve using SVD
        try:
            _, _, V = np.linalg.svd(A)
            X = V[-1, :]  # Last column of V
            
            # Calculate confidence as average of input confidences
            confidence = self._calculate_average_confidence(corr_set.points_2d)
            
            # Convert from homogeneous coordinates and create Point3D
            return self._create_point3d_from_homogeneous(X, confidence, corr_set.point_id)
            
        except np.linalg.LinAlgError:
            return None
    
    def _calculate_reprojection_error(
        self, 
        point_3d: Point3D, 
        corr_set: CorrespondenceSet, 
        cameras: dict
    ) -> float:
        """
        Calculate reprojection error for a triangulated point.
        
        Args:
            point_3d: Triangulated 3D point
            corr_set: Original 2D correspondences
            cameras: Camera objects
            
        Returns:
            Average reprojection error in pixels
        """
        errors = []
        
        for point_2d in corr_set.points_2d:
            camera = cameras[point_2d.camera_name]
            
            # Project 3D point back to this camera
            projected = camera.project_3d_to_2d(point_3d)
            
            # Calculate Euclidean distance
            error = np.sqrt(
                (projected.x - point_2d.x) ** 2 + 
                (projected.y - point_2d.y) ** 2
            )
            errors.append(error)
        
        return float(np.mean(errors))
    
    def triangulate_ordered_points(
        self, 
        cameras: List[Camera], 
        points_per_camera: List[List[Point2D]]
    ) -> TriangulationResult:
        """
        Convenience method for triangulating ordered points.
        Assumes points are in the same order across all cameras.
        
        Args:
            cameras: List of camera objects
            points_per_camera: List of point lists, one per camera
            
        Returns:
            TriangulationResult with triangulated points
        """
        # Validate input
        if len(cameras) != len(points_per_camera):
            return TriangulationResult(
                points_3d=[],
                success=False,
                error_message="Number of cameras must match number of point lists"
            )
        
        # Find minimum number of points across all cameras
        min_points = min(len(points) for points in points_per_camera)
        
        if min_points == 0:
            return TriangulationResult(
                points_3d=[],
                success=False,
                error_message="No points provided"
            )
        
        # Create camera dictionary
        cameras_dict = self._create_camera_dict(cameras)
        
        # Create correspondence sets
        correspondence_sets = self._create_correspondence_sets_from_ordered_points(
            cameras, points_per_camera, min_points
        )
        
        # Create triangulation input
        triangulation_input = TriangulationInput(
            cameras=cameras_dict,
            correspondence_sets=correspondence_sets
        )
        
        return self.triangulate(triangulation_input)
    
    def _create_correspondence_sets_from_ordered_points(
        self,
        cameras: List[Camera],
        points_per_camera: List[List[Point2D]],
        min_points: int
    ) -> List[CorrespondenceSet]:
        """
        Create correspondence sets from ordered points across cameras.
        
        Args:
            cameras: List of camera objects
            points_per_camera: List of point lists, one per camera
            min_points: Minimum number of points to process
            
        Returns:
            List of correspondence sets
        """
        correspondence_sets = []
        
        for i in range(min_points):
            points_2d = []
            
            for j, camera in enumerate(cameras):
                if i < len(points_per_camera[j]):
                    point = points_per_camera[j][i]
                    point.camera_name = camera.name
                    point.point_id = f"point_{i}"
                    points_2d.append(point)
            
            if len(points_2d) >= self.min_cameras:
                correspondence_sets.append(CorrespondenceSet(
                    point_id=f"point_{i}",
                    points_2d=points_2d
                ))
        
        return correspondence_sets
    
    def triangulate_from_yolo_poses(
        self, 
        cameras: List[Camera], 
        pose_data_per_camera: List[any]  # List of PoseData objects from YOLO
    ) -> TriangulationResult:
        """
        Convenience method for triangulating from YOLO pose data.
        Assumes each camera has detected the same person (first skeleton).
        
        Args:
            cameras: List of camera objects
            pose_data_per_camera: List of PoseData from YOLO pose plugin
            
        Returns:
            TriangulationResult with triangulated joint positions
        """
        # Extract joint points from first skeleton in each camera
        points_per_camera = self._convert_yolo_poses_to_points(cameras, pose_data_per_camera)
        
        return self.triangulate_ordered_points(cameras, points_per_camera)

    def _convert_yolo_poses_to_points(
        self,
        cameras: List[Camera],
        pose_data_per_camera: List[any]
    ) -> List[List[Point2D]]:
        """
        Convert YOLO pose data to Point2D lists for each camera.
        
        Args:
            cameras: List of camera objects
            pose_data_per_camera: List of PoseData from YOLO pose plugin
            
        Returns:
            List of Point2D lists, one per camera
        """
        points_per_camera = []
        
        for i, pose_data in enumerate(pose_data_per_camera):
            camera_points = []
            
            if pose_data and pose_data.skeletons:
                skeleton = pose_data.skeletons[0]  # Use first detected person
                
                for joint in skeleton.joints:
                    # Convert YOLO joint to Point2D
                    point_2d = Point2D(
                        x=joint.x * cameras[i].intrinsics.width,  # Convert from normalized
                        y=joint.y * cameras[i].intrinsics.height,
                        confidence=joint.confidence,
                        point_id=joint.label,
                        camera_name=cameras[i].name
                    )
                    camera_points.append(point_2d)
            
            points_per_camera.append(camera_points)
        
        return points_per_camera


def load_cameras_from_config(config_manager) -> dict:
    """
    Utility function to load Camera objects from configuration manager.
    
    Args:
        config_manager: ConfigManager instance
        
    Returns:
        Dictionary mapping camera names to Camera objects
    """
    cameras = {}
    
    # Use dot notation to access triangulate points plugin config
    if hasattr(config_manager.config, 'triangulate_points_plugin'):
        plugin_config = config_manager.config.triangulate_points_plugin
        
        if hasattr(plugin_config, 'cameras'):
            for camera_config in plugin_config.cameras:
                camera = Camera.from_config(camera_config)
                cameras[camera.name] = camera
    
    return cameras