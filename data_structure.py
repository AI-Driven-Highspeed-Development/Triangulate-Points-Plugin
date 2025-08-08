from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np

@dataclass
class Point2D:
    """
    Represents a 2D point in image coordinates.
    """
    x: float
    y: float
    confidence: float = 1.0
    point_id: Optional[str] = None  # Optional ID for point correspondence
    camera_name: Optional[str] = None  # Which camera this point comes from

@dataclass
class CoordinateConverter:
    """
    Handles conversion between normalized (0-1) and pixel coordinates.
    """
    width: int
    height: int
    
    def normalized_to_pixel(self, normalized_x: float, normalized_y: float) -> tuple[float, float]:
        """
        Convert normalized coordinates (0-1 range) to pixel coordinates.
        
        Args:
            normalized_x: X coordinate in range [0, 1]
            normalized_y: Y coordinate in range [0, 1]
            
        Returns:
            Tuple of (pixel_x, pixel_y)
        """
        return (normalized_x * self.width, normalized_y * self.height)
    
    def pixel_to_normalized(self, pixel_x: float, pixel_y: float) -> tuple[float, float]:
        """
        Convert pixel coordinates to normalized coordinates (0-1 range).
        
        Args:
            pixel_x: X coordinate in pixels
            pixel_y: Y coordinate in pixels
            
        Returns:
            Tuple of (normalized_x, normalized_y)
        """
        return (pixel_x / self.width, pixel_y / self.height)
    
    def convert_point2d_to_pixel(self, point: 'Point2D') -> 'Point2D':
        """
        Convert a Point2D with normalized coordinates to pixel coordinates.
        
        Args:
            point: Point2D with normalized coordinates
            
        Returns:
            New Point2D with pixel coordinates
        """
        pixel_x, pixel_y = self.normalized_to_pixel(point.x, point.y)
        return Point2D(
            x=pixel_x,
            y=pixel_y,
            confidence=point.confidence,
            point_id=point.point_id,
            camera_name=point.camera_name
        )
    
    def convert_point2d_to_normalized(self, point: 'Point2D') -> 'Point2D':
        """
        Convert a Point2D with pixel coordinates to normalized coordinates.
        
        Args:
            point: Point2D with pixel coordinates
            
        Returns:
            New Point2D with normalized coordinates
        """
        norm_x, norm_y = self.pixel_to_normalized(point.x, point.y)
        return Point2D(
            x=norm_x,
            y=norm_y,
            confidence=point.confidence,
            point_id=point.point_id,
            camera_name=point.camera_name
        )
    
    @classmethod
    def from_camera_intrinsics(cls, intrinsics: 'CameraIntrinsics') -> 'CoordinateConverter':
        """
        Create a CoordinateConverter from camera intrinsics.
        
        Args:
            intrinsics: CameraIntrinsics object
            
        Returns:
            CoordinateConverter instance
        """
        return cls(width=intrinsics.width, height=intrinsics.height)

@dataclass
class Point3D:
    """
    Represents a 3D point in world coordinates.
    """
    x: float
    y: float
    z: float
    confidence: float = 1.0
    point_id: Optional[str] = None
    reprojection_error: Optional[float] = None  # Error metric for triangulation quality

@dataclass
class CameraIntrinsics:
    """
    Simplified camera intrinsic parameters for HFOV/VFOV based triangulation.
    """
    width: int  # Image width in pixels
    height: int  # Image height in pixels
    hfov: float = 78.0  # Horizontal field of view in degrees
    vfov: float = 62.0  # Vertical field of view in degrees

@dataclass
class CameraExtrinsics:
    """
    Camera extrinsic parameters (position and orientation in world space).
    """
    position: List[float]  # [x, y, z] in world coordinates
    rotation: List[float]  # Euler angles [roll, pitch, yaw] in degrees (not radians!)
    
    def get_rotation_matrix(self) -> np.ndarray:
        """Convert Euler angles to rotation matrix."""
        # Convert degrees to radians
        roll = np.deg2rad(self.rotation[0])
        pitch = np.deg2rad(self.rotation[1])
        yaw = np.deg2rad(self.rotation[2])
        
        # Rotation matrices for each axis
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation matrix (ZYX order)
        return Rz @ Ry @ Rx
    
    def get_translation_vector(self) -> np.ndarray:
        """Get translation vector as numpy array."""
        return np.array(self.position)

@dataclass
class TriangulationCamera:
    """
    Complete camera model for triangulation with intrinsics and extrinsics.
    Renamed to avoid conflicts with webcam plugin Camera class.
    """
    name: str
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics
    
    @classmethod
    def from_config(cls, config_data) -> 'TriangulationCamera':
        """
        Create a TriangulationCamera object from configuration data.
        Supports ConfigManager generated nested config format.
        
        Args:
            config_data: Configuration object with camera data from ConfigManager
            
        Returns:
            TriangulationCamera object
        """
        # ConfigManager generates nested config objects with intrinsics and extrinsics as class instances
        intrinsics_data = config_data.intrinsics
        extrinsics_data = config_data.extrinsics
        
        intrinsics = CameraIntrinsics(
            width=intrinsics_data.width,
            height=intrinsics_data.height,
            hfov=intrinsics_data.hfov,
            vfov=intrinsics_data.vfov
        )
        
        extrinsics = CameraExtrinsics(
            position=list(extrinsics_data.position),
            rotation=list(extrinsics_data.rotation)
        )
        
        return cls(
            name=config_data.name,
            intrinsics=intrinsics,
            extrinsics=extrinsics
        )

@dataclass
class CorrespondenceSet:
    """
    A set of corresponding 2D points across multiple cameras.
    Used for triangulation input.
    """
    point_id: str
    points_2d: List[Point2D]  # One point per camera
    
    def get_cameras_names(self) -> List[str]:
        """Get list of camera names that have this point."""
        return [p.camera_name for p in self.points_2d if p.camera_name is not None]
    
    def get_point_for_camera(self, camera_name: str) -> Optional[Point2D]:
        """Get the 2D point for a specific camera."""
        for point in self.points_2d:
            if point.camera_name == camera_name:
                return point
        return None

@dataclass
class TriangulationInput:
    """
    Input data for triangulation containing cameras and point correspondences.
    """
    cameras: Dict[str, TriangulationCamera]  # Camera name -> TriangulationCamera object
    correspondence_sets: List[CorrespondenceSet]
    
    def validate(self) -> bool:
        """
        Validate that the input data is consistent.
        Returns True if valid, False otherwise.
        """
        # Check that each correspondence set has at least 2 cameras
        for corr_set in self.correspondence_sets:
            if len(corr_set.points_2d) < 2:
                return False
            
            # Check that all camera names in correspondences exist in cameras dict
            for point in corr_set.points_2d:
                if point.camera_name not in self.cameras:
                    return False
        
        return True

@dataclass
class TriangulationResult:
    """
    Result of triangulation containing 3D points and quality metrics.
    """
    points_3d: List[Point3D]
    success: bool
    error_message: Optional[str] = None
    average_reprojection_error: Optional[float] = None
    
    def get_point_by_id(self, point_id: str) -> Optional[Point3D]:
        """Get a 3D point by its ID."""
        for point in self.points_3d:
            if point.point_id == point_id:
                return point
        return None

@dataclass
class Skeleton3D:
    """
    Represents a 3D skeleton, composed of a list of 3D points for each joint.
    """
    joints: List[Point3D]
    
@dataclass
class PoseData3D:
    """
    Container for all the 3D skeletons reconstructed in a single frame.
    """
    skeletons: List[Skeleton3D]
