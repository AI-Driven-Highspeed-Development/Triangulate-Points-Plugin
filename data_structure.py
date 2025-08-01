from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Union
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
    Camera intrinsic parameters.
    """
    fx: float  # Focal length in x
    fy: float  # Focal length in y
    cx: float  # Principal point x
    cy: float  # Principal point y
    width: int  # Image width in pixels
    height: int  # Image height in pixels
    hfov: float = 78.0  # Horizontal field of view in degrees
    vfov: float = 62.0  # Vertical field of view in degrees
    k1: float = 0.0  # Radial distortion coefficient 1
    k2: float = 0.0  # Radial distortion coefficient 2
    p1: float = 0.0  # Tangential distortion coefficient 1
    p2: float = 0.0  # Tangential distortion coefficient 2

    @classmethod
    def from_fov(cls, hfov_deg: float, vfov_deg: float, width: int, height: int) -> 'CameraIntrinsics':
        """
        Create camera intrinsics from field of view angles.
        
        Args:
            hfov_deg: Horizontal field of view in degrees
            vfov_deg: Vertical field of view in degrees
            width: Image width in pixels
            height: Image height in pixels
        """
        hfov_rad = np.deg2rad(hfov_deg)
        vfov_rad = np.deg2rad(vfov_deg)
        
        fx = (width / 2.0) / np.tan(hfov_rad / 2.0)
        fy = (height / 2.0) / np.tan(vfov_rad / 2.0)
        cx = width / 2.0
        cy = height / 2.0
        
        return cls(
            fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height,
            hfov=hfov_deg, vfov=vfov_deg
        )
    
    def update_focal_from_fov(self):
        """
        Update focal length parameters from stored FOV values.
        Useful when FOV values are modified after creation.
        """
        hfov_rad = np.deg2rad(self.hfov)
        vfov_rad = np.deg2rad(self.vfov)
        
        self.fx = (self.width / 2.0) / np.tan(hfov_rad / 2.0)
        self.fy = (self.height / 2.0) / np.tan(vfov_rad / 2.0)

@dataclass
class CameraExtrinsics:
    """
    Camera extrinsic parameters (position and orientation in world space).
    """
    position: Tuple[float, float, float]  # (x, y, z) in world coordinates
    rotation: Tuple[float, float, float]  # Euler angles (roll, pitch, yaw) in radians
    
    def get_rotation_matrix(self) -> np.ndarray:
        """Convert Euler angles to rotation matrix."""
        roll, pitch, yaw = self.rotation
        
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
class Camera:
    """
    Complete camera model with intrinsics and extrinsics.
    """
    name: str
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics
    
    @classmethod
    def from_config(cls, config_data: dict) -> 'Camera':
        """
        Create a Camera object from configuration data.
        
        Args:
            config_data: Dictionary containing camera configuration
            
        Returns:
            Camera object
        """
        intrinsics_data = config_data['intrinsics']
        extrinsics_data = config_data['extrinsics']
        
        intrinsics = CameraIntrinsics(
            fx=intrinsics_data['fx'],
            fy=intrinsics_data['fy'],
            cx=intrinsics_data['cx'],
            cy=intrinsics_data['cy'],
            width=intrinsics_data['width'],
            height=intrinsics_data['height'],
            hfov=intrinsics_data.get('hfov', 78.0),
            vfov=intrinsics_data.get('vfov', 62.0),
            k1=intrinsics_data.get('k1', 0.0),
            k2=intrinsics_data.get('k2', 0.0),
            p1=intrinsics_data.get('p1', 0.0),
            p2=intrinsics_data.get('p2', 0.0)
        )
        
        extrinsics = CameraExtrinsics(
            position=tuple(extrinsics_data['position']),
            rotation=tuple(extrinsics_data['rotation'])
        )
        
        return cls(
            name=config_data['name'],
            intrinsics=intrinsics,
            extrinsics=extrinsics
        )
    
    def get_projection_matrix(self) -> np.ndarray:
        """
        Get the camera projection matrix (3x4).
        Projects 3D world points to 2D image coordinates.
        """
        # Intrinsic matrix (3x3)
        K = np.array([
            [self.intrinsics.fx, 0, self.intrinsics.cx],
            [0, self.intrinsics.fy, self.intrinsics.cy],
            [0, 0, 1]
        ])
        
        # Rotation matrix (3x3)
        R = self.extrinsics.get_rotation_matrix()
        
        # Translation vector (3x1)
        t = self.extrinsics.get_translation_vector()
        
        # Extrinsic matrix [R|t] (3x4)
        RT = np.hstack([R, t.reshape(-1, 1)])
        
        # Projection matrix P = K[R|t] (3x4)
        return K @ RT
    
    def project_3d_to_2d(self, point_3d: Point3D) -> Point2D:
        """Project a 3D world point to 2D image coordinates."""
        world_point = np.array([point_3d.x, point_3d.y, point_3d.z, 1.0])
        P = self.get_projection_matrix()
        
        # Add homogeneous coordinate row to make P 4x4
        P_homo = np.vstack([P, [0, 0, 0, 1]])
        
        image_point_homo = P @ world_point
        
        # Convert from homogeneous coordinates
        if image_point_homo[2] != 0:
            x = image_point_homo[0] / image_point_homo[2]
            y = image_point_homo[1] / image_point_homo[2]
        else:
            x = y = float('inf')
        
        return Point2D(
            x=x, 
            y=y, 
            confidence=point_3d.confidence,
            point_id=point_3d.point_id,
            camera_name=self.name
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
    cameras: Dict[str, Camera]  # Camera name -> Camera object
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
