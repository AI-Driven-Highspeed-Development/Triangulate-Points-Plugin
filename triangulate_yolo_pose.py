from typing import Dict, List, Optional

from plugins.yolo_pose_plugin.data_structure import PoseData, Skeleton, Joint as YoloJoint, COCO_KEYPOINT_NAMES
from .data_structure import Point2D, Point3D, Skeleton3D, PoseData3D, CoordinateConverter
from .triangulate_points import TriangulatePoints
from utils.logger_util.logger import get_logger

class TriangulateYoloPose:
    """
    Reconstructs 3D skeletons from 2D pose data captured from multiple cameras.
    """

    def __init__(self):
        """
        Initializes the TriangulateYoloPose class with a TriangulatePoints instance.
        """
        self.logger = get_logger("TriangulateYoloPose")
        self.triangulator = TriangulatePoints()

    def triangulate_skeletons(self, pose_data_per_camera: Dict[str, PoseData]) -> PoseData3D:
        """
        Triangulates 3D skeletons from 2D pose data from multiple camera views.

        This method implements a simple matching strategy: it assumes the first skeleton
        detected in each camera's view corresponds to the same person.

        Args:
            pose_data_per_camera: A dictionary mapping camera names to PoseData objects.

        Returns:
            A PoseData3D object containing the reconstructed 3D skeletons.
        """
        
        # Simple matching: assume the first skeleton in each view is the same person
        matched_skeletons: Dict[str, Skeleton] = {}
        for cam_name, pose_data in pose_data_per_camera.items():
            if pose_data.skeletons:
                # Assuming the first skeleton is the one we want to track
                matched_skeletons[cam_name] = pose_data.skeletons[0]

        if len(matched_skeletons) < 2:
            return PoseData3D(skeletons=[])

        reconstructed_skeleton = self._reconstruct_single_skeleton(matched_skeletons)
        
        if reconstructed_skeleton:
            return PoseData3D(skeletons=[reconstructed_skeleton])
        else:
            return PoseData3D(skeletons=[])

    def _reconstruct_single_skeleton(self, skeletons_2d: Dict[str, Skeleton]) -> Optional[Skeleton3D]:
        """
        Reconstructs a single 3D skeleton from a matched set of 2D skeletons.

        Args:
            skeletons_2d: A dictionary mapping camera names to 2D Skeleton objects.

        Returns:
            A Skeleton3D object or None if reconstruction is not possible.
        """
        joints_3d: List[Point3D] = []
        
        self.logger.debug(f"Reconstructing skeleton from {len(skeletons_2d)} cameras")
        for cam_name, skeleton in skeletons_2d.items():
            self.logger.debug(f"Camera {cam_name}: {len(skeleton.joints)} joints detected")

        # Iterate through all possible joint types
        for joint_label in COCO_KEYPOINT_NAMES:
            points_for_triangulation: Dict[str, Point2D] = {}

            # Collect the 2D point for the current joint from each camera
            for cam_name, skeleton in skeletons_2d.items():
                joint_2d = self._find_joint_by_label(skeleton, joint_label)
                if joint_2d:
                    # Convert normalized coordinates (0-1) to pixel coordinates
                    # Get camera resolution from triangulator
                    camera = self.triangulator.cameras.get(cam_name)
                    if camera:
                        # Create coordinate converter for this camera
                        converter = CoordinateConverter.from_camera_intrinsics(camera.intrinsics)
                        
                        # Create normalized Point2D from YOLO joint
                        normalized_point = Point2D(
                            x=joint_2d.x,
                            y=joint_2d.y,
                            confidence=joint_2d.confidence,
                            point_id=joint_label,
                            camera_name=cam_name
                        )
                        
                        # Convert to pixel coordinates
                        pixel_point = converter.convert_point2d_to_pixel(normalized_point)
                        points_for_triangulation[cam_name] = pixel_point
            
            # Debug: Print 2D points for this joint
            if len(points_for_triangulation) >= 2:
                self.logger.debug(f"Triangulating joint: {joint_label}")
                for cam_name, pixel_point in points_for_triangulation.items():
                    # Get original normalized coordinates for comparison
                    joint_2d = self._find_joint_by_label(skeletons_2d[cam_name], joint_label)
                    self.logger.debug(
                        f"  {cam_name}: norm=({joint_2d.x:.3f}, {joint_2d.y:.3f}) -> pixel=({pixel_point.x:.0f}, {pixel_point.y:.0f}) conf={pixel_point.confidence:.2f}"
                    )
                
                point_3d = self.triangulator.triangulate_multi_camera(points_for_triangulation)
                if point_3d:
                    self.logger.debug(
                        f"  -> 3D: ({point_3d.x:.3f}, {point_3d.y:.3f}, {point_3d.z:.3f}) conf={point_3d.confidence:.2f} error={point_3d.reprojection_error:.1f}px"
                    )
                    joints_3d.append(point_3d)
                else:
                    self.logger.debug("  -> 3D: FAILED")
            else:
                self.logger.debug(f"{joint_label}: Only {len(points_for_triangulation)} cameras detected this joint")

        self.logger.debug(f"Total triangulated joints: {len(joints_3d)}")

        if not joints_3d:
            return None

        return Skeleton3D(joints=joints_3d)

    def _find_joint_by_label(self, skeleton: Skeleton, label: str) -> Optional[YoloJoint]:
        """
        Finds a joint in a skeleton by its label.

        Args:
            skeleton: The Skeleton object to search in.
            label: The label of the joint to find.

        Returns:
            The Joint object if found, otherwise None.
        """
        for joint in skeleton.joints:
            if joint.label == label:
                return joint
        return None