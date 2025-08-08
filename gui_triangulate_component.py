import cv2
import numpy as np
import math
import time
from typing import List, Optional, Tuple
from plugins.cv2_visualization_plugin.gui_component import GuiComponent
from plugins.triangulate_points_plugin.data_structure import Point3D
from plugins.triangulate_points_plugin.triangulate_points import TriangulatePoints


class GuiTriangulateComponent(GuiComponent):
    """
    3D visualization component for triangulated points that can be embedded in layouts.
    Simplified version without mouse interaction for grid display.
    """

    def __init__(self, name: str, width: int = 320, height: int = 240, auto_rotate: bool = True):
        """Initialize the 3D triangulation component."""
        super().__init__(name, width, height)
        
        # Load triangulation system to get camera positions
        self.triangulator = TriangulatePoints()
        
        # 3D visualization state
        self.rotation_x = 0.2  # Fixed pitch for good viewing angle
        self.rotation_y = 0.5  # Fixed yaw for good viewing angle
        self.rotation_z = 0.0  # No roll
        self.scale = 200.0      # Zoom level (pixels per meter)
        self.auto_rotate = auto_rotate # Auto-rotate for dynamic view
        # Use radians per second for constant-speed rotation
        self.rotation_speed = 0.6
        self._last_time = time.monotonic()
        
        # Mouse interaction state
        self.mouse_dragging = False
        self.mouse_last_x = 0
        self.mouse_last_y = 0
        self.mouse_sensitivity = 0.01
        
        # Keyboard interaction state
        self.key_rotation_speed = 0.1
        self.key_zoom_speed = 1.1
        
        # Data to visualize
        self.points_3d: List[Point3D] = []
        
        # Display options
        self.show_coordinate_axes = True
        self.show_grid = True
        self.show_cameras = True
        self.show_skeleton = True
        self.point_size = 3
        
        # Colors (BGR format for OpenCV)
        self.color_points = (0, 255, 0)      # Green for 3D points
        self.color_cameras = (0, 0, 255)     # Red for cameras
        self.color_axes = (255, 255, 255)    # White for axes
        self.color_grid = (64, 64, 64)       # Dark gray for grid
        self.color_text = (255, 255, 255)    # White for text
        self.color_bg = (20, 20, 20)         # Dark background
        self.color_skeleton = (0, 255, 255)  # Cyan for skeleton lines
        
        # COCO skeleton connections (same as YOLO pose structure)
        self.skeleton_connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # Body
            (5, 11), (6, 12), (11, 12), # Hips
            (11, 13), (13, 15), (12, 14), (14, 16) # Legs
        ]
        
        # COCO joint names for reference
        self.joint_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
    
    def set_points_3d(self, points: List[Point3D]):
        """Set the 3D points to visualize."""
        self.points_3d = points.copy()
    
    def add_point_3d(self, point: Point3D):
        """Add a single 3D point to visualization."""
        self.points_3d.append(point)
    
    def clear_points(self):
        """Clear all 3D points."""
        self.points_3d.clear()
    
    def handle_mouse_event(self, event: int, x: int, y: int, flags: int) -> bool:
        """
        Handle mouse events for rotation and zoom control.
        
        Args:
            event: OpenCV mouse event type
            x, y: Mouse coordinates relative to component
            flags: Mouse event flags
            
        Returns:
            True if event was handled, False otherwise
        """
        # Convert absolute coordinates to component-relative coordinates
        abs_x, abs_y = self.abs_position
        rel_x = x - abs_x
        rel_y = y - abs_y
        
        # Check if mouse is within component bounds
        if not (0 <= rel_x < self.width and 0 <= rel_y < self.height):
            self.mouse_dragging = False
            return False
        
        handled = True
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_dragging = True
            self.mouse_last_x = rel_x
            self.mouse_last_y = rel_y
            # Disable auto-rotation when user starts interacting
            self.auto_rotate = False
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.mouse_dragging = False
        
        elif event == cv2.EVENT_MOUSEMOVE and self.mouse_dragging:
            # Calculate rotation deltas
            delta_x = rel_x - self.mouse_last_x
            delta_y = rel_y - self.mouse_last_y
            
            # Update rotation (with sensitivity factor)
            self.rotation_y += delta_x * self.mouse_sensitivity  # Yaw
            self.rotation_x += delta_y * self.mouse_sensitivity  # Pitch
            
            # Store current mouse position
            self.mouse_last_x = rel_x
            self.mouse_last_y = rel_y
        
        elif event == cv2.EVENT_MOUSEWHEEL:
            # Zoom with mouse wheel
            if flags > 0:  # Scroll up
                self.scale *= self.key_zoom_speed
            else:  # Scroll down
                self.scale /= self.key_zoom_speed
            self.scale = np.clip(self.scale, 10.0, 500.0)  # Limit zoom range
        
        else:
            handled = False
        
        return handled
    
    def handle_keyboard_event(self, key: int) -> bool:
        """
        Handle keyboard input for navigation.
        
        Args:
            key: OpenCV key code
            
        Returns:
            True if key was handled, False otherwise
        """
        handled = True
        
        # Arrow keys for rotation (disable auto-rotation when user controls)
        if key == ord('w') or key == 82:  # W or Up arrow
            self.rotation_x -= self.key_rotation_speed
            self.auto_rotate = False
        elif key == ord('s') or key == 84:  # S or Down arrow
            self.rotation_x += self.key_rotation_speed
            self.auto_rotate = False
        elif key == ord('a') or key == 81:  # A or Left arrow
            self.rotation_y -= self.key_rotation_speed
            self.auto_rotate = False
        elif key == ord('d') or key == 83:  # D or Right arrow
            self.rotation_y += self.key_rotation_speed
            self.auto_rotate = False
        elif key == ord('q'):  # Q for roll left
            self.rotation_z -= self.key_rotation_speed
            self.auto_rotate = False
        elif key == ord('e'):  # E for roll right
            self.rotation_z += self.key_rotation_speed
            self.auto_rotate = False
        
        # Zoom controls
        elif key == ord('+') or key == ord('='):  # + for zoom in
            self.scale *= self.key_zoom_speed
            self.scale = np.clip(self.scale, 10.0, 500.0)
            self.auto_rotate = False
        elif key == ord('-'):  # - for zoom out
            self.scale /= self.key_zoom_speed
            self.scale = np.clip(self.scale, 10.0, 500.0)
            self.auto_rotate = False
        
        # Reset controls
        elif key == ord('r'):  # R to reset view
            self.rotation_x = 0.2
            self.rotation_y = 0.5
            self.rotation_z = 0.0
            self.scale = 50.0
            self.auto_rotate = True
        
        # Toggle auto-rotation
        elif key == ord(' '):  # Space to toggle auto-rotation
            self.auto_rotate = not self.auto_rotate
        
        # Toggle display options
        elif key == ord('1'):  # 1 to toggle coordinate axes
            self.show_coordinate_axes = not self.show_coordinate_axes
        elif key == ord('2'):  # 2 to toggle grid
            self.show_grid = not self.show_grid
        elif key == ord('3'):  # 3 to toggle cameras
            self.show_cameras = not self.show_cameras
        elif key == ord('4'):  # 4 to toggle skeleton
            self.show_skeleton = not self.show_skeleton
        
        else:
            handled = False
        
        return handled
    
    def _get_rotation_matrix(self) -> np.ndarray:
        """Calculate combined rotation matrix from current rotation state."""
        # Individual rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(self.rotation_x), -math.sin(self.rotation_x)],
            [0, math.sin(self.rotation_x), math.cos(self.rotation_x)]
        ])
        
        Ry = np.array([
            [math.cos(self.rotation_y), 0, math.sin(self.rotation_y)],
            [0, 1, 0],
            [-math.sin(self.rotation_y), 0, math.cos(self.rotation_y)]
        ])
        
        Rz = np.array([
            [math.cos(self.rotation_z), -math.sin(self.rotation_z), 0],
            [math.sin(self.rotation_z), math.cos(self.rotation_z), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation: Rz * Ry * Rx
        return Rz @ Ry @ Rx
    
    def _project_3d_to_2d(self, point_3d: np.ndarray) -> Tuple[int, int]:
        """
        Project a 3D point to 2D screen coordinates.
        
        Args:
            point_3d: 3D point [x, y, z]
            
        Returns:
            2D screen coordinates (x, y)
        """
        # Apply rotation
        rotation_matrix = self._get_rotation_matrix()
        rotated = rotation_matrix @ point_3d
        
        # Orthographic projection matching OpenCV convention:
        # In cv2/image coordinates, smaller y is up; in 3D, up corresponds to negative Y.
        # Therefore, map rotated[1] directly to screen Y (negative values go up on screen).
        screen_x = rotated[0] * self.scale + self.width // 2
        screen_y = rotated[1] * self.scale + self.height // 2
        
        return (int(screen_x), int(screen_y))
    
    def _draw_coordinate_axes(self, canvas: np.ndarray):
        """Draw 3D coordinate axes."""
        if not self.show_coordinate_axes:
            return
        
        # Axis endpoints (1 meter in each direction)
        origin = np.array([0.0, 0.0, 0.0])
        x_axis = np.array([1.0, 0.0, 0.0])
        # In OpenCV convention, up corresponds to -Y; draw the Y axis pointing upwards on screen
        y_axis = np.array([0.0, -1.0, 0.0])
        z_axis = np.array([0.0, 0.0, 1.0])
        
        # Project to 2D (using local canvas coordinates)
        origin_2d = self._project_3d_to_2d(origin)
        x_axis_2d = self._project_3d_to_2d(x_axis)
        y_axis_2d = self._project_3d_to_2d(y_axis)
        z_axis_2d = self._project_3d_to_2d(z_axis)
        
        # Draw axes
        cv2.line(canvas, origin_2d, x_axis_2d, (0, 0, 255), 1)  # X = Red
        cv2.line(canvas, origin_2d, y_axis_2d, (0, 255, 0), 1)  # Y = Green
        cv2.line(canvas, origin_2d, z_axis_2d, (255, 0, 0), 1)  # Z = Blue
    
    def _draw_grid(self, canvas: np.ndarray):
        """Draw a reference grid on the XZ plane."""
        if not self.show_grid:
            return
        
        grid_size = 2  # 2x2 meter grid (smaller for component view)
        grid_step = 1  # 1 meter steps
        
        # Draw grid lines (using local canvas coordinates)
        for i in range(-grid_size, grid_size + 1, grid_step):
            # Lines parallel to X-axis
            start_3d = np.array([float(-grid_size), 0.0, float(i)])
            end_3d = np.array([float(grid_size), 0.0, float(i)])
            start_2d = self._project_3d_to_2d(start_3d)
            end_2d = self._project_3d_to_2d(end_3d)
            cv2.line(canvas, start_2d, end_2d, self.color_grid, 1)
            
            # Lines parallel to Z-axis
            start_3d = np.array([float(i), 0.0, float(-grid_size)])
            end_3d = np.array([float(i), 0.0, float(grid_size)])
            start_2d = self._project_3d_to_2d(start_3d)
            end_2d = self._project_3d_to_2d(end_3d)
            cv2.line(canvas, start_2d, end_2d, self.color_grid, 1)
    
    def _draw_cameras(self, canvas: np.ndarray):
        """Draw camera positions and orientations."""
        if not self.show_cameras:
            return
        
        for name, camera in self.triangulator.cameras.items():
            # Get camera position
            position = np.array(camera.extrinsics.position)
            position_2d = self._project_3d_to_2d(position)
            
            # Draw camera as a filled circle
            cv2.circle(canvas, position_2d, 6, self.color_cameras, -1)
            
            # Draw camera name
            cv2.putText(canvas, name, 
                       (position_2d[0] + 8, position_2d[1] - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.color_cameras, 1)
            
            # Draw camera orientation (viewing direction)
            # Convert rotation from degrees to radians and create rotation matrix
            rotation_deg = camera.extrinsics.rotation
            rx, ry, rz = np.radians(rotation_deg)
            
            # Create rotation matrix (Euler ZYX, same order as elsewhere)
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(rx), -np.sin(rx)],
                [0, np.sin(rx), np.cos(rx)]
            ])
            
            Ry = np.array([
                [np.cos(ry), 0, np.sin(ry)],
                [0, 1, 0],
                [-np.sin(ry), 0, np.cos(ry)]
            ])
            
            Rz = np.array([
                [np.cos(rz), -np.sin(rz), 0],
                [np.sin(rz), np.cos(rz), 0],
                [0, 0, 1]
            ])
            
            # Combined rotation matrix
            rotation_matrix = Rz @ Ry @ Rx
            
            # For [R|t] with R mapping world->camera, the camera forward in world is R^T * [0,0,1]
            forward_cam = np.array([0.0, 0.0, 1.0])  # camera Z axis
            world_forward = rotation_matrix.T @ forward_cam
            
            # Scale the direction vector for visibility
            direction_end = position + world_forward * 0.5  # 0.5 meter direction indicator
            direction_end_2d = self._project_3d_to_2d(direction_end)
            
            # Draw direction line and arrowhead
            cv2.line(canvas, position_2d, direction_end_2d, self.color_cameras, 2)
            direction_vector = np.array(direction_end_2d) - np.array(position_2d)
            if np.linalg.norm(direction_vector) > 0:
                direction_vector = direction_vector / np.linalg.norm(direction_vector)
                arrow_length = 8
                arrow_angle = 0.5
                arrow_tip1 = direction_end_2d - arrow_length * (
                    direction_vector * np.cos(arrow_angle) + 
                    np.array([-direction_vector[1], direction_vector[0]]) * np.sin(arrow_angle)
                )
                arrow_tip2 = direction_end_2d - arrow_length * (
                    direction_vector * np.cos(arrow_angle) - 
                    np.array([-direction_vector[1], direction_vector[0]]) * np.sin(arrow_angle)
                )
                cv2.line(canvas, direction_end_2d, tuple(arrow_tip1.astype(int)), self.color_cameras, 2)
                cv2.line(canvas, direction_end_2d, tuple(arrow_tip2.astype(int)), self.color_cameras, 2)
    
    def _draw_points_3d(self, canvas: np.ndarray):
        """Draw the triangulated 3D points."""
        for point in self.points_3d:
            # Project to 2D (using local canvas coordinates)
            point_3d = np.array([point.x, point.y, point.z])
            point_2d = self._project_3d_to_2d(point_3d)
            
            # Color based on confidence
            confidence_color = int(255 * point.confidence)
            color = (0, confidence_color, 0)  # Green intensity based on confidence
            
            # Draw point
            cv2.circle(canvas, point_2d, self.point_size, color, -1)
    
    def _draw_skeleton_3d(self, canvas: np.ndarray):
        """Draw skeleton connections between related joints."""
        if not self.show_skeleton or len(self.points_3d) < 2:
            return
        
        # Group points by skeleton (assume single skeleton for now)
        # Create a mapping from joint index/name to 3D point
        joint_map = {}
        
        for point in self.points_3d:
            # Try to map joint using point_id (joint name or index)
            if point.point_id:
                # Handle both string names and numeric indices
                if isinstance(point.point_id, str):
                    # Map joint name to index
                    try:
                        joint_idx = self.joint_names.index(point.point_id)
                        joint_map[joint_idx] = point
                    except ValueError:
                        # If not a standard COCO name, try to parse as number
                        try:
                            joint_idx = int(point.point_id)
                            joint_map[joint_idx] = point
                        except ValueError:
                            continue
                else:
                    # Assume it's already a numeric index
                    joint_map[int(point.point_id)] = point
        
        # Draw skeleton connections
        for connection in self.skeleton_connections:
            joint_a_idx, joint_b_idx = connection
            
            # Check if both joints exist in our data
            if joint_a_idx in joint_map and joint_b_idx in joint_map:
                joint_a = joint_map[joint_a_idx]
                joint_b = joint_map[joint_b_idx]
                
                # Project both points to 2D
                point_a_3d = np.array([joint_a.x, joint_a.y, joint_a.z])
                point_b_3d = np.array([joint_b.x, joint_b.y, joint_b.z])
                point_a_2d = self._project_3d_to_2d(point_a_3d)
                point_b_2d = self._project_3d_to_2d(point_b_3d)
                
                # Calculate line confidence based on both joint confidences
                line_confidence = min(joint_a.confidence, joint_b.confidence)
                
                # Adjust line color based on confidence
                if line_confidence > 0.5:
                    # High confidence - bright cyan
                    line_color = self.color_skeleton
                    line_thickness = 2
                elif line_confidence > 0.3:
                    # Medium confidence - dimmer cyan
                    line_color = (0, int(255 * line_confidence), int(255 * line_confidence))
                    line_thickness = 1
                else:
                    # Low confidence - very dim or skip
                    line_color = (0, int(128 * line_confidence), int(128 * line_confidence))
                    line_thickness = 1
                
                # Draw the skeleton line
                cv2.line(canvas, point_a_2d, point_b_2d, line_color, line_thickness)
    
    def _draw_info_text(self, canvas: np.ndarray):
        """Draw information text."""
        # Title (using local canvas coordinates)
        cv2.putText(canvas, "3D Pose",
                   (5, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.color_text, 1)
        
        # Point count
        cv2.putText(canvas, f"Points: {len(self.points_3d)}",
                   (5, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.color_text, 1)
        
        # Camera count
        cv2.putText(canvas, f"Cameras: {len(self.triangulator.cameras)}",
                   (5, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.color_text, 1)
        
        # Auto-rotation status
        auto_status = "ON" if self.auto_rotate else "OFF"
        cv2.putText(canvas, f"Auto: {auto_status}",
                   (5, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.color_text, 1)
        
        # Skeleton status
        skeleton_status = "ON" if self.show_skeleton else "OFF"
        cv2.putText(canvas, f"Skeleton: {skeleton_status}",
                   (5, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.color_text, 1)
        
        # Controls info (smaller text)
        controls = [
            "Mouse: Drag to rotate",
            "Wheel: Zoom",
            "WASD/Arrows: Rotate",
            "Q/E: Roll, +/-: Zoom",
            "R: Reset, Space: Auto",
            "1-4: Toggle Display"
        ]
        
        y_offset = self.height - 72
        for i, control in enumerate(controls):
            cv2.putText(canvas, control,
                       (5, y_offset + i * 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.25, (180, 180, 180), 1)
    
    def draw(self):
        """Draw the 3D visualization on the component's canvas."""
        # Create canvas if not exists
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Fill component area with background color
        cv2.rectangle(self.canvas, 
                     (0, 0), 
                     (self.width, self.height),
                     self.color_bg, -1)
        
        # Auto-rotate for dynamic view at constant speed (time-based)
        now = time.monotonic()
        dt = max(0.0, min(now - self._last_time, 0.1))  # clamp to avoid big jumps
        self._last_time = now
        if self.auto_rotate:
            self.rotation_y += self.rotation_speed * dt
        
        # Draw scene elements (all drawing methods need to be updated to use canvas coordinates)
        self._draw_grid(self.canvas)
        self._draw_coordinate_axes(self.canvas)
        self._draw_cameras(self.canvas)
        self._draw_skeleton_3d(self.canvas)  # Draw skeleton lines first (behind points)
        self._draw_points_3d(self.canvas)
        self._draw_info_text(self.canvas)
        
        # Draw border
        cv2.rectangle(self.canvas, 
                     (0, 0), 
                     (self.width, self.height),
                     (100, 100, 100), 1)
