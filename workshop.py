import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Union, Callable

import cv2
import numpy as np
import mujoco
from PIL import Image

# Third-party imports
from google import genai
from google.genai import types
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower


# --- Configuration & Constants ---

# Suppress noisy logs from libraries
logging.getLogger("lerobot").setLevel(logging.WARNING)

# Configure local logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Joint names for the SO-101 arm
JOINT_NAMES: List[str] = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

# Sensible "Home" pose for arm in degrees
HOME_POSE: np.ndarray = np.array([0.0, -30.0, -30.0, 75.0, -60.0, 0.0])
# Retract pose to move arm out of camera view
RETRACT_POSE: np.ndarray = np.array([0.0, -95.0, 95.0, -95.0, 0.0, 0.0])
CALIBRATION_FILE: str = "homography_calibration.npy"

# ChArUco Board Configuration (Must match your physical board)
SQUARES_X: int = 5
SQUARES_Y: int = 7
SQUARE_LENGTH: float = 0.035  # meters
MARKER_LENGTH: float = 0.026  # meters
DICT_TYPE: int = cv2.aruco.DICT_4X4_250

HOVER_HEIGHT: float = 0.10  # Meters above table
POINT_HEIGHT: float = 0.02  # Meters above table


# --- Helper Functions ---

def show_image(
    image_bgr: Optional[np.ndarray], 
    window_name: str = "Vision Feedback", 
    status_text: Optional[str] = None
) -> None:
    """Displays the image using OpenCV.

    Args:
        image_bgr: The image to display in BGR format.
        window_name: The title of the window.
        status_text: Optional text to overlay on the image (e.g., "LIVE", "PAUSED").
                     "LIVE" will be green, others will be red.
    """
    if image_bgr is None:
        return

    # Create the window and show the image
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    
    # Resize image for better viewing
    h, w = image_bgr.shape[:2]
    disp = image_bgr
    if h > 0:
        scale_factor = 500 / h
        disp = cv2.resize(image_bgr, (int(w * scale_factor), 500))

    # Draw Status Text if provided
    if status_text:
        # Green for LIVE, Red for PAUSED/BUSY
        color = (0, 255, 0) if "LIVE" in status_text else (0, 0, 255)
        cv2.putText(
            disp, 
            status_text, 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            color, 
            2
        )

    cv2.imshow(window_name, disp)

    # Wait for a short duration to update the display
    key = cv2.waitKey(1)
    if key == ord("q"):
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # Workaround for macOS window closing issue
        logger.info("👋 Exiting via video window.")
        sys.exit(0)


def wait_and_show(
    duration: float, 
    cap: Optional[cv2.VideoCapture], 
    status_text: str = "WAITING",
    draw_callback: Optional[Callable[[np.ndarray], None]] = None
) -> None:
    """Waits for a specific duration while keeping the camera feed alive.

    This function replaces `time.sleep()` for blocking operations where we still
    want to see the camera feed.

    Args:
        duration: Time to wait in seconds.
        cap: The OpenCV video capture object. If None, falls back to `time.sleep()`.
        status_text: Text to display on the overlay (e.g., "MOVING", "WAITING").
        draw_callback: Optional function to draw on the frame before display.
    """
    start_time = time.time()
    while (time.time() - start_time) < duration:
        if cap:
            # Try to grab a frame to keep feed alive
            ret, frame = cap.read()
            if ret:
                if draw_callback:
                    draw_callback(frame)
                show_image(frame, status_text=status_text)
        
        # Small sleep to prevent busy loop if no camera, or just to yield
        # If we have camera, show_image already waits 1ms.
        # We check if we still have significant time left to sleep
        elapsed = time.time() - start_time
        remaining = duration - elapsed
        if remaining > 0.005:
            time.sleep(0.001)
        else:
            break


def move_to_joints(
    robot: Optional[SO101Follower],
    target_joints_deg: List[float],
    gripper_pos: float = 0,
    duration: float = 2.0,
    cap: Optional[cv2.VideoCapture] = None,
    draw_callback: Optional[Callable[[np.ndarray], None]] = None
) -> None:
    """Interpolates directly to specific joint angles (no IK).

    Args:
        robot: The robot instance.
        target_joints_deg: List of 6 joint angles in degrees.
        gripper_pos: Desired gripper position (range depends on robot config).
        duration: Time to take for the move in seconds.
        cap: Optional camera capture object for live video feedback.
        draw_callback: Optional function to draw on the frame.
    """
    if robot is None:
        logger.warning(
            "⚠️ Robot not connected. Simulating joint move to %s deg.",
            np.round(target_joints_deg, 2)
        )
        wait_and_show(duration, cap, status_text="SIMULATING MOVE", draw_callback=draw_callback)
        return

    # Get current angles
    q_current = np.array([robot.get_observation()[n] for n in JOINT_NAMES])
    target_joints_deg_full = np.copy(target_joints_deg)

    # Simple interpolation loop
    steps = int(duration * 50)
    if steps < 1:
        steps = 1
        
    for i in range(1, steps + 1):
        t = i / steps
        q_interp = q_current + t * (target_joints_deg_full - q_current)
        robot.send_action({name: val for name, val in zip(JOINT_NAMES, q_interp)})
        wait_and_show(duration / steps, cap, status_text="MOVING (LIVE)", draw_callback=draw_callback)


# --- Kinematics Engine Class ---

class KinematicsEngine:
    """Handles robot kinematics using the MuJoCo physics engine.

    This class abstracts the complexity of Inverse Kinematics (IK). It loads a
    MuJoCo model of the robot and uses it to calculate the joint angles required
    to reach a specific 3D pose in space.
    """

    def __init__(self, model_dir: str = "third_party/SO101"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.ee_link: str = "gripperframe"  # MuJoCo XML site name

        logger.info("⚙️ Initializing Kinematics with MuJoCo...")

        self._setup_mujoco()

    def _setup_mujoco(self) -> None:
        """Sets up MuJoCo physics engine."""
        try:
            self.xml_path = self.model_dir / "so101_new_calib.xml"
            
            # Fallback check
            if not self.xml_path.exists():
                 candidates = [
                     Path("so101_new_calib.xml"),
                     Path("third_party/SO101/so101_new_calib.xml")
                 ]
                 for c in candidates:
                     if c.exists():
                         self.xml_path = c
                         break
            
            if not self.xml_path.exists():
                raise FileNotFoundError(f"MuJoCo XML not found. Expected at {self.xml_path}")

            self.mj_model = mujoco.MjModel.from_xml_path(str(self.xml_path))
            self.mj_data = mujoco.MjData(self.mj_model)
            
            # Cache joint IDs
            self.joint_ids = []
            for name in JOINT_NAMES:
                # Remove .pos suffix to get joint name
                j_name = name.replace(".pos", "")
                j_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
                if j_id == -1:
                    logger.warning("Joint %s not found in MuJoCo model", j_name)
                self.joint_ids.append(j_id)
                
            logger.info("   ✅ MuJoCo Kinematics ready.")
            
        except Exception as e:
            logger.error("   ❌ MuJoCo Load Error: %s", e)
            self.mj_model = None

    def compute_ik(
        self, 
        current_joints_deg: np.ndarray, 
        target_pos: np.ndarray, 
        target_quat: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """Computes Inverse Kinematics (IK) to find joint angles for a target pose.

        Uses the Damped Least Squares (DLS) method to iteratively solve for the
        joint angles that minimize the error between the current end-effector
        position and the target position.

        Args:
            current_joints_deg: Current joint angles in degrees.
            target_pos: Target [x, y, z] position in meters.
            target_quat: Target orientation quaternion [w, x, y, z] (optional).
                         If None, orientation is ignored (position-only IK).

        Returns:
            A numpy array of joint angles in degrees, or None if IK fails to converge.
        """
        try:
            if not hasattr(self, "mj_model") or self.mj_model is None:
                return None
            
            # 1. Set current state in MuJoCo
            # We must initialize the simulation with the robot's current actual angles
            # so the IK solver starts from a valid state.
            for i, j_id in enumerate(self.joint_ids):
                if j_id != -1:
                    self.mj_data.qpos[self.mj_model.jnt_qposadr[j_id]] = np.deg2rad(current_joints_deg[i])
            
            # Forward kinematics to update site positions based on joint angles
            mujoco.mj_forward(self.mj_model, self.mj_data)
            
            # 2. IK Loop (Damped Least Squares Method)
            # We iteratively adjust joint angles to minimize the distance to the target.
            site_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, self.ee_link)
            
            if site_id == -1:
                return None

            step_size = 0.5    # How much of the calculated delta to apply per step
            damping = 1e-3     # Damping factor to ensure numerical stability near singularities
            max_iters = 50     # Maximum number of iterations before giving up
            tol = 0.005        # Tolerance: Stop if error is less than 5mm
            arm_dof = 5        # We only solve for the first 5 joints (arm), ignoring gripper
            
            for _ in range(max_iters):
                # Get current end-effector position
                current_pos = self.mj_data.site_xpos[site_id]
                
                # Calculate error vector (Difference between target and current)
                error = target_pos - current_pos
                
                # Check if we are close enough
                if np.linalg.norm(error) < tol:
                    break
                    
                # Calculate Jacobian Matrix (J)
                # J relates changes in joint angles to changes in end-effector position.
                jacp = np.zeros((3, self.mj_model.nv)) # Position Jacobian
                jacr = np.zeros((3, self.mj_model.nv)) # Orientation Jacobian (unused for pos-only)
                mujoco.mj_jacSite(self.mj_model, self.mj_data, jacp, jacr, site_id)
                
                # Extract only the columns corresponding to our arm joints
                J = np.zeros((3, arm_dof))
                for i in range(arm_dof):
                    j_id = self.joint_ids[i]
                    dof_adr = self.mj_model.jnt_dofadr[j_id]
                    J[:, i] = jacp[:, dof_adr]
                
                # Solve for change in joint angles (dq) using Damped Least Squares:
                # dq = (J^T * J + lambda * I)^-1 * J^T * error
                J_T = J.T
                H = J_T @ J + damping * np.eye(arm_dof) # Hessian matrix with damping
                g = J_T @ error                         # Gradient
                dq = np.linalg.solve(H, g)              # Solve linear system
                
                # Apply the update to joint angles
                for i in range(arm_dof):
                    j_id = self.joint_ids[i]
                    q_adr = self.mj_model.jnt_qposadr[j_id]
                    self.mj_data.qpos[q_adr] += dq[i] * step_size
                    
                # Update simulation state with new angles
                mujoco.mj_forward(self.mj_model, self.mj_data)
            
            # 3. Extract solution
            # Convert the final joint angles from radians back to degrees
            q_sol = []
            for j_id in self.joint_ids:
                q_adr = self.mj_model.jnt_qposadr[j_id]
                q_sol.append(np.rad2deg(self.mj_data.qpos[q_adr]))
            q_sol = np.array(q_sol)

            # Check if gripper pos is missing and restore if needed (LeRobot/Argo only compute 5 arm joints)
            if q_sol is not None and len(q_sol) == 5:
                # Append current gripper position to the 5 arm joint solutions
                q_sol = np.append(q_sol, current_joints_deg[-1])

            return q_sol

        except Exception as e:
            logger.error("IK Computation Error: %s", e)
            return None


# --- Core Logic Functions ---

def perform_move(
    bot: Optional[SO101Follower],
    engine: KinematicsEngine,
    target_xyz: Union[List[float], np.ndarray],
    gripper_pos: float = 0,
    duration: float = 1.5,
    min_z: Optional[float] = None,
    cap: Optional[cv2.VideoCapture] = None,
    draw_callback: Optional[Callable[[np.ndarray], None]] = None
) -> bool:
    """Calculates IK and moves the robot smoothly to the target XYZ.

    Args:
        bot: The robot instance.
        engine: The kinematics engine instance.
        target_xyz: Target position [x, y, z] in meters.
        gripper_pos: Desired gripper position.
        duration: Time to take for the move in seconds.
        min_z: Minimum allowed Z height for safety.
        cap: Optional camera capture object for live video feedback.
        draw_callback: Optional function to draw on the frame.

    Returns:
        True if the move was successful (IK found and executed), False otherwise.
    """
    if bot is None:
        logger.warning(
            "⚠️ Robot not connected (Sim Mode). Target: %s m. Skipping move.",
            np.round(target_xyz, 3)
        )
        wait_and_show(duration, cap, status_text="SIMULATING MOVE", draw_callback=draw_callback)
        return True

    # 0. Safety Check
    if min_z is not None and target_xyz[2] < min_z:
        logger.error(
            "⛔ SAFETY STOP: Target Z %.4f is below minimum safe Z %.4f",
            target_xyz[2], min_z
        )
        return False

    # 1. Get current state (all 6 joints, including gripper)
    q_current = np.array([bot.get_observation()[n] for n in JOINT_NAMES])

    # 2. Construct Target Pose
    # We need to determine the target orientation for the end-effector.
    # For this pointing task, we want the gripper to point straight down (vertical).
    # In the robot's base frame, this corresponds to a specific quaternion.
    # We use a fixed quaternion [0, 1, 0, 0] which typically represents a 180-degree
    # rotation around the X-axis, pointing the Z-axis of the gripper downwards.
    target_quat = np.array([0, 1, 0, 0])
    
    # 3. Compute IK
    q_sol = engine.compute_ik(q_current, target_xyz, target_quat)

    if q_sol is None:
        logger.warning("⚠️ IK failed for target %s", target_xyz)
        return False

    # 4. Execute Move
    move_to_joints(bot, q_sol, gripper_pos, duration, cap=cap, draw_callback=draw_callback)
    return True


def confirm_action(message: str) -> None:
    """Asks the user for confirmation before proceeding."""
    while True:
        response = input(f"\n⚠️  {message} (y/n): ").strip().lower()
        if response == "y":
            return
        elif response == "n":
            logger.info("❌ Action cancelled by user. Exiting.")
            sys.exit(0)
        else:
            print("Please answer 'y' or 'n'.")


def get_object_center_gemini(
    client: genai.Client,
    image_bgr: np.ndarray, 
    target_name: str,
    model_name: str = "gemini-3-flash-preview",
    retries: int = 3
) -> Optional[np.ndarray]:
    """Queries the Gemini API to locate a specific object in an image.

    Args:
        client: The configured Google GenAI client.
        image_bgr: The input image in BGR format (OpenCV default).
        target_name: The name or description of the object to find.
        model_name: The Gemini model version to use.
        retries: Number of times to retry the API call on failure.

    Returns:
        A numpy array [x, y] representing the pixel coordinates of the object's center,
        or None if the object was not found or the API call failed.
    """
    if client is None:
        logger.error("Gemini client is not initialized.")
        return None

    # Convert OpenCV BGR to PIL RGB
    img_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    h_px, w_px = image_bgr.shape[:2]

    prompt = (
        f"Locate the center of the {target_name}. "
        "Return ONLY JSON in this format: {'point': [y, x]} "
        "where y and x are normalized coordinates from 0 to 1000."
    )

    for attempt in range(retries):
        try:
            # Using the Gemini model
            response = client.models.generate_content(
                model=model_name,
                contents=[img_pil, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json", temperature=0.5
                ),
            )

            # Parse JSON
            coords = json.loads(response.text.strip())["point"]
            y_norm, x_norm = coords

            # Convert normalized (0-1000) to pixels
            x_px = int(x_norm / 1000.0 * w_px)
            y_px = int(y_norm / 1000.0 * h_px)

            return np.array([x_px, y_px])

        except Exception as e:
            logger.warning("Gemini Vision Attempt %d/%d Failed: %s", attempt + 1, retries, e)
            time.sleep(1.0)
    
    logger.error("❌ Gemini Vision failed after %d attempts.", retries)
    return None



def calibrate_system(
    cap: cv2.VideoCapture, 
    board_origin_xy: List[float],
    robot: Optional[SO101Follower] = None
) -> Tuple[np.ndarray, float]:
    """Performs ChArUco board calibration to map camera pixels to robot coordinates.

    This process calculates a Homography Matrix (H) that transforms 2D points from
    the camera image plane (pixels) to the 2D plane of the table (meters).

    The calibration assumes:
    1. The camera is fixed.
    2. The table is a flat 2D plane.
    3. The ChArUco board is placed flat on the table.
    4. We know the physical location of the board's "Anchor Corner" (ID 0) relative
       to the robot base.

    Args:
        cap: The OpenCV video capture object.
        board_origin_xy: [x, y] coordinates in meters of the board's ID 0 corner
                         relative to the robot base.
        robot: Optional robot instance. If provided, the arm will move to a
               safe 'RETRACT' pose before capturing the image to ensure it doesn't
               block the view.

    Returns:
        A tuple containing:
        - h_matrix: The 3x3 Homography Matrix.
        - z_surface: The estimated Z-height of the table (usually 0.0).
    """
    logger.info("📏 Starting Calibration...")

    # 1. Safety Retraction
    # If we have control of the robot, move it out of the way so the camera
    # has a clear view of the calibration board.
    if robot:
        logger.info("🔙 Retracting arm for clear view...")
        move_to_joints(robot, RETRACT_POSE, gripper_pos=0, duration=2.0, cap=cap)
        time.sleep(1.0) # Wait for vibrations to settle

    # 2. Capture Image
    # Clear the buffer to ensure we get a fresh frame
    for _ in range(10):
        cap.read()
    
    logger.info("📸 Capturing image for calibration...")
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to capture image from camera.")

    # 3. Detect ChArUco Board
    # We use OpenCV's Aruco module to find the checkerboard corners.
    logger.info("📸 Looking for ChArUco board. Ensure the arm is not obscuring it.")
    
    # Define the board dictionary and layout
    dictionary = cv2.aruco.getPredefinedDictionary(DICT_TYPE)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, dictionary
    )
    detector = cv2.aruco.CharucoDetector(board)

    # Detect the board and corners in the image
    corners, ids, _, _ = detector.detectBoard(frame)

    if ids is not None and len(ids) > 4:
        logger.info("✅ Detected %d corners. Calculating Homography...", len(ids))

        obj_points = []  # Real-world Robot coordinates (Meters)
        img_points = []  # Camera coordinates (Pixels)

        all_board_corners = board.getChessboardCorners()

        # 4. Map Pixels to Robot Coordinates
        # For each detected corner, we know:
        #   a. Its pixel location (from detection)
        #   b. Its local position on the board (from board definition)
        #   c. The board's global position (from board_origin_xy arg)

        # 1. Get the local board coordinate of Corner ID 0 (The Anchor)
        origin_local = all_board_corners[0]

        for i, charuco_id in enumerate(ids.flatten()):
            img_points.append(corners[i][0])

            # Get local board XYZ for this specific corner (e.g., [0.035, 0.07, 0.0] meters)
            current_local = all_board_corners[charuco_id]

            # 2. Calculate Relative Distance from Anchor (ID 0) in meters
            # This tells us how far "down" and "right" this corner is on the board itself.
            diff_x_board_meters = current_local[0] - origin_local[0]
            diff_y_board_meters = current_local[1] - origin_local[1]

            # 3. Map to Robot Frame (Relative to User Measurement)
            # We assume the board is placed such that:
            # - Board X axis aligns with Robot -Y axis (Left)
            # - Board Y axis aligns with Robot -X axis (Backwards)
            #
            # Therefore:
            # Robot X (Forward) = Anchor_X (meters) - Distance_Along_Board_Y (meters)
            rx = board_origin_xy[0] - diff_y_board_meters

            # Robot Y (Left) = Anchor_Y (meters) - Distance_Along_Board_X (meters)
            ry = board_origin_xy[1] - diff_x_board_meters

            obj_points.append([rx, ry])

        # 5. Compute Homography Matrix
        # Find the perspective transformation that maps image points (pixels) to object points (meters).
        H, _ = cv2.findHomography(np.array(img_points), np.array(obj_points))

        # Save to file
        np.save(CALIBRATION_FILE, {"H": H, "z": 0.0})
        logger.info("✅ Calibration Saved to '%s'", CALIBRATION_FILE)

        # --- Verification Output & Visualization ---
        origin_indices = np.where(ids == 0)[0]
        if len(origin_indices) > 0:
            idx = origin_indices[0]
            px = corners[idx][0].astype(int)
            rob = obj_points[idx]
            logger.info("   🎯 VERIFICATION (ID 0): Pixel %s -> Robot %s m", px, np.round(rob, 4))

        disp = frame.copy()
        cv2.aruco.drawDetectedCornersCharuco(disp, corners, ids)

        if len(origin_indices) > 0:
            idx = origin_indices[0]
            origin_px = corners[idx][0].astype(int)
            cv2.circle(disp, tuple(origin_px), 10, (0, 0, 255), -1)
            cv2.putText(
                disp,
                "Anchor",
                (origin_px[0] + 15, origin_px[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        show_image(disp, "Calibration Verification (Press 'q' to close or wait 10s)")
        cv2.waitKey(10000)
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # Workaround for macOS window closing issue

        return H, 0.0  # Return H matrix and z_surface
    else:
        logger.error("❌ Not enough corners detected for calibration.")
        disp = frame.copy()
        if ids is not None:
             cv2.aruco.drawDetectedCornersCharuco(disp, corners, ids)
        cv2.putText(disp, "Calibration Failed: Not enough corners", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        show_image(disp, "Calibration Failed")
        cv2.waitKey(5000)
        return None, None


# --- Setup Functions ---

def setup_robot(args: argparse.Namespace) -> Optional[SO101Follower]:
    """Initializes and connects to the SO-101 robot arm.

    Attempts to connect to the robot hardware using the provided port and ID.
    If connection fails after multiple attempts, it falls back to simulation mode
    (returning None) to allow the script to run without hardware.

    Args:
        args: Command-line arguments containing 'port', 'robot_id', and 'calibration_dir'.

    Returns:
        The connected SO101Follower instance, or None if connection failed (Sim Mode).
    """
    robot = None
    logger.info("⏳ Connecting to robot on %s...", args.port)
    
    for attempt in range(3):
        try:
            if args.calibration_dir is None:
                config = SO101FollowerConfig(port=args.port, id=args.robot_id)
            else:
                config = SO101FollowerConfig(
                    port=args.port,
                    id=args.robot_id,
                    calibration_dir=args.calibration_dir,
                )
            robot = SO101Follower(config)
            robot.connect()
            robot.bus.disable_torque()
            logger.info("✅ Robot Hardware Connected & Torque Disabled.")
            return robot

        except Exception as e:
            logger.warning("⚠️ Connection attempt %d/3 failed: %s", attempt + 1, e)
            if attempt < 2:
                time.sleep(2.0)
            else:
                logger.warning("   ➡️ Proceeding in Simulation Mode (motion commands will be skipped).")
                return None
    return None


def setup_camera(index: int) -> Optional[cv2.VideoCapture]:
    """Initializes the USB camera and verifies it is working.

    Opens the camera at the specified index, warms up the buffer, and displays
    an initial frame to confirm the feed is active.

    Args:
        index: The USB camera index (e.g., 0 for default webcam).

    Returns:
        The OpenCV VideoCapture object, or None if initialization failed.
    """
    cap = cv2.VideoCapture(index)
    for _ in range(5):
        cap.read()  # Warmup buffer

    ret, frame = cap.read()
    if not ret:
        logger.error("❌ Camera failed on index %d.", index)
        cap.release()
        return None
    
    logger.info("✅ Camera Connected! Resolution: %dx%d", frame.shape[1], frame.shape[0])
    show_image(frame, "Initial Camera Check")
    cv2.waitKey(2000)  # Show for 2 seconds
    cv2.destroyAllWindows()
    return cap


def setup_gemini(api_key: str) -> Optional[Any]:
    """Initializes the Google Gemini API client.

    Sets the environment variable and creates the GenAI client.

    Args:
        api_key: The Google AI Studio API key.

    Returns:
        The configured GenAI client instance, or None if setup failed.
    """
    if not api_key:
        logger.error("❌ Error: Google API Key is required.")
        return None

    os.environ["GOOGLE_API_KEY"] = api_key
    try:
        if genai is None:
             raise ImportError("Google GenAI library not installed.")
        client = genai.Client(api_key=api_key)
        logger.info("✅ Gemini API Client Configured.")
        return client
    except Exception as e:
        logger.error("⚠️ Gemini API Setup Failed: %s", e)
        return None


def get_calibration(
    cap: cv2.VideoCapture, args: argparse.Namespace, robot: Optional[SO101Follower] = None
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """Manages the camera-to-robot calibration process.

    Checks for an existing calibration file. If found and --recalibrate is not set,
    loads the calibration. Otherwise, triggers the `calibrate_system` routine.
    If loading fails, it automatically falls back to performing a new calibration.

    Args:
        cap: The camera capture object.
        args: Command-line arguments (checked for 'recalibrate' flag and 'board_origin').
        robot: Optional robot instance (used for retraction during calibration).

    Returns:
        A tuple of (Homography Matrix, Table Z-Height), or (None, None) if failed.
    """
    h_matrix, z_surface = None, None

    if Path(CALIBRATION_FILE).exists() and not args.recalibrate:
        logger.info("Found existing calibration file: %s.", CALIBRATION_FILE)
        try:
            calib_data = np.load(CALIBRATION_FILE, allow_pickle=True).item()
            h_matrix = calib_data["H"]
            z_surface = calib_data["z"]
            logger.info("✅ Calibration Loaded. Table Z-Plane: %.4fm", z_surface)
        except Exception as e:
            logger.warning("❌ Failed to load calibration: %s. Recalibrating.", e)
            confirm_action("Calibration failed to load. The robot will move to RETRACT pose for recalibration. Ensure area is clear.")
            h_matrix, z_surface = calibrate_system(cap, args.board_origin, robot)
    else:
        confirm_action("Starting calibration. The robot will move to RETRACT pose. Ensure area is clear.")
        h_matrix, z_surface = calibrate_system(cap, args.board_origin, robot)
    
    return h_matrix, z_surface


# --- Main Loop ---

def main_loop(
    robot: Optional[SO101Follower],
    kin_engine: KinematicsEngine,
    client: genai.Client,
    cap: cv2.VideoCapture,
    h_matrix: np.ndarray,
    z_surface: float,
    model_name: str
) -> None:
    """Runs the main interaction loop.

    Args:
        robot: The connected robot instance.
        kin_engine: The initialized kinematics engine.
        client: The Gemini API client.
        cap: The OpenCV video capture object.
        h_matrix: The homography matrix for coordinate mapping.
        z_surface: The Z-height of the table surface.
        model_name: The name of the Gemini model to use.
    """
    logger.info("🤖 SYSTEM READY. Type 'q' to quit.")

    # Move to Home first
    logger.info("🏠 Moving to Home Position...")
    move_to_joints(robot, HOME_POSE, gripper_pos=0, duration=2.0, cap=cap)

    # Safety floor: 5mm above table
    SAFETY_Z = z_surface + 0.005

    while True:
        # Show PAUSED status before blocking input
        ret, frame = cap.read()
        if ret:
            show_image(frame, status_text="PAUSED (Waiting for Input)")
        
        target_name = input(
            "\n⌨️ What should I point at? (e.g., 'blue block', 'pen'): "
        ).strip()
        if target_name.lower() == "q":
            logger.info("👋 Exiting.")
            break

        # Retract arm to ensure clear view for camera
        logger.info("🔙 Retracting arm for clear view...")
        move_to_joints(robot, RETRACT_POSE, duration=1.5, cap=cap)
        # Wait for camera autofocus to settle
        wait_and_show(2.0, cap, status_text="FOCUSING...")

        # 1. Observe (Vision)
        # Capture a fresh frame from the camera to see the current state of the world.
        for _ in range(5):
            cap.read()  # Clear buffer to get the latest frame
        ret, frame = cap.read()
        if not ret:
            logger.error("❌ Camera Error")
            continue

        # 2. Think (Language & Vision Model)
        # Send the image and the user's text query to Gemini.
        # Gemini acts as the "brain", understanding the image and finding the object.
        logger.info("🤔 Asking Gemini to find '%s'...", target_name)
        show_image(frame, status_text="PROCESSING (Gemini)")
        pixel_center = get_object_center_gemini(client, frame, target_name, model_name=model_name)

        if pixel_center is not None:
            # 3. Ground (Coordinate Transformation)
            # Convert the 2D pixel coordinates (u, v) from the image into
            # 3D real-world coordinates (x, y, z) for the robot.
            
            # We use the Homography Matrix (H) calculated during calibration.
            # H maps pixels -> table plane (z=0).
            px_array = np.array([[pixel_center]], dtype="float32")
            px_array = px_array.reshape(-1, 1, 2) # Shape required by perspectiveTransform

            # Perform the homography transformation
            robot_xy = cv2.perspectiveTransform(px_array, h_matrix)[0][0]

            # Define target 3D points:
            # - Point: The actual object location (with a small offset above table)
            # - Hover: A safe height directly above the object
            target_xyz = [robot_xy[0], robot_xy[1], z_surface + POINT_HEIGHT]
            hover_xyz = [robot_xy[0], robot_xy[1], z_surface + HOVER_HEIGHT]

            logger.info(
                "📍 Mapped: Pixels %s -> Robot %s m",
                pixel_center, np.round(target_xyz, 3)
            )

            # Define drawing callback for live feed
            coord_text = f"XYZ: [{target_xyz[0]:.3f}, {target_xyz[1]:.3f}, {target_xyz[2]:.3f}]"
            
            def draw_overlay(img: np.ndarray) -> None:
                cv2.circle(img, tuple(pixel_center), 10, (0, 255, 0), 2)
                cv2.drawMarker(img, tuple(pixel_center), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
                cv2.putText(
                    img, coord_text, 
                    (pixel_center[0] + 15, pixel_center[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )

            # Show initial frame with overlay
            disp = frame.copy()
            draw_overlay(disp)
            show_image(disp, status_text="TARGET FOUND")

            # 4. Act (Motion Planning & Control)
            # Execute the movement sequence:
            #   Home -> Hover (Approach) -> Point (Descend)
            logger.info("🏠 Moving to HOME...")
            move_to_joints(robot, HOME_POSE, duration=1.5, cap=cap, draw_callback=draw_overlay)
            wait_and_show(0.2, cap, status_text="HOME", draw_callback=draw_overlay)
            
            logger.info("🚀 Moving to HOVER...")
            if perform_move(robot, kin_engine, hover_xyz, duration=1.5, min_z=SAFETY_Z, cap=cap, draw_callback=draw_overlay):
                wait_and_show(0.2, cap, status_text="HOVER", draw_callback=draw_overlay)
                logger.info("👇 Descending to POINT...")
                perform_move(robot, kin_engine, target_xyz, duration=1.0, min_z=SAFETY_Z, cap=cap, draw_callback=draw_overlay)

        else:
            logger.warning("🤷 Gemini could not locate the object.")


def main(args: argparse.Namespace) -> None:
    """Main entry point for the script.

    Orchestrates the entire application lifecycle:
    1. Initializes hardware (Robot, Camera) and AI (Gemini).
    2. Performs or loads calibration.
    3. Enters the main interaction loop.
    4. Handles cleanup on exit.

    Args:
        args: Parsed command-line arguments.
    """
    # --- 1. Initialization ---
    logger.info("Setting things up, might be slow the first time...")

    # Kinematics Engine Setup
    # Kinematics Engine Setup
    kin_engine = KinematicsEngine()

    robot = None
    cap = None
    
    try:
        # Robot Hardware Connection
        robot = setup_robot(args)

        # Camera Setup
        cap = setup_camera(args.camera_index)
        if cap is None:
            return

        # Gemini API Setup
        client = setup_gemini(args.api_key)
        if client is None:
            return

        # --- 2. Calibration ---
        # 3. Calibration (with robot for retraction)
        h_matrix, z_surface = get_calibration(cap, args, robot)

        if h_matrix is None:
            logger.error("❌ Calibration failed. Exiting.")
            return

        # 4. Main Loop
        confirm_action("Entering autonomous mode. The robot will move to HOME pose. Ensure area is clear.")
        main_loop(robot, kin_engine, client, cap, h_matrix, z_surface, args.gemini_model)

    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    except Exception as e:
        logger.exception("An unexpected error occurred: %s", e)
    finally:
        # --- Cleanup ---
        if robot:
            logger.info("Disabling torque and closing robot connection.")
            try:
                robot.bus.disable_torque()
                robot.disconnect()
            except Exception as e:
                logger.error("Error closing robot connection: %s", e)
        if cap:
            logger.info("Releasing camera.")
            cap.release()
        
        cv2.destroyAllWindows()
        # Mac-specific cleanup: pump events to ensure windows close
        for _ in range(5):
            cv2.waitKey(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vision-Guided Manipulation Script for SO-101 Robot Arm."
    )

    # Hardware/Connection Parameters
    parser.add_argument(
        "--port",
        type=str,
        required=True,
        help="The serial port for the robot arm (e.g., /dev/tty.usbmodem... or COM3).",
    )
    parser.add_argument(
        "--robot-id",
        type=str,
        required=True,
        help="Identifier for the robot; must match calibration filename without extension.",
    )
    parser.add_argument(
        "--calibration-dir",
        type=str,
        help="Directory containing the arm calibration files.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        required=True,
        help="The index of the USB camera to use (e.g., 0, 1, 2).",
    )

    # Kinematics/Brain Parameters

    parser.add_argument(
        "--api-key",
        type=str,
        required=True,
        help="Your Google AI Studio API Key for Gemini models.",
    )
    parser.add_argument(
        "--gemini-model",
        type=str,
        default="gemini-3-flash-preview",
        help="The Gemini model to use (default: gemini-3-flash-preview).",
    )

    # Calibration Parameters
    parser.add_argument(
        "--board-origin",
        type=float,
        nargs=2,
        default=[0.29, 0.0525],
        metavar=("X_FORWARD", "Y_LEFT"),
        help="Robot coordinates (meters) for the ChArUco board origin (Corner ID 0).",
    )
    parser.add_argument(
        "--recalibrate",
        action="store_true",
        help="Force recalibration even if a calibration file exists.",
    )

    args = parser.parse_args()
    main(args)
