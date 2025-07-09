import cv2
import copy
import numpy as np
import torch
import time
import threading
import queue
import os

from src import util_fast
from src.body_fast import Body

# Performance configuration
KEYPOINT_CONFIDENCE_THRESHOLD = 0.1  # Lowered threshold for detection
POSE_SKIP_FRAMES = 3  # Process every other frame for better performance
RENDER_SKIP_FRAMES = 1  # Show all frames for visual feedback
USE_LOWER_RESOLUTION = True  # Process smaller images for faster detection
DETECTION_SCALE_FACTOR = 0.3  # Scale down images by this factor for detection

# Ball physics constants
BALL_RADIUS = 20
GRAVITY = 0.9  # Strong gravity
ELASTICITY = 0.7  # Reduced elasticity for more realistic bounces
FRICTION = 0.98  # Friction coefficient
ARM_COLLISION_PADDING = 10  # Distance beyond ball radius for arm collision detection

# Palm detection constants
PALM_BASE_SIZE = 40  # Base palm size for collision detection
MIN_PALM_SIZE = 30  # Minimum palm size regardless of distance
MAX_PALM_SIZE = 60  # Maximum palm size regardless of distance
PALM_DISTANCE_FACTOR = 1.5  # How much the palm size changes with hand position
WRIST_ELBOW_REF_DISTANCE = 100  # Reference distance between wrist and elbow at medium range

# Debug flags
DEBUG_MODE = False  # Set to True for detailed debugging info
REDUCED_MODEL_PRECISION = True  # Use FP16 precision if available with torch.amp
DISPLAY_KEYPOINTS = True  # Set to False for even better performance
ENABLE_PALM_DETECTION = True  # Enable palm collision detection

print(f"Starting ball physics demo with optimized settings...")

# Initialize body estimation with optimized settings
try:
    body_estimation = Body('model/body_coco.pth')
    print("Pose estimation model loaded successfully")
except Exception as e:
    print(f"Error loading pose model: {e}")
    print("Make sure the model file exists in the 'model' directory")
    exit(1)

# If CUDA is available, use it and optimize further
if torch.cuda.is_available():
    print(f"Using CUDA: {torch.cuda.get_device_name()}")
    # Set CUDA optimization parameters
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Try to use mixed precision for better performance
    if REDUCED_MODEL_PRECISION:
        try:
            if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
                print("Enabling mixed precision with torch.amp")
        except Exception as e:
            print(f"Could not configure mixed precision: {e}")
else:
    print("CUDA not available, running on CPU")

# Set lower process priority to avoid slowing down the system
try:
    import psutil
    process = psutil.Process(os.getpid())
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == 'nt' else 10)
    print("Set process to lower priority to avoid system slowdowns")
except ImportError:
    print("psutil not installed, skipping process priority adjustment")
except Exception as e:
    print(f"Could not adjust process priority: {e}")

# Initialize camera and get resolution
print("Initializing camera...")
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera")
    exit(1)

# Set camera properties for better performance
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
try:
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
except:
    print("Could not set MJPG format, using default")

# Get the first frame to determine actual camera resolution
ret, test_frame = cap.read()
if not ret:
    print("Error: Could not read frame from camera")
    cap.release()
    exit(1)

# Set canvas dimensions from actual camera resolution
CANVAS_WIDTH = test_frame.shape[1]
CANVAS_HEIGHT = test_frame.shape[0]
print(f"Camera resolution: {CANVAS_WIDTH}x{CANVAS_HEIGHT}")

# Detection resolution (lower for faster processing)
if USE_LOWER_RESOLUTION:
    DETECTION_WIDTH = int(CANVAS_WIDTH * DETECTION_SCALE_FACTOR)
    DETECTION_HEIGHT = int(CANVAS_HEIGHT * DETECTION_SCALE_FACTOR)
    print(f"Using scaled detection resolution: {DETECTION_WIDTH}x{DETECTION_HEIGHT}")
else:
    DETECTION_WIDTH = CANVAS_WIDTH
    DETECTION_HEIGHT = CANVAS_HEIGHT

# Ball state variables - initialize at center of screen
ball_pos = np.array([CANVAS_WIDTH // 2, - CANVAS_HEIGHT + BALL_RADIUS*2], dtype=np.float32)
ball_velocity = np.array([0, 0], dtype=np.float32)
floor_y = CANVAS_HEIGHT - 1  # Floor at bottom of screen
ball_active = False
ball_on_ground = False
ground_contact_time = 0
ball_ground_start_time = 0  
# Countdown state variables
countdown_active = True
countdown_start_time = time.time()
countdown_duration = 3.0  # 3 seconds countdown
COUNTDOWN_SPAWN_HEIGHT = 50  # Distance from top of screen to spawn ball

# Score tracking variables
current_score = 0
high_score = 0
last_hit_was_bounce = False  # Track if we should award a point

# Auto-reset variables
auto_reset_duration = 3.0  # Reset ball if on ground for 3 seconds
ball_ground_start_time = 0

# Volleyball shot detection variables
VOLLEYBALL_SHOT_COOLDOWN = 1.0  # Prevent multiple detections of same shot
last_volleyball_shot_time = 0
shot_message = ""
shot_message_time = 0
SHOT_MESSAGE_DURATION = 2.0  # How long to show the shot message

# Shot detection thresholds
DIG_DISTANCE_THRESHOLD = 150  # Max distance between hands for dig shot
SET_HEIGHT_THRESHOLD = 100  # Min height above head for set shot
SPIKE_VELOCITY_THRESHOLD = 15  # Min velocity added for spike detection

# Palm detection state
palm_positions = []  # Array of palm positions with their sizes
palm_history = []  # Track palm positions over time
PALM_HISTORY_LENGTH = 5  # Number of frames to keep palm history

# Time tracking for physics
previous_time = time.time()
fixed_time_step = 1/60.0  # Target 60 physics updates per second
accumulated_time = 0.0

# Pose detection tracking
frame_count = 0
last_valid_keypoints = None

# FPS calculation variables
fps_update_interval = 1.0
fps_last_update = time.time()
fps_frame_count = 0
current_fps = 0

# Thread-safe queues for pose estimation
pose_queue = queue.Queue(maxsize=1)
pose_result_queue = queue.Queue(maxsize=2)  # Buffer for results
pose_thread_running = False
pose_result = None

# Track position history for better collision detection
last_arm_positions = []
ARM_HISTORY_LENGTH = 3

# Physics counters for debug
physics_counter = 0
physics_rate = 0

# Shared state to indicate when a new result is ready
new_pose_result = False

def point_to_segment_distance(p, v, w):
    """
    Calculate the minimum distance from point p to line segment vw
    Returns the distance and the closest point on the segment
    
    p: point (ball position) as a numpy array [x, y]
    v: segment start point as a numpy array [x, y]
    w: segment end point as a numpy array [x, y]
    """
    # Convert all inputs to float32 to ensure consistent calculations
    p = p.astype(np.float32)
    v = v.astype(np.float32)
    w = w.astype(np.float32)
    
    # Vector from v to w
    segment_vector = w - v
    
    # Vector from v to p
    point_vector = p - v
    
    # Calculate squared length of segment
    segment_length_squared = np.dot(segment_vector, segment_vector)
    
    # If segment is essentially a point, return distance to v
    if segment_length_squared < 1e-6:
        return np.linalg.norm(p - v), v
    
    # Calculate projection of point_vector onto segment_vector
    projection_ratio = np.dot(point_vector, segment_vector) / segment_length_squared
    
    # Clamp projection to segment (0 to 1)
    projection_ratio = max(0, min(1, projection_ratio))
    
    # Find closest point on segment
    closest_point = v + projection_ratio * segment_vector
    
    # Calculate distance from p to closest_point
    distance = np.linalg.norm(p - closest_point)
    
    return distance, closest_point

def reset_ball(center_x=None, center_y=None):
    """Reset the ball to the given position or screen center"""
    global ball_pos, ball_velocity, ball_active, ball_on_ground, ball_ground_start_time, current_score  
    if center_x is None or center_y is None:
        center_x, center_y = CANVAS_WIDTH // 2, CANVAS_HEIGHT // 2
    
    ball_pos = np.array([center_x, center_y], dtype=np.float32)
    ball_velocity = np.array([0, 0], dtype=np.float32)
    ball_active = False
    ball_on_ground = False
    ball_ground_start_time = 0 
    current_score = 0

    if DEBUG_MODE:
        print(f"Ball reset to position: ({ball_pos[0]}, {ball_pos[1]})")

def start_countdown_spawn(spawn_x=None):
    """Start countdown and spawn ball at top of screen"""
    global ball_pos, ball_velocity, ball_active, ball_on_ground, countdown_active, countdown_start_time, ball_ground_start_time, current_score 

    if spawn_x is None:
        spawn_x = CANVAS_WIDTH // 2
    
    # Position ball at top of screen
    ball_pos = np.array([spawn_x, COUNTDOWN_SPAWN_HEIGHT], dtype=np.float32)
    ball_velocity = np.array([0, 0], dtype=np.float32)
    ball_active = False  # Ball not active during countdown
    ball_on_ground = False
    ball_ground_start_time = 0
    current_score = 0
    
    # Start countdown
    countdown_active = True
    countdown_start_time = time.time()
    
    print("Countdown started - ball will activate in 3 seconds!")

def update_countdown():
    """Update countdown state and activate ball when countdown ends"""
    global countdown_active, ball_active
    
    if not countdown_active:
        return None
    
    elapsed_time = time.time() - countdown_start_time
    remaining_time = countdown_duration - elapsed_time
    
    if remaining_time <= 0:
        # Countdown finished - activate ball
        countdown_active = False
        ball_active = True
        print("Ball activated!")
        return None
    else:
        # Return countdown number to display
        return int(remaining_time) + 1

def check_wall_collision():
    """Check and handle collisions with screen boundaries"""
    global ball_pos, ball_velocity, ball_on_ground, ground_contact_time, ball_ground_start_time,  current_score, last_hit_was_bounce
    collision_happened = False
    
    # Debug output
    if DEBUG_MODE:
        print(f"Ball position before collision check: ({ball_pos[0]:.1f}, {ball_pos[1]:.1f})")
    
    # Check left wall collision
    if ball_pos[0] - BALL_RADIUS < 0:
        ball_pos[0] = BALL_RADIUS + 1
        ball_velocity[0] = abs(ball_velocity[0]) * ELASTICITY
        collision_happened = True
        if DEBUG_MODE:
            print("Left wall collision")
    
    # Check right wall collision
    elif ball_pos[0] + BALL_RADIUS > CANVAS_WIDTH:
        ball_pos[0] = CANVAS_WIDTH - BALL_RADIUS - 1
        ball_velocity[0] = -abs(ball_velocity[0]) * ELASTICITY
        collision_happened = True
        if DEBUG_MODE:
            print("Right wall collision")
    
    # Check ceiling collision
    if ball_pos[1] - BALL_RADIUS < 0:
        ball_pos[1] = BALL_RADIUS + 1
        ball_velocity[1] = abs(ball_velocity[1]) * ELASTICITY
        collision_happened = True
        if DEBUG_MODE:
            print("Ceiling collision")
    
# Check floor collision - use bottom of ball
    bottom_edge = ball_pos[1] + BALL_RADIUS

    if bottom_edge >= floor_y:
        # Position the ball exactly at the floor
        ball_pos[1] = floor_y - BALL_RADIUS
        
        # Handle bouncing vs. resting
        if abs(ball_velocity[1]) > 0.5:  # Changed from 0.1 to 0.5 for better threshold
            # Bounce with elasticity and extra damping
            ball_velocity[1] = -abs(ball_velocity[1]) * ELASTICITY * 0.8
            ball_velocity[0] *= 0.9  # Horizontal damping on bounce
            collision_happened = True
            if DEBUG_MODE:
                print("Floor bounce collision")
        else:
            # Ball is resting on ground - set vertical velocity to 0
            ball_velocity[1] = 0
            ball_on_ground = True
            
            if current_score > 0:
                print(f"Ball hit ground! Score reset from {current_score} to 0")
            current_score = 0
            last_hit_was_bounce = False

            # Track when ball first hit the ground
            if not ball_on_ground or ball_ground_start_time == 0:
                ball_ground_start_time = time.time()
                
            # Apply strong floor friction
            ball_velocity[0] *= 0.85  # Stronger friction
            if abs(ball_velocity[0]) < 0.1:
                ball_velocity[0] = 0
    else:
        # Ball is not on the ground
        ball_on_ground = False
        ball_ground_start_time = 0  # Reset the timer when ball leaves ground

    # Cap velocity if collision occurred
    if collision_happened:
        max_velocity = 15.0  # Max velocity limit
        current_magnitude = np.linalg.norm(ball_velocity)
        if current_magnitude > max_velocity:
            ball_velocity = (ball_velocity / current_magnitude) * max_velocity
        
        if DEBUG_MODE:
            print(f"Ball position after collision: ({ball_pos[0]:.1f}, {ball_pos[1]:.1f})")
            print(f"New velocity: ({ball_velocity[0]:.1f}, {ball_velocity[1]:.1f})")
    
    return collision_happened

def update_physics(delta_time):
    """Update ball physics for one time step"""
    global ball_pos, ball_velocity, ball_active, ball_on_ground
    
    if not ball_active:
        return False
    
    # Apply physics only if the ball is active
    if not ball_on_ground:
        # Apply gravity only when ball is not on the ground
        ball_velocity[1] += GRAVITY
    
    # Apply friction to slow down over time
    ball_velocity *= FRICTION
    
    # If ball is on ground with almost no velocity, stop it completely
    if ball_on_ground and np.linalg.norm(ball_velocity) < 0.2:
        ball_velocity = np.array([0, 0], dtype=np.float32)
    
    # Update position based on velocity and delta time (scaled for 60fps physics)
    ball_pos += ball_velocity * delta_time * 60.0
    
    # Check for wall collisions
    collision = check_wall_collision()
    
    return collision

def calculate_palm_size(wrist_pos, elbow_pos):
    """
    Calculate the palm size based on the distance between wrist and elbow.
    This estimates the distance from camera - a larger distance means the hand appears smaller.
    
    Returns the palm radius for collision detection
    """
    # Calculate distance between wrist and elbow
    wrist_elbow_distance = np.linalg.norm(np.array(wrist_pos) - np.array(elbow_pos))
    
    # Scale palm size based on distance - closer hands (larger on screen) get larger palm detection
    if wrist_elbow_distance > 0:
        # Calculate palm size as a proportion of the reference distance
        palm_size = PALM_BASE_SIZE * (wrist_elbow_distance / WRIST_ELBOW_REF_DISTANCE) * PALM_DISTANCE_FACTOR
        
        # Clamp palm size between min and max values
        palm_size = max(MIN_PALM_SIZE, min(MAX_PALM_SIZE, palm_size))
        
        return palm_size
    else:
        # Default size if distance calculation fails
        return PALM_BASE_SIZE

def estimate_palm_position(wrist_pos, elbow_pos):
    """
    Estimate palm position based on wrist and elbow positions.
    The palm is positioned a short distance from the wrist, away from the elbow.
    """
    # Convert to numpy arrays for vector operations
    wrist = np.array(wrist_pos)
    elbow = np.array(elbow_pos)
    
    # Vector from elbow to wrist
    direction = wrist - elbow
    
    # Normalize direction vector
    length = np.linalg.norm(direction)
    if length > 0:
        direction = direction / length
    else:
        # Default direction if wrist and elbow are at the same position
        direction = np.array([0, -1])  # Assume hand is pointing up
    
    # Position palm slightly past the wrist (about 15% of the forearm length)
    palm_offset_distance = length * 0.15
    palm_pos = wrist + direction * palm_offset_distance
    
    return palm_pos

def detect_palms(candidate, subset):
    """
    Detect palm positions based on pose estimation results.
    For each detected person, estimate palm positions and calculate their sizes
    based on the distance from the camera.
    
    Returns a list of palm positions and their sizes.
    """
    global palm_positions, palm_history
    
    # Clear previous palm positions
    palm_positions = []
    
    # Need valid candidates and subsets
    if candidate is None or subset is None or len(candidate) == 0 or subset.shape[0] == 0:
        return
    
    # Process all detected people
    for person_idx in range(len(subset)):
        # Check right arm (elbow to wrist)
        right_elbow_idx = int(subset[person_idx][3])  # Index for right elbow
        right_wrist_idx = int(subset[person_idx][4])  # Index for right wrist
        
        if (right_elbow_idx != -1 and right_wrist_idx != -1 and 
            right_elbow_idx < len(candidate) and right_wrist_idx < len(candidate)):
            # Check confidence
            if (candidate[right_elbow_idx][2] > KEYPOINT_CONFIDENCE_THRESHOLD and 
                candidate[right_wrist_idx][2] > KEYPOINT_CONFIDENCE_THRESHOLD):
                # Get coordinates
                elbow_pos = (candidate[right_elbow_idx][0], candidate[right_elbow_idx][1])
                wrist_pos = (candidate[right_wrist_idx][0], candidate[right_wrist_idx][1])
                
                # Estimate palm position
                palm_pos = estimate_palm_position(wrist_pos, elbow_pos)
                
                # Calculate palm size based on apparent distance from camera
                palm_size = calculate_palm_size(wrist_pos, elbow_pos)
                
                # Add to palm positions
                palm_positions.append({
                    "position": palm_pos,
                    "size": palm_size,
                    "side": "right"
                })
        
        # Check left arm (elbow to wrist)
        left_elbow_idx = int(subset[person_idx][6])  # Index for left elbow
        left_wrist_idx = int(subset[person_idx][7])  # Index for left wrist
        
        if (left_elbow_idx != -1 and left_wrist_idx != -1 and 
            left_elbow_idx < len(candidate) and left_wrist_idx < len(candidate)):
            # Check confidence
            if (candidate[left_elbow_idx][2] > KEYPOINT_CONFIDENCE_THRESHOLD and 
                candidate[left_wrist_idx][2] > KEYPOINT_CONFIDENCE_THRESHOLD):
                # Get coordinates
                elbow_pos = (candidate[left_elbow_idx][0], candidate[left_elbow_idx][1])
                wrist_pos = (candidate[left_wrist_idx][0], candidate[left_wrist_idx][1])
                
                # Estimate palm position
                palm_pos = estimate_palm_position(wrist_pos, elbow_pos)
                
                # Calculate palm size based on apparent distance from camera
                palm_size = calculate_palm_size(wrist_pos, elbow_pos)
                
                # Add to palm positions
                palm_positions.append({
                    "position": palm_pos,
                    "size": palm_size,
                    "side": "left"
                })
    
    # Add current palm positions to history
    if palm_positions:
        palm_history.append(palm_positions)
        # Maintain history length
        if len(palm_history) > PALM_HISTORY_LENGTH:
            palm_history.pop(0)

def check_palm_collision():
    """
    Check for collisions between the ball and palm positions.
    Palm collisions should provide a more intuitive interaction than arm-based collisions.
    """
    global ball_pos, ball_velocity, ball_active, ball_on_ground, current_score, high_score, last_hit_was_bounce, shot_message, shot_message_time
    
    if not ENABLE_PALM_DETECTION or not palm_history:
        return False
    
    collision_happened = False
    
    # Check all recent palm positions
    for positions in palm_history:
        for palm in positions:
            # Get palm position and size
            palm_pos = np.array(palm["position"])
            palm_size = palm["size"]
            
            # Calculate distance from ball to palm
            distance = np.linalg.norm(ball_pos - palm_pos)
            
            # Check if close enough for collision
            collision_threshold = BALL_RADIUS + palm_size
            
            if DEBUG_MODE:
                print(f"Distance to {palm['side']} palm: {distance:.1f}, threshold: {collision_threshold}")
            
            if distance < collision_threshold:
                # Activate the ball if it's not already active
                if not ball_active:
                    print(f"Ball activated by {palm['side']} palm hit!")
                    ball_active = True
                
                # Vector from palm to ball
                palm_to_ball_vector = ball_pos - palm_pos
                palm_to_ball_length = np.linalg.norm(palm_to_ball_vector)
                
                if palm_to_ball_length > 0:
                    palm_to_ball_vector = palm_to_ball_vector / palm_to_ball_length
                else:
                    # Default direction if they're at the exact same point
                    palm_to_ball_vector = np.array([0, -1])
                
                # Calculate reflection - simulating a bounce off the palm
                # First, calculate the normal vector to the palm surface
                # (assuming palm normal is the direction from palm to ball)
                normal_vector = palm_to_ball_vector
                
                # Calculate reflection vector
                dot_product = np.dot(ball_velocity, normal_vector)
                reflection = ball_velocity - 2 * dot_product * normal_vector
                
                velocity_before = ball_velocity.copy()

                # Apply reflection with elasticity
                ball_velocity = reflection * ELASTICITY
                
                # Add a hit force in the direction away from the palm
                palm_force = 10.0  # Stronger force for palm hits
                
                # Apply force in direction away from palm
                ball_velocity += palm_to_ball_vector * palm_force
                
                # Move ball slightly away from palm to prevent multiple collisions
                ball_pos += palm_to_ball_vector * 5
                
                ball_on_ground = False

                # Ball is no longer on ground after being hit
                shot_name, bonus_points = detect_volleyball_shot(palm_positions, ball_pos, velocity_before, ball_velocity)

                if last_hit_was_bounce and ball_active:
                    points_to_add = 1 + bonus_points  # Regular point plus bonus
                    current_score += points_to_add
                    if current_score > high_score:
                        high_score = current_score
                    
                    if shot_name:
                        shot_message = f"{shot_name} +{bonus_points} bonus!"
                        shot_message_time = time.time()
                        print(f"{shot_name} Score: {current_score} (High: {high_score})")
                    else:
                        print(f"Score: {current_score} (High: {high_score})")
                        
                last_hit_was_bounce = True

                # Log collision for debugging
                if DEBUG_MODE:
                    print(f"Collision with {palm['side']} palm at position {ball_pos[0]:.1f}, {ball_pos[1]:.1f}")
                    print(f"New velocity: {ball_velocity[0]:.1f}, {ball_velocity[1]:.1f}")
                
                collision_happened = True
                break  # Only process one collision per frame
        
        if collision_happened:
            break
    
    return collision_happened

def detect_volleyball_shot(palm_positions, ball_pos, velocity_before, velocity_after):
    """
    Detect volleyball shots based on hand positions and ball physics
    Returns: (shot_name, bonus_points) or (None, 0)
    """
    global last_volleyball_shot_time
    
    # Cooldown to prevent multiple detections
    if time.time() - last_volleyball_shot_time < VOLLEYBALL_SHOT_COOLDOWN:
        return None, 0
    
    if len(palm_positions) < 2:
        return None, 0
    
    # Get left and right palm positions
    left_palm = None
    right_palm = None
    
    for palm in palm_positions:
        if palm["side"] == "left":
            left_palm = palm
        elif palm["side"] == "right":
            right_palm = palm
    
    if left_palm and right_palm:
        left_pos = np.array(left_palm["position"])
        right_pos = np.array(right_palm["position"])
        
        # Calculate distance between palms
        palm_distance = np.linalg.norm(left_pos - right_pos)
        
        # Average palm position
        avg_palm_pos = (left_pos + right_pos) / 2
        
        # DIG SHOT: Both hands close together and hit from below
        if palm_distance < DIG_DISTANCE_THRESHOLD:
            # Check if hit was from below (palms below ball at contact)
            if avg_palm_pos[1] > ball_pos[1]:
                last_volleyball_shot_time = time.time()
                return "DIG SHOT!", 3
        
        # BUMP/PASS: Hands together, hit at medium height
        if palm_distance < DIG_DISTANCE_THRESHOLD * 1.2:
            # Check if hit was at chest/shoulder level
            if abs(avg_palm_pos[1] - ball_pos[1]) < 50:
                last_volleyball_shot_time = time.time()
                return "BUMP PASS!", 2
        
        # SET SHOT: Both hands above head, gentle upward push
        if avg_palm_pos[1] < ball_pos[1] - SET_HEIGHT_THRESHOLD:
            # Check for gentle upward velocity
            if velocity_after[1] < -5 and velocity_after[1] > -15:
                last_volleyball_shot_time = time.time()
                return "SET SHOT!", 3
    
    # SPIKE: Single hand hit with high velocity
    if len(palm_positions) > 0:
        # Check velocity increase
        velocity_increase = np.linalg.norm(velocity_after) - np.linalg.norm(velocity_before)
        
        if velocity_increase > SPIKE_VELOCITY_THRESHOLD:
            # Check if hit was from above
            for palm in palm_positions:
                palm_pos = np.array(palm["position"])
                if palm_pos[1] < ball_pos[1]:
                    last_volleyball_shot_time = time.time()
                    return "SPIKE!", 5
    
    return None, 0

def scale_keypoints(candidate, subset, scale_factor=1.0):
    """Scale keypoints coordinates if detection was done at a different resolution"""
    if scale_factor == 1.0:
        return candidate, subset
    
    # Create a copy to avoid modifying the original
    scaled_candidate = copy.deepcopy(candidate)
    
    # Scale the x,y coordinates
    for i in range(len(scaled_candidate)):
        scaled_candidate[i][0] = scaled_candidate[i][0] * scale_factor
        scaled_candidate[i][1] = scaled_candidate[i][1] * scale_factor
    
    return scaled_candidate, subset

def check_arm_collision(candidate, subset):
    """Check for collisions between the ball and arm segments"""
    global ball_pos, ball_velocity, ball_active, ball_on_ground, last_arm_positions, current_score, high_score, last_hit_was_bounce
    
    # Need valid candidates and subsets
    if candidate is None or subset is None or len(candidate) == 0 or subset.shape[0] == 0:
        return False
    
    collision_happened = False
    
    # Collect arm segments from all detected people
    arm_segments = []
    
    # Process all detected people
    for person_idx in range(len(subset)):
        # Check right arm (elbow to wrist) - COCO indices are 0-based in array
        right_elbow_idx = int(subset[person_idx][3]) 
        right_wrist_idx = int(subset[person_idx][4])
        
        if (right_elbow_idx != -1 and right_wrist_idx != -1 and 
            right_elbow_idx < len(candidate) and right_wrist_idx < len(candidate)):
            if (candidate[right_elbow_idx][2] > KEYPOINT_CONFIDENCE_THRESHOLD and 
                candidate[right_wrist_idx][2] > KEYPOINT_CONFIDENCE_THRESHOLD):
                # Get coordinates
                elbow_x, elbow_y = candidate[right_elbow_idx][0], candidate[right_elbow_idx][1]
                wrist_x, wrist_y = candidate[right_wrist_idx][0], candidate[right_wrist_idx][1]
                
                arm_segments.append(
                    ("right_lower", 
                     np.array([elbow_x, elbow_y]), 
                     np.array([wrist_x, wrist_y]))
                )
        
        # Check left arm (elbow to wrist) - COCO indices are 0-based in array
        left_elbow_idx = int(subset[person_idx][6])
        left_wrist_idx = int(subset[person_idx][7])
        
        if (left_elbow_idx != -1 and left_wrist_idx != -1 and 
            left_elbow_idx < len(candidate) and left_wrist_idx < len(candidate)):
            if (candidate[left_elbow_idx][2] > KEYPOINT_CONFIDENCE_THRESHOLD and 
                candidate[left_wrist_idx][2] > KEYPOINT_CONFIDENCE_THRESHOLD):
                # Get coordinates
                elbow_x, elbow_y = candidate[left_elbow_idx][0], candidate[left_elbow_idx][1]
                wrist_x, wrist_y = candidate[left_wrist_idx][0], candidate[left_wrist_idx][1]
                
                arm_segments.append(
                    ("left_lower", 
                     np.array([elbow_x, elbow_y]), 
                     np.array([wrist_x, wrist_y]))
                )
    
    # Add current arm positions to history
    if arm_segments:
        last_arm_positions.append(arm_segments)
        # Maintain history length
        if len(last_arm_positions) > ARM_HISTORY_LENGTH:
            last_arm_positions.pop(0)
    
    # Check all recent arm positions to detect fast movements
    for positions in last_arm_positions:
        for name, start, end in positions:
            # Calculate distance from ball to arm segment
            distance, closest_point = point_to_segment_distance(ball_pos, start, end)
            
            # Check if close enough for collision
            collision_threshold = BALL_RADIUS + ARM_COLLISION_PADDING
            
            if DEBUG_MODE:
                print(f"Distance to {name}: {distance:.1f}, threshold: {collision_threshold}")
            
            if distance < collision_threshold:
                # Activate the ball if it's not already active
                if not ball_active:
                    print(f"Ball activated by {name} arm hit!")
                    ball_active = True
                
                # Calculate arm segment direction for velocity influence
                segment_vector = end - start
                segment_length = np.linalg.norm(segment_vector)
                
                if segment_length > 0:  # Prevent division by zero
                    # Get normalized vectors
                    segment_unit_vector = segment_vector / segment_length
                    
                    # Normal to the arm segment (perpendicular)
                    normal_vector = np.array([-segment_unit_vector[1], segment_unit_vector[0]])
                    
                    # Vector from closest point to ball
                    ball_to_arm_vector = ball_pos - closest_point
                    ball_to_arm_length = np.linalg.norm(ball_to_arm_vector)
                    
                    if ball_to_arm_length > 0:
                        ball_to_arm_vector = ball_to_arm_vector / ball_to_arm_length
                    else:
                        ball_to_arm_vector = normal_vector  # Fallback
                    
                    # Calculate proper reflection
                    dot_product = np.dot(ball_velocity, normal_vector)
                    reflection = ball_velocity - 2 * dot_product * normal_vector
                    
                    # Apply reflection with elasticity
                    ball_velocity = reflection * ELASTICITY
                    
                    # Add "hit force" based on arm segment direction
                    arm_force = 8.0  # Strong force for forearm/wrist
                    
                    # Apply force in direction away from arm
                    ball_velocity += ball_to_arm_vector * arm_force
                    
                    # Move ball slightly away from arm to prevent multiple collisions
                    ball_pos += ball_to_arm_vector * 5
                    
                    # Ball is no longer on ground after being hit
                    ball_on_ground = False
                    
                    if last_hit_was_bounce and ball_active:
                        current_score += 1
                        if current_score > high_score:
                            high_score = current_score
                        print(f"Score: {current_score} (High: {high_score})")
                    last_hit_was_bounce = True

                    collision_happened = True
                    break  # Only process one collision per frame
        
        if collision_happened:
            break
    
    return collision_happened

def pose_estimation_thread_func():
    """Background thread function for pose estimation"""
    global pose_thread_running, pose_result, new_pose_result
    
    print("Pose estimation thread started")
    while pose_thread_running:
        try:
            # Get the next frame from the queue with a timeout
            try:
                img = pose_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            
            # Check if the image is valid
            if img is None or img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
                print("Warning: Empty image in pose thread")
                continue
            
            # Check if we need to resize for detection
            if USE_LOWER_RESOLUTION:
                # Resize for faster processing
                processed_img = cv2.resize(img, (DETECTION_WIDTH, DETECTION_HEIGHT))
            else:
                processed_img = img
                
            # Run pose estimation
            candidate, subset = body_estimation(processed_img)
            
            # Scale keypoints back to original size if needed
            if USE_LOWER_RESOLUTION:
                scale_factor = CANVAS_WIDTH / DETECTION_WIDTH
                candidate, subset = scale_keypoints(candidate, subset, scale_factor)
            
            # Put the result in the result queue
            try:
                # Use non-blocking put and clear old items if queue is full
                if pose_result_queue.full():
                    try:
                        pose_result_queue.get_nowait()  # Remove old result
                    except queue.Empty:
                        pass
                pose_result_queue.put_nowait((candidate, subset))
                new_pose_result = True
            except queue.Full:
                pass  # Skip this frame if queue is still full
            
        except Exception as e:
            print(f"Error in pose thread: {e}")
            
        finally:
            # Always mark the task as done
            pose_queue.task_done()
    
    print("Pose estimation thread stopped")

# Start the pose estimation thread
pose_thread_running = True
pose_thread = threading.Thread(target=pose_estimation_thread_func)
pose_thread.daemon = True
pose_thread.start()

# Main loop
try:
    print("Starting main loop - press Q to quit, R to reset ball, SPACE to activate")
    
    while True:
        # Timing for FPS calculation
        current_time = time.time()
        delta_time = current_time - previous_time
        previous_time = current_time
        
        # FPS counter update
        fps_frame_count += 1
        if current_time - fps_last_update > fps_update_interval:
            current_fps = fps_frame_count / (current_time - fps_last_update)
            fps_last_update = current_time
            fps_frame_count = 0
            print(f"Current FPS: {current_fps:.1f}")
            if DEBUG_MODE:
                physics_rate = physics_counter
                physics_counter = 0
                print(f"Physics updates per second: {physics_rate}")
        
        # Cap delta time to prevent large jumps
        if delta_time > 0.1:
            delta_time = 0.1
        
        # Accumulate time for fixed physics updates
        accumulated_time += delta_time
        
        # Get camera frame
        ret, oriImg = cap.read()
        if not ret:
            print("Failed to grab frame")
            time.sleep(0.01)
            continue
            
        # Flip the image horizontally for more intuitive interaction
        oriImg = cv2.flip(oriImg, 1)
        
        # Create canvas copy efficiently
        canvas = copy.copy(oriImg)
        
        # Check for new pose results (non-blocking)
        if new_pose_result:
            try:
                pose_result = pose_result_queue.get_nowait()
                last_valid_keypoints = pose_result
                # Update palm positions based on the new pose
                if ENABLE_PALM_DETECTION:
                    detect_palms(last_valid_keypoints[0], last_valid_keypoints[1])
                new_pose_result = False
            except queue.Empty:
                pass
        
        # Process pose detection on a schedule
        frame_count += 1
        
        # Only submit a new frame for pose analysis if the queue is empty
        # and it's time to do another detection
        if frame_count % POSE_SKIP_FRAMES == 0 and pose_queue.empty():
            try:
                img_for_thread = oriImg.copy()  # Make a copy for the thread
                pose_queue.put_nowait(img_for_thread)  # Non-blocking put
            except queue.Full:
                pass  # Queue is full, skip this frame
        
        # Run fixed timestep physics updates
        physics_update_count = 0
        while accumulated_time >= fixed_time_step:
            physics_update_count += 1
            physics_counter += 1
            
            # Limit physics updates per frame to prevent spiral of death
            if physics_update_count > 5:
                accumulated_time = 0
                break
            
            # Check for palm collisions first (preferred over arm collisions)
            if ENABLE_PALM_DETECTION:
                palm_collision = check_palm_collision()
                
                # If no palm collision, check for arm collision as fallback
                if not palm_collision and last_valid_keypoints is not None:
                    check_arm_collision(last_valid_keypoints[0], last_valid_keypoints[1])
            # If palm detection disabled, just use arm collision
            elif last_valid_keypoints is not None:
                check_arm_collision(last_valid_keypoints[0], last_valid_keypoints[1])
            
            # Update physics with fixed time step
            update_physics(fixed_time_step)
            accumulated_time -= fixed_time_step
        
        # Update countdown if active
        countdown_number = update_countdown()
        
        # Check for auto-reset if ball has been on ground too long
        if ball_on_ground and ball_active and ball_ground_start_time > 0:
            if time.time() - ball_ground_start_time > auto_reset_duration:
                print("Ball auto-reset after being on ground for 3 seconds")
                start_countdown_spawn()

        # Draw body pose if we have valid keypoints and display is enabled
        if DISPLAY_KEYPOINTS and last_valid_keypoints is not None:
            try:
                canvas = util_fast.draw_bodypose(canvas, last_valid_keypoints[0], last_valid_keypoints[1])
                
                # For debugging - highlight arm segments used for collision detection
                if DEBUG_MODE:
                    candidate, subset = last_valid_keypoints
                    # Draw right arm (elbow to wrist) for each person
                    for person_idx in range(len(subset)):
                        right_elbow_idx = int(subset[person_idx][3])
                        right_wrist_idx = int(subset[person_idx][4])
                        
                        if (right_elbow_idx != -1 and right_wrist_idx != -1 and 
                            right_elbow_idx < len(candidate) and right_wrist_idx < len(candidate)):
                            elbow_x, elbow_y = int(candidate[right_elbow_idx][0]), int(candidate[right_elbow_idx][1])
                            wrist_x, wrist_y = int(candidate[right_wrist_idx][0]), int(candidate[right_wrist_idx][1])
                            cv2.line(canvas, (elbow_x, elbow_y), (wrist_x, wrist_y), (0, 255, 255), 2)
                            
                        # Draw left arm (elbow to wrist)
                        left_elbow_idx = int(subset[person_idx][6])
                        left_wrist_idx = int(subset[person_idx][7])
                        
                        if (left_elbow_idx != -1 and left_wrist_idx != -1 and 
                            left_elbow_idx < len(candidate) and left_wrist_idx < len(candidate)):
                            elbow_x, elbow_y = int(candidate[left_elbow_idx][0]), int(candidate[left_elbow_idx][1])
                            wrist_x, wrist_y = int(candidate[left_wrist_idx][0]), int(candidate[left_wrist_idx][1])
                            cv2.line(canvas, (elbow_x, elbow_y), (wrist_x, wrist_y), (0, 255, 255), 2)
            except Exception as e:
                print(f"Error drawing body pose: {e}")
        
        # Draw palm positions if palm detection is enabled
        if ENABLE_PALM_DETECTION and palm_positions:
            for palm in palm_positions:
                palm_pos = palm["position"]
                palm_size = palm["size"]
                palm_side = palm["side"]
                
                # Draw palm circle with size based on distance
                if palm_side == "right":
                    palm_color = (0, 255, 0)  # Green for right palm
                else:
                    palm_color = (255, 0, 255)  # Magenta for left palm
                
                cv2.circle(canvas, 
                          (int(palm_pos[0]), int(palm_pos[1])), 
                          int(palm_size), 
                          palm_color, 
                          2)  # Draw as outline
                
                # Draw a small filled circle at palm center for visibility
                cv2.circle(canvas, 
                          (int(palm_pos[0]), int(palm_pos[1])), 
                          5, 
                          palm_color, 
                          -1)  # Filled
                
                # Show palm size as text if in debug mode
                if DEBUG_MODE:
                    cv2.putText(canvas, 
                               f"{int(palm_size)}", 
                               (int(palm_pos[0] + palm_size), int(palm_pos[1])),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, palm_color, 1)
        
        # Draw floor line at bottom of screen
        cv2.line(canvas, (0, floor_y), (CANVAS_WIDTH, floor_y), (0, 255, 255), 1)
        
        # Draw ball with a dynamic color based on velocity
        velocity_magnitude = np.linalg.norm(ball_velocity)
        color_intensity = min(255, int(velocity_magnitude * 15))
        
        # Different color for active vs inactive ball
        if ball_active:
            if ball_on_ground and velocity_magnitude < 0.2:
                ball_color = (0, 100, 200)  # Light blue when resting
            else:
                ball_color = (0, color_intensity, 255 - color_intensity)  # Dynamic color
        else:
            ball_color = (0, 0, 255)  # Red for inactive ball
            
        # Draw ball position
        ball_center = (int(ball_pos[0]), int(ball_pos[1]))
        cv2.circle(canvas, ball_center, BALL_RADIUS, ball_color, -1)
        
        # Draw collision radius for debugging
        if DEBUG_MODE:
            # Draw the collision detection radius around the ball
            cv2.circle(canvas, ball_center, BALL_RADIUS + ARM_COLLISION_PADDING, (255, 255, 0), 1)
        
        # Display velocity vector line from ball center
        if ball_active and velocity_magnitude > 0.5:
            velocity_line_end = (
                int(ball_pos[0] + ball_velocity[0] * 2),
                int(ball_pos[1] + ball_velocity[1] * 2)
            )
            cv2.line(canvas, ball_center, velocity_line_end, (255, 0, 0), 2)
        
        # Display FPS and instructions on screen
        cv2.putText(canvas, f'FPS: {current_fps:.1f}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display ball state
        if countdown_active:
            state_text = f"Ball: Countdown in progress..."
        elif ball_active:
            state_text = "Ball: Active"
            if ball_on_ground and velocity_magnitude < 0.2:
                if ball_ground_start_time > 0:
                    time_remaining = auto_reset_duration - (time.time() - ball_ground_start_time)
                    if time_remaining > 0:
                        state_text = f"Ball: Resting (reset in {time_remaining:.1f}s)"
                    else:
                        state_text = "Ball: Resting"
                else:
                    state_text = "Ball: Resting"
        else:
            state_text = "Ball: Waiting for spawn"
            
        cv2.putText(canvas, state_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display velocity
        if ball_active:
            cv2.putText(canvas, f'Vel: {velocity_magnitude:.1f}', (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            
            # Display position for debugging
            if DEBUG_MODE:
                cv2.putText(canvas, f'Pos: {ball_pos[0]:.0f},{ball_pos[1]:.0f}', (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(canvas, f'On ground: {ball_on_ground}', (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(canvas, f'Physics: {physics_rate}/s', (10, 180), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        

        # Display score and high score
        cv2.putText(canvas, f'Score: {current_score}', (CANVAS_WIDTH - 150, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(canvas, f'High: {high_score}', (CANVAS_WIDTH - 150, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                # Display countdown if active

        # Display volleyball shot message
        if shot_message and time.time() - shot_message_time < SHOT_MESSAGE_DURATION:
            # Calculate fade effect
            time_elapsed = time.time() - shot_message_time
            alpha = 1.0 - (time_elapsed / SHOT_MESSAGE_DURATION)
            
            # Display shot message in center of screen
            text_size = cv2.getTextSize(shot_message, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            text_x = (CANVAS_WIDTH - text_size[0]) // 2
            text_y = CANVAS_HEIGHT // 3
            
            # Draw background box
            padding = 20
            cv2.rectangle(canvas, 
                        (text_x - padding, text_y - text_size[1] - padding),
                        (text_x + text_size[0] + padding, text_y + padding),
                        (0, 0, 0), -1)
            cv2.rectangle(canvas, 
                        (text_x - padding, text_y - text_size[1] - padding),
                        (text_x + text_size[0] + padding, text_y + padding),
                        (0, 255, 255), 3)
            
            # Draw text
            cv2.putText(canvas, shot_message, (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        else:
            shot_message = ""

        if countdown_active:
            countdown_num = update_countdown()
            if countdown_num is not None:
                # Draw large countdown number
                countdown_text = str(countdown_num)
                text_size = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_SIMPLEX, 3, 5)[0]
                text_x = (CANVAS_WIDTH - text_size[0]) // 2
                text_y = (CANVAS_HEIGHT + text_size[1]) // 2
                
                # Draw background circle for countdown
                cv2.circle(canvas, (text_x + text_size[0]//2, text_y - text_size[1]//2), 80, (0, 0, 0), -1)
                cv2.circle(canvas, (text_x + text_size[0]//2, text_y - text_size[1]//2), 80, (0, 255, 255), 3)
                
                # Draw countdown number
                cv2.putText(canvas, countdown_text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 5)

        # Show controls at bottom of screen
        cv2.putText(canvas, "r/space: countdown spawn | q: quit | d: debug | k: keypoints | p: palm", 
                   (10, CANVAS_HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Display the processed video
        cv2.imshow('Ball Physics Demo', canvas)
        
        # Check for key presses - use waitKey(1) for maximum responsiveness
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Start countdown spawn instead of immediate reset
            start_countdown_spawn()
            print("Ball countdown spawn started")
        elif key == ord(' '):
            # Start countdown spawn if ball is not active, otherwise give it a kick
            if not ball_active and not countdown_active:
                start_countdown_spawn()
                print("Ball countdown spawn started")
            elif ball_active:
                # Give ball a small kick if already active
                ball_velocity += np.array([1, -8], dtype=np.float32)
                print("Ball kicked")
        elif key == ord('d'):
            # Toggle debug mode
            DEBUG_MODE = not DEBUG_MODE
            print(f"Debug mode: {DEBUG_MODE}")
        elif key == ord('k'):
            # Toggle keypoint display for even better performance
            DISPLAY_KEYPOINTS = not DISPLAY_KEYPOINTS
            print(f"Keypoint display: {DISPLAY_KEYPOINTS}")
        elif key == ord('p'):
            # Toggle palm detection
            ENABLE_PALM_DETECTION = not ENABLE_PALM_DETECTION
            print(f"Palm detection: {ENABLE_PALM_DETECTION}")
        elif key == ord('+'):
            # Increase palm size multiplier
            PALM_DISTANCE_FACTOR += 0.1
            print(f"Palm distance factor: {PALM_DISTANCE_FACTOR:.1f}")
        elif key == ord('-'):
            # Decrease palm size multiplier
            PALM_DISTANCE_FACTOR = max(0.5, PALM_DISTANCE_FACTOR - 0.1)
            print(f"Palm distance factor: {PALM_DISTANCE_FACTOR:.1f}")
except KeyboardInterrupt:
    print("Program interrupted by user")
except Exception as e:
    print(f"Error in main loop: {e}")
finally:
    # Clean up resources
    pose_thread_running = False
    if pose_thread and pose_thread.is_alive():
        pose_thread.join(timeout=1.0)
        
    # Empty the queue to avoid hanging
    while not pose_queue.empty():
        try:
            pose_queue.get_nowait()
            pose_queue.task_done()
        except:
            pass
            
    # Release camera and close windows
    cap.release()
    cv2.destroyAllWindows()
    print("Program terminated successfully")