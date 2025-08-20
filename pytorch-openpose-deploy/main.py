import cv2
import numpy as np
import torch
import time
import threading
import queue
import os
from collections import deque

from src import util_fast
from src.body_fast import Body
from config import Config
from menu_system import MenuSystem

# Load configuration (now centralized)
cfg = Config()

# Apply performance profile if specified
profile = os.getenv('PERFORMANCE_PROFILE', 'balanced')
if profile != 'balanced':
    profile_settings = Config.get_performance_profile(profile)
    for key, value in profile_settings.items():
        setattr(cfg, key, value)
    print(f"Applied {profile} performance profile")

# Configuration aliases for backward compatibility
KEYPOINT_CONFIDENCE_THRESHOLD = cfg.KEYPOINT_CONFIDENCE_THRESHOLD
POSE_SKIP_FRAMES = cfg.POSE_SKIP_FRAMES
RENDER_SKIP_FRAMES = cfg.RENDER_SKIP_FRAMES
USE_LOWER_RESOLUTION = cfg.USE_LOWER_RESOLUTION
DETECTION_SCALE_FACTOR = cfg.DETECTION_SCALE_FACTOR
BALL_RADIUS = cfg.BALL_RADIUS
GRAVITY = cfg.GRAVITY
ELASTICITY = cfg.ELASTICITY
FRICTION = cfg.FRICTION
ARM_COLLISION_PADDING = cfg.ARM_COLLISION_PADDING
PALM_BASE_SIZE = cfg.PALM_BASE_SIZE
MIN_PALM_SIZE = cfg.MIN_PALM_SIZE
MAX_PALM_SIZE = cfg.MAX_PALM_SIZE
PALM_DISTANCE_FACTOR = cfg.PALM_DISTANCE_FACTOR
WRIST_ELBOW_REF_DISTANCE = cfg.WRIST_ELBOW_REF_DISTANCE
DEBUG_MODE = cfg.DEBUG_MODE
REDUCED_MODEL_PRECISION = cfg.REDUCED_MODEL_PRECISION
DISPLAY_KEYPOINTS = cfg.DISPLAY_KEYPOINTS
ENABLE_PALM_DETECTION = True  # Always enabled for gameplay
DISPLAY_PALM_MARKINGS = cfg.DISPLAY_KEYPOINTS  # Separate toggle for visual markings

# Game constants
COUNTDOWN_DURATION = 3.0
COUNTDOWN_SPAWN_HEIGHT = 50
SUCCESSFUL_VOLLEY_TIMEOUT = 2.0
HIT_COOLDOWN_DURATION = 0.5
AUTO_RESET_DURATION = 3.0
VOLLEYBALL_SHOT_COOLDOWN = 1.0
SHOT_MESSAGE_DURATION = 2.0

# Volleyball shot detection thresholds
DIG_DISTANCE_THRESHOLD = 150
SET_HEIGHT_THRESHOLD = 100
SPIKE_VELOCITY_THRESHOLD = 15

# History lengths
PALM_HISTORY_LENGTH = 5
ARM_HISTORY_LENGTH = 3

# Physics constants
FIXED_TIME_STEP = 1/60.0
MAX_VELOCITY = 15.0
MAX_PHYSICS_STEPS = 3

# Camera and display constants
FPS_UPDATE_INTERVAL = 1.0

# UI Color constants
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
MAGENTA = (255, 0, 255)
ORANGE = (0, 150, 255)
GRAY = (150, 150, 150)
DARK_GRAY = (50, 50, 50)
LIGHT_BLUE = (255, 200, 100)

print(f"Starting ball physics demo with optimized settings...")
if DEBUG_MODE:
    cfg.print_config()

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

# Set camera properties for better performance (configurable)
cap.set(cv2.CAP_PROP_FPS, cfg.CAMERA_FPS)
cap.set(cv2.CAP_PROP_BUFFERSIZE, cfg.CAMERA_BUFFER_SIZE)
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
    # Pre-allocate detection buffer for optimized resizing
    detection_buffer = np.empty((DETECTION_HEIGHT, DETECTION_WIDTH, 3), dtype=np.uint8)
else:
    DETECTION_WIDTH = CANVAS_WIDTH
    DETECTION_HEIGHT = CANVAS_HEIGHT
    detection_buffer = None

# Ball state variables - initialize at center of screen
ball_pos = np.array([CANVAS_WIDTH // 2, - CANVAS_HEIGHT + BALL_RADIUS*2], dtype=np.float32)
ball_velocity = np.array([0, 0], dtype=np.float32)
floor_y = CANVAS_HEIGHT - 1  # Floor at bottom of screen
ball_active = False
ball_on_ground = False
ground_contact_time = 0
ball_ground_start_time = 0

# Ball trail effect
ball_trail = deque(maxlen=15)  # Store last 15 positions

# Particle effects for hits
hit_particles = []  
# Countdown state variables
countdown_active = True
countdown_start_time = time.time()

# Enhanced score tracking variables
current_score = 0
high_score = 0
last_hit_was_bounce = False  # Track if we should award a point

# Scoring system variables
consecutive_hits = 0  # Track consecutive successful hits
last_ground_contact_time = 0

# Cooldown system to prevent multiple points per touch
last_hit_time = 0

# Auto-reset variables
ball_ground_start_time = 0

# Volleyball shot detection variables
last_volleyball_shot_time = 0
shot_message = ""
shot_message_time = 0


# Palm detection state
palm_positions = []  # Array of palm positions with their sizes
palm_history = []  # Track palm positions over time

# Time tracking for physics
previous_time = time.time()
accumulated_time = 0.0

# Pose detection tracking
frame_count = 0
last_valid_keypoints = None

# FPS calculation variables
fps_last_update = time.time()
fps_frame_count = 0
current_fps = 0

# Optimized thread-safe queues for pose estimation
pose_queue = queue.Queue(maxsize=1)
pose_results = deque(maxlen=2)  # Lockless circular buffer for results
pose_thread_running = False
pose_result = None

# Pre-allocated buffers for performance
detection_buffer = None
temp_vector_2d = np.empty(2, dtype=np.float32)
collision_buffer = np.empty((10, 2), dtype=np.float32)

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
    # Use pre-allocated temp vectors and ensure float32 type (optimized)
    if p.dtype != np.float32:
        p = p.astype(np.float32)
    if v.dtype != np.float32:
        v = v.astype(np.float32)
    if w.dtype != np.float32:
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
    global ball_pos, ball_velocity, ball_active, ball_on_ground, ball_ground_start_time, current_score, consecutive_hits, last_ground_contact_time, last_hit_time, ball_trail, hit_particles
    
    if center_x is None or center_y is None:
        center_x, center_y = CANVAS_WIDTH // 2, CANVAS_HEIGHT // 2
    
    ball_pos = np.array([center_x, center_y], dtype=np.float32)
    ball_velocity = np.array([0, 0], dtype=np.float32)
    ball_active = False
    ball_on_ground = False
    ball_ground_start_time = 0 
    current_score = 0
    
    # Reset scoring variables
    consecutive_hits = 0
    last_ground_contact_time = 0
    last_hit_time = 0  # Reset cooldown
    
    # Clear visual effects
    ball_trail.clear()
    hit_particles.clear()

    if DEBUG_MODE:
        print(f"Ball reset to position: ({ball_pos[0]}, {ball_pos[1]})")

def start_countdown_spawn(spawn_x=None):
    """Start countdown and spawn ball at top of screen"""
    global ball_pos, ball_velocity, ball_active, ball_on_ground, countdown_active, countdown_start_time, ball_ground_start_time, current_score, consecutive_hits, last_ground_contact_time, last_hit_time

    if spawn_x is None:
        spawn_x = CANVAS_WIDTH // 2
    
    # Position ball at top of screen
    ball_pos = np.array([spawn_x, COUNTDOWN_SPAWN_HEIGHT], dtype=np.float32)
    ball_velocity = np.array([0, 0], dtype=np.float32)
    ball_active = False  # Ball not active during countdown
    ball_on_ground = False
    ball_ground_start_time = 0
    current_score = 0
    
    # Reset scoring variables
    consecutive_hits = 0
    last_hit_time = 0  # Reset cooldown
    last_ground_contact_time = 0
    
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
    remaining_time = COUNTDOWN_DURATION - elapsed_time
    
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
    global ball_pos, ball_velocity, ball_on_ground, ground_contact_time, ball_ground_start_time, current_score, last_hit_was_bounce, consecutive_hits, last_ground_contact_time
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
            
            # Enhanced ground contact handling
            last_ground_contact_time = time.time()
            
            if current_score > 0:
                # Reset score when ball hits ground for too long
                print(f"Ball hit ground! Score reset from {current_score} to 0")
                current_score = 0
                consecutive_hits = 0
            
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

    # Cap velocity if collision occurred (optimized)
    if collision_happened:
        max_velocity = MAX_VELOCITY
        velocity_mag_sq = np.dot(ball_velocity, ball_velocity)  # Squared magnitude (faster)
        if velocity_mag_sq > max_velocity * max_velocity:
            velocity_mag = np.sqrt(velocity_mag_sq)
            ball_velocity *= (max_velocity / velocity_mag)
        
        if DEBUG_MODE:
            print(f"Ball position after collision: ({ball_pos[0]:.1f}, {ball_pos[1]:.1f})")
            print(f"New velocity: ({ball_velocity[0]:.1f}, {ball_velocity[1]:.1f})")
    
    return collision_happened


class Particle:
    """Simple particle for hit effects"""
    def __init__(self, x, y, vx, vy, color, life=1.0):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.color = color
        self.life = life
        self.max_life = life
        
    def update(self, delta_time):
        self.x += self.vx * delta_time * 60
        self.y += self.vy * delta_time * 60
        self.vy += 0.3  # Small gravity
        self.vx *= 0.99  # Air resistance
        self.life -= delta_time * 2
        return self.life > 0

def create_hit_particles(x, y, velocity, color=(255, 255, 100)):
    """Create particles at hit location"""
    global hit_particles
    
    # Create 8-12 particles radiating outward
    import random
    num_particles = random.randint(8, 12)
    
    for _ in range(num_particles):
        angle = random.uniform(0, 2 * np.pi)
        speed = random.uniform(3, 8)
        
        # Add some velocity influence from ball
        vel_influence = 0.3
        px_vel = np.cos(angle) * speed + velocity[0] * vel_influence
        py_vel = np.sin(angle) * speed + velocity[1] * vel_influence
        
        # Vary particle colors
        particle_color = (
            min(255, color[0] + random.randint(-30, 30)),
            min(255, color[1] + random.randint(-30, 30)),
            min(255, color[2] + random.randint(-30, 30))
        )
        
        hit_particles.append(Particle(x, y, px_vel, py_vel, particle_color, random.uniform(0.5, 1.2)))

def update_particles(delta_time):
    """Update all particles and remove dead ones"""
    global hit_particles
    hit_particles = [p for p in hit_particles if p.update(delta_time)]

def update_physics(delta_time):
    """Update ball physics with improved mechanics"""
    global ball_pos, ball_velocity, ball_active, ball_on_ground, ball_trail
    
    if not ball_active:
        return False
    
    # Add current position to trail
    ball_trail.append((int(ball_pos[0]), int(ball_pos[1])))
    
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

def estimate_hand_segment(wrist_pos, elbow_pos, shoulder_pos=None):
    """
    Estimate hand segment from wrist to fingertips for line-based collision detection.
    Returns both wrist position and fingertip position.
    """
    # Convert to numpy arrays for vector operations
    wrist = np.array(wrist_pos, dtype=np.float32)
    elbow = np.array(elbow_pos, dtype=np.float32)
    
    # Vector from elbow to wrist (forearm direction)
    forearm_vector = wrist - elbow
    forearm_length = np.linalg.norm(forearm_vector)
    
    if forearm_length < 1e-6:  # Degenerate case
        return wrist, wrist + np.array([0, -40], dtype=np.float32)  # Default fallback
    
    # Normalize forearm direction
    forearm_unit = forearm_vector / forearm_length
    
    # Hand length estimation based on forearm length
    # Typical hand length is about 70% of forearm length
    base_hand_length = forearm_length * 0.7
    
    # Consider shoulder position for better hand orientation estimation
    if shoulder_pos is not None:
        shoulder = np.array(shoulder_pos, dtype=np.float32)
        
        # Vector from shoulder to elbow
        upper_arm_vector = elbow - shoulder
        upper_arm_length = np.linalg.norm(upper_arm_vector)
        
        if upper_arm_length > 1e-6:
            upper_arm_unit = upper_arm_vector / upper_arm_length
            
            # Calculate angle between upper arm and forearm
            cos_angle = np.clip(np.dot(upper_arm_unit, forearm_unit), -1, 1)
            arm_angle = np.arccos(cos_angle)
            
            # Adjust hand length based on arm bend (extended arm = longer reach)
            angle_factor = 0.8 + 0.4 * (arm_angle / np.pi)  # 0.8 to 1.2 range
            base_hand_length *= angle_factor
    
    # Calculate fingertip position
    fingertip_pos = wrist + forearm_unit * base_hand_length
    
    return wrist, fingertip_pos

def detect_hands(candidate, subset):
    """
    Detect hand segments (wrist to fingertips) for line-based collision detection.
    For each detected person, estimate hand segments using enhanced algorithm.
    
    Returns a list of hand segments with their properties.
    """
    global palm_positions, palm_history
    
    # Clear previous palm positions
    palm_positions = []
    
    # Need valid candidates and subsets
    if candidate is None or subset is None or len(candidate) == 0 or subset.shape[0] == 0:
        return
    
    # Process all detected people
    for person_idx in range(len(subset)):
        # Get shoulder positions for better hand estimation
        right_shoulder_idx = int(subset[person_idx][2])  # Index for right shoulder
        left_shoulder_idx = int(subset[person_idx][5])   # Index for left shoulder
        
        # Get shoulder positions if available
        right_shoulder_pos = None
        left_shoulder_pos = None
        
        if (right_shoulder_idx != -1 and right_shoulder_idx < len(candidate) and 
            candidate[right_shoulder_idx][2] > KEYPOINT_CONFIDENCE_THRESHOLD):
            right_shoulder_pos = (candidate[right_shoulder_idx][0], candidate[right_shoulder_idx][1])
            
        if (left_shoulder_idx != -1 and left_shoulder_idx < len(candidate) and 
            candidate[left_shoulder_idx][2] > KEYPOINT_CONFIDENCE_THRESHOLD):
            left_shoulder_pos = (candidate[left_shoulder_idx][0], candidate[left_shoulder_idx][1])
        
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
                
                # Estimate hand segment (wrist to fingertips)
                wrist_pos_calc, fingertip_pos = estimate_hand_segment(wrist_pos, elbow_pos, right_shoulder_pos)
                
                # Calculate collision radius based on apparent distance from camera
                collision_radius = calculate_palm_size(wrist_pos, elbow_pos) * 0.5  # Thinner for line collision
                
                # Add to hand segments
                palm_positions.append({
                    "wrist": wrist_pos_calc,
                    "fingertips": fingertip_pos,
                    "radius": collision_radius,
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
                
                # Estimate hand segment (wrist to fingertips)
                wrist_pos_calc, fingertip_pos = estimate_hand_segment(wrist_pos, elbow_pos, left_shoulder_pos)
                
                # Calculate collision radius based on apparent distance from camera
                collision_radius = calculate_palm_size(wrist_pos, elbow_pos) * 0.5  # Thinner for line collision
                
                # Add to hand segments
                palm_positions.append({
                    "wrist": wrist_pos_calc,
                    "fingertips": fingertip_pos,
                    "radius": collision_radius,
                    "side": "left"
                })
    
    # Add current hand segments to history
    if palm_positions:
        palm_history.append(palm_positions)
        # Maintain history length
        if len(palm_history) > PALM_HISTORY_LENGTH:
            palm_history.pop(0)

def check_palm_collision():
    """
    Check for collisions between the ball and hand segments (line-based collision).
    Uses line segment collision detection for more realistic hand interaction.
    """
    global ball_pos, ball_velocity, ball_active, ball_on_ground, current_score, high_score, last_hit_was_bounce, shot_message, shot_message_time, last_hit_time, consecutive_hits
    
    if not ENABLE_PALM_DETECTION or not palm_history:
        return False
    
    collision_happened = False
    
    # Check all recent hand segments
    for positions in palm_history:
        for hand in positions:
            # Get hand segment endpoints
            wrist_pos = np.array(hand["wrist"], dtype=np.float32)
            fingertip_pos = np.array(hand["fingertips"], dtype=np.float32)
            collision_radius = hand["radius"]
            
            # Use line segment collision detection
            collision_threshold = BALL_RADIUS + collision_radius
            
            # Calculate distance from ball to hand segment line
            distance, closest_point = point_to_segment_distance(ball_pos, wrist_pos, fingertip_pos)
            
            if DEBUG_MODE:
                print(f"Distance to {hand['side']} hand segment: {distance:.1f}, threshold: {collision_threshold}")
            
            if distance < collision_threshold:
                # Activate the ball if it's not already active
                if not ball_active:
                    print(f"Ball activated by {hand['side']} hand hit!")
                    ball_active = True
                
                # Vector from closest point on hand to ball
                hand_to_ball_vector = ball_pos - closest_point
                hand_to_ball_length = np.linalg.norm(hand_to_ball_vector)
                
                if hand_to_ball_length > 0:
                    hand_to_ball_vector = hand_to_ball_vector / hand_to_ball_length
                else:
                    # Default direction if they're at the exact same point
                    hand_to_ball_vector = np.array([0, -1])
                
                # Calculate hand segment direction for more realistic physics
                hand_segment_vector = fingertip_pos - wrist_pos
                hand_segment_length = np.linalg.norm(hand_segment_vector)
                
                if hand_segment_length > 0:
                    hand_segment_unit = hand_segment_vector / hand_segment_length
                    # Normal to hand segment (perpendicular)
                    hand_normal = np.array([-hand_segment_unit[1], hand_segment_unit[0]])
                else:
                    hand_normal = hand_to_ball_vector
                
                # Use the normal that points towards the ball
                if np.dot(hand_normal, hand_to_ball_vector) < 0:
                    hand_normal = -hand_normal
                
                # Calculate reflection vector
                dot_product = np.dot(ball_velocity, hand_normal)
                reflection = ball_velocity - 2 * dot_product * hand_normal
                
                velocity_before = ball_velocity.copy()

                # Apply reflection with elasticity
                ball_velocity = reflection * ELASTICITY
                
                # Add a hit force in the direction away from the hand
                hand_force = 10.0  # Stronger force for hand hits
                
                # Apply force in direction away from hand (combination of normal and hand-to-ball)
                force_direction = (hand_normal + hand_to_ball_vector) * 0.5
                force_direction = force_direction / np.linalg.norm(force_direction)
                ball_velocity += force_direction * hand_force
                
                # Create hit particles
                particle_color = (0, 255, 150) if hand['side'] == 'right' else (255, 0, 255)
                create_hit_particles(ball_pos[0], ball_pos[1], ball_velocity, particle_color)
                
                # Move ball slightly away from hand to prevent multiple collisions
                ball_pos += hand_to_ball_vector * 5
                
                ball_on_ground = False

                # Enhanced scoring system with immediate validation
                current_time = time.time()
                
                # Check cooldown to prevent multiple points per touch
                if current_time - last_hit_time < HIT_COOLDOWN_DURATION:
                    return True  # Collision detected but no scoring
                
                last_hit_time = current_time
                
                # Detect volleyball shot type first (pass keypoint data for head detection)
                keypoint_data = last_valid_keypoints if 'last_valid_keypoints' in globals() else (None, None)
                shot_name, bonus_points = detect_volleyball_shot(palm_positions, ball_pos, velocity_before, ball_velocity, 
                                                                keypoint_data[0] if keypoint_data[0] is not None else None,
                                                                keypoint_data[1] if keypoint_data[1] is not None else None)
                
                # Immediate validation based on velocity change (no threading)
                velocity_magnitude_before = np.linalg.norm(velocity_before)
                velocity_magnitude_after = np.linalg.norm(ball_velocity)
                velocity_change = abs(velocity_magnitude_after - velocity_magnitude_before)
                
                # Simple immediate validation: check if ball is moving upward with sufficient force
                is_valid_hit = (velocity_change > 2.0 and ball_velocity[1] < -1.0)  # Ball moving up with force
                
                if is_valid_hit:
                    # Award points for successful hit
                    points_to_add = 1 + bonus_points
                    current_score += points_to_add
                    consecutive_hits += 1
                    
                    if current_score > high_score:
                        high_score = current_score
                    
                    # Show shot message for volleyball shots
                    volleyball_shots = ["SMASH!", "SET!", "DIG!"]
                    
                    if shot_name and shot_name in volleyball_shots:
                        # Show shot name with bonus points
                        if bonus_points > 0:
                            shot_message = f"{shot_name} +{bonus_points} pts!"
                        else:
                            shot_message = shot_name
                        shot_message_time = time.time()
                        print(f"{shot_name} Score: {current_score} (High: {high_score}) Hits: {consecutive_hits}")
                    else:
                        print(f"Good Hit! Score: {current_score} (High: {high_score}) Hits: {consecutive_hits}")
                else:
                    # Hit was not successful - no points awarded
                    if DEBUG_MODE:
                        print(f"Hit not awarded: velocity_change={velocity_change:.1f}, upward_velocity={ball_velocity[1]:.1f}")
                
                last_hit_was_bounce = True

                # Log collision for debugging
                if DEBUG_MODE:
                    print(f"Collision with {hand['side']} palm at position {ball_pos[0]:.1f}, {ball_pos[1]:.1f}")
                    print(f"New velocity: {ball_velocity[0]:.1f}, {ball_velocity[1]:.1f}")
                
                collision_happened = True
                break  # Only process one collision per frame
        
        if collision_happened:
            break
    
    return collision_happened

def detect_volleyball_shot(hand_positions, ball_pos, velocity_before, velocity_after, candidate=None, subset=None):
    """
    Simplified volleyball shot detection for Dig, Set, and Smash only.
    Returns: (shot_name, bonus_points) or (None, 0)
    """
    global last_volleyball_shot_time
    
    # Reduced cooldown for better responsiveness
    if time.time() - last_volleyball_shot_time < 0.5:
        return None, 0
    
    # Check if we have enough velocity change to be a meaningful hit
    velocity_magnitude_before = np.linalg.norm(velocity_before)
    velocity_magnitude_after = np.linalg.norm(velocity_after)
    velocity_change = abs(velocity_magnitude_after - velocity_magnitude_before)
    
    if velocity_change < 3.0:  # Higher threshold for cleaner detection
        return None, 0
    
    # Must have upward velocity component after hit
    if velocity_after[1] > -2.0:  # Ball must move upward with decent force
        return None, 0
    
    # Get hand position(s) for shot analysis
    if len(hand_positions) == 0:
        return None, 0
    
    # SMASH detection - Single hand, high velocity, hand above ball
    if len(hand_positions) == 1:
        hand = hand_positions[0]
        hand_pos = np.array(hand["fingertips"])
        
        # SMASH: High velocity downward strike with hand above ball
        if (velocity_magnitude_after > 10.0 and 
            hand_pos[1] < ball_pos[1] - 25 and  # Hand well above ball
            velocity_after[1] < -8.0):  # Strong downward-then-upward motion
            last_volleyball_shot_time = time.time()
            return "SMASH!", 5
    
    # SET and DIG detection - Two-handed shots
    if len(hand_positions) >= 2:
        left_hand = None
        right_hand = None
        
        for hand in hand_positions:
            if hand["side"] == "left":
                left_hand = hand
            elif hand["side"] == "right":
                right_hand = hand
        
        if left_hand and right_hand:
            left_pos = np.array(left_hand["fingertips"])
            right_pos = np.array(right_hand["fingertips"])
            
            # Calculate distance between hands and average position
            hand_distance = np.linalg.norm(left_pos - right_pos)
            avg_hand_pos = (left_pos + right_pos) / 2
            
            # Hands must be reasonably close together for team shots
            if hand_distance < 150:
                
                # SET: Simplified detection - hands and ball both above head level
                # Use actual head keypoints if available, otherwise fall back to screen position
                head_y = None
                
                if candidate is not None and subset is not None and len(subset) > 0:
                    # Find the person with hands (assuming first person for simplicity)
                    for person_idx in range(len(subset)):
                        # Check for nose (keypoint 0) or neck (keypoint 1) 
                        nose_idx = int(subset[person_idx][0])  # Nose
                        neck_idx = int(subset[person_idx][1])  # Neck
                        
                        if (nose_idx != -1 and nose_idx < len(candidate) and 
                            candidate[nose_idx][2] > KEYPOINT_CONFIDENCE_THRESHOLD):
                            head_y = candidate[nose_idx][1]
                            break
                        elif (neck_idx != -1 and neck_idx < len(candidate) and 
                              candidate[neck_idx][2] > KEYPOINT_CONFIDENCE_THRESHOLD):
                            head_y = candidate[neck_idx][1] 
                            break
                
                # Fall back to screen position if no head keypoints found
                if head_y is None:
                    head_y = CANVAS_HEIGHT // 3
                
                if (avg_hand_pos[1] < head_y and  # Hands above head level
                    ball_pos[1] < head_y and  # Ball also above head level
                    velocity_after[1] < -3.0):  # Ball goes up
                    last_volleyball_shot_time = time.time()
                    return "SET!", 3
                
                # DIG: Forearm pass, hands at or below ball level, strong upward motion
                elif (avg_hand_pos[1] >= ball_pos[1] - 10 and  # Hands at or below ball
                      velocity_after[1] < -6.0):  # Strong upward velocity (defensive save)
                    last_volleyball_shot_time = time.time()
                    return "DIG!", 4
    
    # No specific shot detected
    return None, 0

def scale_keypoints(candidate, subset, scale_factor=1.0):
    """Scale keypoints coordinates if detection was done at a different resolution (optimized)"""
    if scale_factor == 1.0:
        return candidate, subset
    
    # Optimized: Use numpy vectorized operations instead of loops and deep copy
    if len(candidate) > 0:
        scaled_candidate = np.array(candidate, dtype=np.float32)
        scaled_candidate[:, :2] *= scale_factor  # Scale only x,y coordinates
        return scaled_candidate.tolist(), subset
    
    return candidate, subset

def check_arm_collision(candidate, subset):
    """Check for collisions between the ball and arm segments"""
    global ball_pos, ball_velocity, ball_active, ball_on_ground, last_arm_positions, current_score, high_score, last_hit_was_bounce, last_hit_time
    
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
    
    # Check all recent arm positions to detect fast movements (optimized)
    collision_threshold = BALL_RADIUS + ARM_COLLISION_PADDING
    
    for positions in last_arm_positions:
        for name, start, end in positions:
            # Optimized: Quick bounding box check first
            min_x, max_x = min(start[0], end[0]), max(start[0], end[0])
            min_y, max_y = min(start[1], end[1]), max(start[1], end[1])
            
            # Early exit if ball is far from arm segment bounding box
            if (ball_pos[0] < min_x - collision_threshold or ball_pos[0] > max_x + collision_threshold or
                ball_pos[1] < min_y - collision_threshold or ball_pos[1] > max_y + collision_threshold):
                continue
            
            # Calculate distance from ball to arm segment
            distance, closest_point = point_to_segment_distance(ball_pos, start, end)
            
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
                    
                    # Check cooldown to prevent multiple points per touch
                    current_time = time.time()
                    if current_time - last_hit_time >= HIT_COOLDOWN_DURATION:
                        last_hit_time = current_time
                        
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
        img = None
        try:
            # Get the next frame from the queue with a timeout
            try:
                img = pose_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            
            # Enhanced image validation
            if img is None:
                print("Warning: None image in pose thread")
                pose_queue.task_done()
                continue
            
            if not hasattr(img, 'shape') or len(img.shape) != 3:
                print("Warning: Invalid image format in pose thread")
                pose_queue.task_done()
                continue
                
            if img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
                print("Warning: Empty image in pose thread")
                pose_queue.task_done()
                continue
            
            # Check if we need to resize for detection (fixed buffer issue)
            if USE_LOWER_RESOLUTION:
                # Resize for faster processing (back to standard approach)
                processed_img = cv2.resize(img, (DETECTION_WIDTH, DETECTION_HEIGHT), interpolation=cv2.INTER_LINEAR)
            else:
                processed_img = img
                
            # Run pose estimation with error handling
            try:
                candidate, subset = body_estimation(processed_img)
                if candidate is None or subset is None:
                    print("Warning: Pose estimation returned None")
                    pose_queue.task_done()
                    continue
            except Exception as pose_error:
                print(f"Error in pose estimation: {pose_error}")
                pose_queue.task_done()
                continue
            
            # Scale keypoints back to original size if needed
            if USE_LOWER_RESOLUTION:
                scale_factor = CANVAS_WIDTH / DETECTION_WIDTH
                candidate, subset = scale_keypoints(candidate, subset, scale_factor)
            
            # Put the result in the optimized circular buffer (lockless)
            pose_results.append((candidate, subset))
            new_pose_result = True
            
            # Mark task as done only if we successfully got an item
            pose_queue.task_done()
            
        except Exception as e:
            print(f"Error in pose thread: {e}")
            # Only mark task done if we actually got an item from the queue
            if img is not None:
                pose_queue.task_done()
    
    print("Pose estimation thread stopped")

# Start the pose estimation thread
pose_thread_running = True
pose_thread = threading.Thread(target=pose_estimation_thread_func)
pose_thread.daemon = True
pose_thread.start()

# Initialize menu system
menu = MenuSystem(CANVAS_WIDTH, CANVAS_HEIGHT)

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
        if current_time - fps_last_update > FPS_UPDATE_INTERVAL:
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
        
        # Get camera frame with enhanced error handling
        try:
            ret, oriImg = cap.read()
            if not ret:
                print("Failed to grab frame")
                time.sleep(0.01)
                continue
            
            if oriImg is None or oriImg.size == 0:
                print("Warning: Empty frame from camera")
                time.sleep(0.01)
                continue
                
        except Exception as camera_error:
            print(f"Camera error: {camera_error}")
            time.sleep(0.1)
            continue
            
        # Flip the image horizontally for more intuitive interaction
        oriImg = cv2.flip(oriImg, 1)
        
        # Create canvas copy efficiently (optimized)
        canvas = oriImg.copy()
        
        # Check for new pose results (optimized with circular buffer)
        if new_pose_result and pose_results:
            pose_result = pose_results[-1]  # Get latest result from circular buffer
            last_valid_keypoints = pose_result
            # Update hand segments based on the new pose
            if ENABLE_PALM_DETECTION:
                detect_hands(last_valid_keypoints[0], last_valid_keypoints[1])
            new_pose_result = False
        
        # Process pose detection on a schedule
        frame_count += 1
        
        # Only submit a new frame for pose analysis if the queue is empty
        # and it's time to do another detection (fixed copying issue)
        if frame_count % POSE_SKIP_FRAMES == 0 and pose_queue.empty():
            try:
                # Always make a copy to avoid threading issues
                img_for_thread = oriImg.copy()
                pose_queue.put_nowait(img_for_thread)  # Non-blocking put
            except queue.Full:
                pass  # Queue is full, skip this frame
        
        # Run physics updates only during gameplay
        if menu.is_playing():
            # Run fixed timestep physics updates (optimized with max iterations)
            physics_update_count = 0
            while accumulated_time >= FIXED_TIME_STEP and physics_update_count < MAX_PHYSICS_STEPS:
                physics_update_count += 1
                physics_counter += 1
                
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
                update_physics(FIXED_TIME_STEP)
                accumulated_time -= FIXED_TIME_STEP
            
            # Update particles
            update_particles(delta_time)
        else:
            # Reset accumulated time when not playing to prevent physics buildup
            accumulated_time = 0
        
        # Game-specific logic only during gameplay
        if menu.is_playing():
            # Update countdown if active
            countdown_number = update_countdown()
            
            # Check for auto-reset if ball has been on ground too long
            if ball_on_ground and ball_active and ball_ground_start_time > 0:
                if time.time() - ball_ground_start_time > AUTO_RESET_DURATION:
                    # End game when ball stays on ground too long
                    print("Game over - ball stayed on ground too long!")
                    menu.show_game_over(current_score)

        # Draw body pose if we have valid keypoints and display is enabled
        if DISPLAY_KEYPOINTS and last_valid_keypoints is not None:
            try:
                # Validate keypoints before drawing
                candidate, subset = last_valid_keypoints
                if candidate is not None and subset is not None and len(candidate) > 0:
                    canvas = util_fast.draw_bodypose(canvas, candidate, subset)
                
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
        
        # Draw hand segments if visual markings are enabled
        if DISPLAY_PALM_MARKINGS and palm_positions:
            for hand in palm_positions:
                wrist_pos = hand["wrist"]
                fingertip_pos = hand["fingertips"]
                radius = hand["radius"]
                hand_side = hand["side"]
                
                # Choose color based on hand side
                if hand_side == "right":
                    hand_color = (0, 255, 0)  # Green for right hand
                else:
                    hand_color = (255, 0, 255)  # Magenta for left hand
                
                # Draw hand segment line (wrist to fingertips)
                cv2.line(canvas, 
                        (int(wrist_pos[0]), int(wrist_pos[1])), 
                        (int(fingertip_pos[0]), int(fingertip_pos[1])), 
                        hand_color, 
                        max(2, int(radius * 0.5)))  # Line thickness based on radius
                
                # Draw wrist and fingertip markers
                cv2.circle(canvas, 
                          (int(wrist_pos[0]), int(wrist_pos[1])), 
                          6, 
                          hand_color, 
                          -1)  # Filled circle at wrist
                
                cv2.circle(canvas, 
                          (int(fingertip_pos[0]), int(fingertip_pos[1])), 
                          4, 
                          hand_color, 
                          -1)  # Smaller filled circle at fingertips
                
                # Show collision radius as text if in debug mode
                if DEBUG_MODE:
                    cv2.putText(canvas, 
                               f"{int(radius)}", 
                               (int(fingertip_pos[0] + 10), int(fingertip_pos[1])),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, hand_color, 1)
        
        # Draw game elements only during gameplay
        if menu.is_playing():
            # Draw floor line at bottom of screen
            cv2.line(canvas, (0, floor_y), (CANVAS_WIDTH, floor_y), (0, 255, 255), 1)
            
            # Draw ball trail effect
            if len(ball_trail) > 1:
                for i in range(len(ball_trail) - 1):
                    alpha = i / len(ball_trail)  # Fade out older trail points
                    trail_color = (int(100 * alpha), int(150 * alpha), int(255 * alpha))
                    thickness = max(1, int(7 * alpha))
                    
                    if i < len(ball_trail) - 1:
                        cv2.line(canvas, ball_trail[i], ball_trail[i + 1], trail_color, thickness)
            
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
        
        # Draw ball
        ball_center = (int(ball_pos[0]), int(ball_pos[1]))
        cv2.circle(canvas, ball_center, BALL_RADIUS, ball_color, -1)
        
        # Draw hit particles
        for particle in hit_particles:
            alpha = particle.life / particle.max_life
            color = tuple(int(c * alpha) for c in particle.color)
            size = max(1, int(3 * alpha))
            cv2.circle(canvas, (int(particle.x), int(particle.y)), size, color, -1)
        
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
        
        # Modern UI styling helper functions
        def draw_rounded_rect(img, pt1, pt2, color, thickness=-1, radius=15):
            """Draw a rounded rectangle"""
            x1, y1 = pt1
            x2, y2 = pt2
            
            # Create mask for rounded corners
            overlay = img.copy()
            
            # Draw main rectangle
            cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
            cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
            
            # Draw corner circles
            cv2.circle(overlay, (x1 + radius, y1 + radius), radius, color, thickness)
            cv2.circle(overlay, (x2 - radius, y1 + radius), radius, color, thickness)
            cv2.circle(overlay, (x1 + radius, y2 - radius), radius, color, thickness)
            cv2.circle(overlay, (x2 - radius, y2 - radius), radius, color, thickness)
            
            # Blend with original image for transparency effect
            alpha = 0.8
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        
        def draw_gradient_text(img, text, position, font, font_scale, color1, color2, thickness=2):
            """Draw text with gradient effect"""
            x, y = position
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            
            # Create gradient background
            gradient = np.zeros((text_size[1] + 10, text_size[0] + 20, 3), dtype=np.uint8)
            for i in range(gradient.shape[1]):
                ratio = i / gradient.shape[1]
                blended_color = [
                    int(color1[j] * (1 - ratio) + color2[j] * ratio) for j in range(3)
                ]
                gradient[:, i] = blended_color
            
            # Resize gradient to fit
            if gradient.shape[1] > 0 and gradient.shape[0] > 0:
                # Draw background with gradient
                roi_y1 = max(0, y - text_size[1] - 5)
                roi_y2 = min(img.shape[0], y + 5)
                roi_x1 = max(0, x - 10)
                roi_x2 = min(img.shape[1], x + text_size[0] + 10)
                
                if roi_y2 > roi_y1 and roi_x2 > roi_x1:
                    resized_gradient = cv2.resize(gradient, (roi_x2 - roi_x1, roi_y2 - roi_y1))
                    img[roi_y1:roi_y2, roi_x1:roi_x2] = cv2.addWeighted(
                        img[roi_y1:roi_y2, roi_x1:roi_x2], 0.3, resized_gradient, 0.7, 0)
            
            # Draw text with shadow effect
            cv2.putText(img, text, (x + 2, y + 2), font, font_scale, (0, 0, 0), thickness + 1)
            cv2.putText(img, text, (x, y), font, font_scale, color1, thickness)

        # Enhanced FPS display with modern styling
        fps_text = f'FPS: {current_fps:.1f}'
        draw_rounded_rect(canvas, (5, 5), (140, 35), (30, 30, 30), -1, 8)
        cv2.putText(canvas, fps_text, (12, 26), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 150), 2)
        
        # Enhanced ball state display with color coding
        if countdown_active:
            state_text = "COUNTDOWN"
            state_color = (0, 150, 255)  # Orange
        elif ball_active:
            if ball_on_ground and velocity_magnitude < 0.2:
                if ball_ground_start_time > 0:
                    time_remaining = AUTO_RESET_DURATION - (time.time() - ball_ground_start_time)
                    if time_remaining > 0:
                        state_text = f"RESTING ({time_remaining:.1f}s)"
                        state_color = (100, 100, 255)  # Light red
                    else:
                        state_text = "RESTING"
                        state_color = (100, 100, 255)
                else:
                    state_text = "RESTING"
                    state_color = (100, 100, 255)
            else:
                state_text = "ACTIVE"
                state_color = (0, 255, 100)  # Bright green
        else:
            state_text = "WAITING"
            state_color = (150, 150, 150)  # Gray
            
        # Draw enhanced state box
        state_size = cv2.getTextSize(state_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        draw_rounded_rect(canvas, (5, 45), (state_size[0] + 20, 75), (30, 30, 30), -1, 8)
        cv2.putText(canvas, state_text, (12, 66), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
        
        # Enhanced velocity display
        if ball_active:
            vel_text = f'VEL: {velocity_magnitude:.1f}'
            vel_intensity = min(255, int(velocity_magnitude * 20))
            vel_color = (0, 255 - vel_intensity, vel_intensity)  # Green to red based on speed
            
            draw_rounded_rect(canvas, (5, 85), (120, 115), (30, 30, 30), -1, 8)
            cv2.putText(canvas, vel_text, (12, 106), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, vel_color, 2)
            
            
            # Display position for debugging
            if DEBUG_MODE:
                cv2.putText(canvas, f'Pos: {ball_pos[0]:.0f},{ball_pos[1]:.0f}', (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(canvas, f'On ground: {ball_on_ground}', (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(canvas, f'Physics: {physics_rate}/s', (10, 180), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        

        # Enhanced score display with modern styling
        score_panel_width = 160
        score_panel_height = 80
        score_x = CANVAS_WIDTH - score_panel_width - 10
        score_y = 10
        
        # Draw score panel background with gradient effect
        draw_rounded_rect(canvas, (score_x, score_y), 
                         (score_x + score_panel_width, score_y + score_panel_height), 
                         (20, 20, 40), -1, 12)
        
        # Score text with glow effect
        score_text = f'SCORE: {current_score}'
        cv2.putText(canvas, score_text, (score_x + 3, score_y + 27), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)  # Shadow
        cv2.putText(canvas, score_text, (score_x, score_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 255), 2)  # Main text
        
        # High score with different color
        high_text = f'HIGH: {high_score}'
        cv2.putText(canvas, high_text, (score_x + 3, score_y + 57), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)  # Shadow
        cv2.putText(canvas, high_text, (score_x, score_y + 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)  # Gold color
                # Display countdown if active

        # Enhanced volleyball shot message with animations
        if shot_message and time.time() - shot_message_time < SHOT_MESSAGE_DURATION:
            # Calculate fade and pulse effects
            time_elapsed = time.time() - shot_message_time
            alpha = 1.0 - (time_elapsed / SHOT_MESSAGE_DURATION)
            pulse = 1.0 + 0.3 * np.sin(time_elapsed * 8)  # Pulsing effect
            
            # Dynamic text size based on pulse
            font_scale = 1.2 * pulse
            text_size = cv2.getTextSize(shot_message, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 4)[0]
            text_x = (CANVAS_WIDTH - text_size[0]) // 2
            text_y = CANVAS_HEIGHT // 3
            
            # Enhanced background with multiple layers
            padding = 30
            bg_x1 = text_x - padding
            bg_y1 = text_y - text_size[1] - padding
            bg_x2 = text_x + text_size[0] + padding
            bg_y2 = text_y + padding
            
            # Outer glow effect
            for i in range(5):
                glow_alpha = alpha * (0.8 - i * 0.15)
                glow_color = (int(50 * glow_alpha), int(200 * glow_alpha), int(255 * glow_alpha))
                draw_rounded_rect(canvas, 
                                (bg_x1 - i*3, bg_y1 - i*3), 
                                (bg_x2 + i*3, bg_y2 + i*3), 
                                glow_color, -1, 20 + i*2)
            
            # Main background
            draw_rounded_rect(canvas, (bg_x1, bg_y1), (bg_x2, bg_y2), 
                            (20, 20, 60), -1, 20)
            
            # Animated border
            border_color = (int(100 * alpha), int(255 * alpha), int(255 * alpha))
            draw_rounded_rect(canvas, (bg_x1, bg_y1), (bg_x2, bg_y2), 
                            border_color, 4, 20)
            
            # Enhanced text with multiple effects
            text_color = (int(255 * alpha), int(255 * alpha), int(255 * alpha))
            shadow_color = (0, 0, 0)
            
            # Shadow layers for depth
            for offset in [(4, 4), (2, 2)]:
                cv2.putText(canvas, shot_message, 
                           (text_x + offset[0], text_y + offset[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, shadow_color, 5)
            
            # Main text with gradient-like effect
            cv2.putText(canvas, shot_message, (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 4)
            
            # Highlight overlay
            highlight_color = (int(150 * alpha), int(255 * alpha), int(200 * alpha))
            cv2.putText(canvas, shot_message, (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, highlight_color, 2)
        else:
            shot_message = ""

        if countdown_active:
            countdown_num = update_countdown()
            if countdown_num is not None:
                # Enhanced countdown with pulsing animation
                countdown_text = str(countdown_num)
                
                # Pulsing effect based on time
                pulse_time = (time.time() - countdown_start_time) % 1.0
                pulse_scale = 1.0 + 0.5 * np.sin(pulse_time * 2 * np.pi * 2)
                
                font_scale = 4 * pulse_scale
                text_size = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 8)[0]
                text_x = (CANVAS_WIDTH - text_size[0]) // 2
                text_y = (CANVAS_HEIGHT + text_size[1]) // 2
                
                center_x = text_x + text_size[0] // 2
                center_y = text_y - text_size[1] // 2
                
                # Animated countdown colors
                countdown_colors = [
                    (0, 100, 255),  # Red for 3
                    (0, 200, 255),  # Orange for 2  
                    (0, 255, 100),  # Green for 1
                ]
                color_idx = max(0, min(len(countdown_colors) - 1, countdown_num - 1))
                main_color = countdown_colors[color_idx]
                
                # Multi-layer circle background with glow
                base_radius = 100
                for i in range(8):
                    glow_radius = int(base_radius * pulse_scale + i * 8)
                    glow_alpha = 0.8 - i * 0.1
                    glow_color = tuple(int(c * glow_alpha) for c in main_color)
                    cv2.circle(canvas, (center_x, center_y), glow_radius, glow_color, -1)
                
                # Main circle with gradient effect
                cv2.circle(canvas, (center_x, center_y), int(base_radius * pulse_scale), (20, 20, 40), -1)
                cv2.circle(canvas, (center_x, center_y), int(base_radius * pulse_scale), main_color, 6)
                
                # Inner highlight circle
                cv2.circle(canvas, (center_x, center_y), int(base_radius * pulse_scale * 0.8), 
                          tuple(int(c * 0.3) for c in main_color), -1)
                
                # Enhanced countdown number with multiple effects
                shadow_offset = int(6 * pulse_scale)
                
                # Multiple shadow layers for depth
                for offset in [(shadow_offset, shadow_offset), (shadow_offset//2, shadow_offset//2)]:
                    cv2.putText(canvas, countdown_text, 
                               (text_x + offset[0], text_y + offset[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 10)
                
                # Main countdown number
                cv2.putText(canvas, countdown_text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 8)
                
                # Highlight overlay
                cv2.putText(canvas, countdown_text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, main_color, 4)

        # Enhanced controls display at bottom
        controls_text = "R/SPACE: Start | Q: Quit | D: Debug | K: Keypoints | P: Palm Markings"
        controls_size = cv2.getTextSize(controls_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        controls_x = (CANVAS_WIDTH - controls_size[0]) // 2
        controls_y = CANVAS_HEIGHT - 15
        
        # Background for controls
        draw_rounded_rect(canvas, 
                         (controls_x - 10, controls_y - controls_size[1] - 5), 
                         (controls_x + controls_size[0] + 10, controls_y + 5), 
                         (25, 25, 25), -1, 8)
        
        # Controls text with better visibility
        cv2.putText(canvas, controls_text, (controls_x + 1, controls_y + 1), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)  # Shadow
        cv2.putText(canvas, controls_text, (controls_x, controls_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)  # Main text

        # Display menu or game based on state
        if menu.is_in_menu():
            # Draw menu on top of the camera feed or create a dark overlay
            menu.draw_current_screen(canvas)
            cv2.imshow('Ball Physics Demo - Volleyball Edition', canvas)
        else:
            # Display the processed video during gameplay
            cv2.imshow('Ball Physics Demo - Volleyball Edition', canvas)
        
        # Check for key presses - use waitKey(1) for maximum responsiveness
        key = cv2.waitKey(1) & 0xFF
        
        # Handle menu input if not playing
        if menu.is_in_menu():
            action = menu.handle_key_input(key)
            if action == "quit":
                break
            elif action == "start_game":
                # Initialize game state
                menu.set_game_start_time()
                start_countdown_spawn()
                print(f"Game started for player: {menu.current_player}")
            elif action == "restart_game":
                # Reset game state
                menu.set_game_start_time()
                current_score = 0
                start_countdown_spawn()
                print(f"Game restarted for player: {menu.current_player}")
        
        # Handle game controls when playing
        elif menu.is_playing():
            if key == ord('q'):
                # End game and show game over screen
                menu.show_game_over(current_score)
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
                # Toggle palm markings display (detection stays active)
                DISPLAY_PALM_MARKINGS = not DISPLAY_PALM_MARKINGS
                print(f"Palm markings display: {DISPLAY_PALM_MARKINGS}")
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