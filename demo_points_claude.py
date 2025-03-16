import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import torch
import time

from src import util_fast
from src.body_fast import Body

KEYPOINT_CONFIDENCE_THRESHOLD = 0.3  # Adjust this value as needed

body_estimation = Body('model/body_coco.pth')

print(f"Torch device: {torch.cuda.get_device_name()}")

# Ball properties
ball_pos = np.array([320, 240], dtype=np.float64)  # Ball position as float64
ball_velocity = np.array([0, 0], dtype=np.float64)  # Start with zero velocity
ball_radius = 20
gravity = 0.2
elasticity = 0.8  # Bounce factor

# Add at the top with other constants
friction = 0.99  # Slight air resistance

# Add a state variable to track if the ball is active
ball_active = False  # Start with the ball stationary

# Key to activate the ball (press 'space' to activate)
activate_key = ord(' ')

# Time tracking for consistent physics updates
previous_time = time.time()
fixed_time_step = 1/60.0  # Target 60 physics updates per second
accumulated_time = 0.0

cap = cv2.VideoCapture(0)

# Reduce camera resolution for better performance
cap.set(3, 320)  # Width
cap.set(4, 240)  # Height

# Set camera properties to improve performance
cap.set(cv2.CAP_PROP_FPS, 30)  # Request 30 FPS from the camera
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffering

# Only run pose detection every few frames
pose_skip_frames = 5
frame_count = 0
last_valid_keypoints = None

# FPS calculation variables
fps_update_interval = 0.5  # Update FPS every half second
fps_last_update = time.time()
fps_frame_count = 0
current_fps = 0

def point_to_segment_distance(p, v, w):
    """
    Calculates the minimum distance between a point `p` and a line segment defined by `v` and `w`.
    p: The point (ball position)
    v, w: The endpoints of the line segment (elbow and wrist)
    Returns the distance from the point to the segment.
    """
    # Vector from v to w
    segment_vector = w - v
    # Vector from v to p
    point_vector = p - v
    # Project point_vector onto segment_vector and compute the dot product
    segment_length_squared = np.dot(segment_vector, segment_vector)
    
    if segment_length_squared == 0:
        return np.linalg.norm(p - v)  # v and w are the same point
    
    projection = np.dot(point_vector, segment_vector) / segment_length_squared
    # Find the closest point on the segment by clamping the projection to [0, 1]
    projection = np.clip(projection, 0, 1)
    closest_point = v + projection * segment_vector
    
    # Return the distance from the point to the closest point on the segment
    return np.linalg.norm(p - closest_point)

# Reset ball function
def reset_ball(center_x=None, center_y=None):
    global ball_pos, ball_velocity, ball_active
    
    # If no position is provided, use the center of the screen
    if center_x is None or center_y is None:
        center_x, center_y = 160, 120  # Half of 320x240
    
    ball_pos = np.array([center_x, center_y], dtype=np.float64)
    ball_velocity = np.array([0, 0], dtype=np.float64)
    ball_active = False

# Improved wall collision function
def check_wall_collision(canvas_width, canvas_height):
    global ball_pos, ball_velocity
    collision_happened = False
    
    # Check for horizontal wall collisions
    if ball_pos[0] - ball_radius < 0:
        # Push ball away from wall
        ball_pos[0] = ball_radius + 2  
        # Reverse x velocity and apply proper elasticity (damping)
        ball_velocity[0] = abs(ball_velocity[0]) * elasticity  # Positive elasticity < 1
        collision_happened = True
    elif ball_pos[0] + ball_radius > canvas_width:
        ball_pos[0] = canvas_width - ball_radius - 2
        ball_velocity[0] = -abs(ball_velocity[0]) * elasticity  # Negative direction, positive elasticity
        collision_happened = True
    
    # Check for vertical wall collisions 
    if ball_pos[1] - ball_radius < 0:
        ball_pos[1] = ball_radius + 2
        ball_velocity[1] = abs(ball_velocity[1]) * elasticity
        collision_happened = True
    elif ball_pos[1] + ball_radius > canvas_height:
        ball_pos[1] = canvas_height - ball_radius - 2
        ball_velocity[1] = -abs(ball_velocity[1]) * elasticity
        collision_happened = True
    
    # Safety check: ensure velocity doesn't exceed maximum limit
    if collision_happened:
        # Cap maximum velocity to prevent extreme speeds
        max_velocity = 10.0
        current_magnitude = np.linalg.norm(ball_velocity)
        if current_magnitude > max_velocity:
            # Scale velocity vector down to maximum allowed
            ball_velocity = (ball_velocity / current_magnitude) * max_velocity
    
    return collision_happened

# Update physics with fixed time step
def update_physics(delta_time):
    global ball_pos, ball_velocity, ball_active
    
    if not ball_active:
        return
    
    # Apply physics only if the ball is active
    ball_velocity[1] += gravity  # Apply gravity to y-velocity
    ball_velocity *= friction    # Apply friction
    
    # Update position based on velocity
    ball_pos += ball_velocity * delta_time * 60.0  # Scale by time step
    
    # Cap maximum velocity
    max_velocity = 10.0
    current_magnitude = np.linalg.norm(ball_velocity)
    if current_magnitude > max_velocity:
        ball_velocity = (ball_velocity / current_magnitude) * max_velocity
      
    check_wall_collision(canvas.shape[1], canvas.shape[0])
    
    # Check if the ball has come to rest (very low velocity)
    if np.linalg.norm(ball_velocity) < 0.1:
        ball_velocity = np.array([0, 0], dtype=np.float64)

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
    
    # Cap delta time to prevent large jumps if the game freezes temporarily
    if delta_time > 0.1:
        delta_time = 0.1
    
    # Accumulate time for fixed physics updates
    accumulated_time += delta_time
    
    # Get camera frame
    ret, oriImg = cap.read()
    if not ret:
        print("Failed to grab frame")
        time.sleep(0.01)  # Small delay to prevent CPU hogging if camera fails
        continue
        
    # Flip the image horizontally for a more intuitive interaction
    oriImg = cv2.flip(oriImg, 1)
    
    # Create canvas first so we have dimensions for physics
    canvas = copy.deepcopy(oriImg)
    
    # Run fixed timestep physics updates
    while accumulated_time >= fixed_time_step:
        # Process pose detection on a schedule
        frame_count += 1
        if frame_count % pose_skip_frames == 0:
            try:
                candidate, subset = body_estimation(oriImg)
                if len(candidate) > 0 and subset.shape[0] > 0:
                    last_valid_keypoints = (candidate, subset)
            except Exception as e:
                print(f"Error in pose estimation: {e}")
        
        # Use last valid keypoints if available
        if last_valid_keypoints is not None:
            candidate, subset = last_valid_keypoints
            
            # Check for arm collision
            if candidate.shape[0] > 4:
                elbow_x, elbow_y, elbow_conf, _ = candidate[3]
                wrist_x, wrist_y, wrist_conf, _ = candidate[4]

                # Only consider keypoints above the confidence threshold
                if elbow_conf > KEYPOINT_CONFIDENCE_THRESHOLD and wrist_conf > KEYPOINT_CONFIDENCE_THRESHOLD:
                    segment_start = np.array([elbow_x, elbow_y])
                    segment_end = np.array([wrist_x, wrist_y])
                    distance = point_to_segment_distance(ball_pos, segment_start, segment_end)

                    if distance < ball_radius + 10:  # If the arm hits the ball
                        if not ball_active:
                            print("Ball activated by arm hit!")
                            ball_active = True
                        
                        # Calculate the normal vector to the segment
                        segment_vector = segment_end - segment_start
                        segment_length = np.linalg.norm(segment_vector)
                        
                        if segment_length > 0:  # Prevent division by zero
                            segment_unit_vector = segment_vector / segment_length
                            normal_vector = np.array([-segment_unit_vector[1], segment_unit_vector[0]])
                            
                            # Calculate proper reflection vector
                            dot_product = np.dot(ball_velocity, normal_vector)
                            reflection = ball_velocity - 2 * dot_product * normal_vector
                            
                            # Apply reflection with elasticity
                            ball_velocity = reflection * elasticity
                            
                            # Add some "hit force" based on arm direction
                            arm_force = 3.0  # Adjustable force multiplier
                            ball_velocity += segment_unit_vector * arm_force
                            
                            # Move ball slightly away from arm to prevent multiple collisions
                            ball_pos += normal_vector * 2
        
        # Update physics with fixed time step
        update_physics(fixed_time_step)
        accumulated_time -= fixed_time_step
    
    # Draw body pose if we have valid keypoints
    if last_valid_keypoints is not None:
        # Draw body pose more efficiently - this is an expensive operation
        try:
            canvas = util_fast.draw_bodypose(canvas, candidate, subset)
        except Exception as e:
            print(f"Error drawing body pose: {e}")
    
    # Draw ball with a dynamic color based on velocity
    velocity_magnitude = np.linalg.norm(ball_velocity)
    color_intensity = min(255, int(velocity_magnitude * 20))
    
    # Different color for active vs inactive ball
    if ball_active:
        ball_color = (0, color_intensity, 255 - color_intensity)  # Dynamic blue-green color
    else:
        ball_color = (0, 0, 255)  # Red for inactive ball
        
    cv2.circle(canvas, (int(ball_pos[0]), int(ball_pos[1])), ball_radius, ball_color, -1)
    
    # Only draw keypoint indices when FPS is good enough (optimization)
    if current_fps > 15 and last_valid_keypoints is not None:
        for idx, point in enumerate(candidate):
            x, y, confidence, _ = point
            if confidence > 0.5:  # Higher threshold for rendering text
                cv2.putText(canvas, str(idx), (int(x), int(y)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    
    # Display FPS and instructions on screen
    cv2.putText(canvas, f'FPS: {current_fps:.1f}', (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Display ball state
    state_text = "Ball: Active" if ball_active else "Ball: Waiting for hit"
    cv2.putText(canvas, state_text, (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Streamline on-screen instructions to reduce rendering overhead
    cv2.putText(canvas, "r: reset | space: activate | q: quit", (10, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow('demo', canvas)  # Display the processed video
    
    # Check for key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        # Reset ball position and state
        reset_ball()
        print("Ball reset")
    elif key == ord(' '):
        # Activate the ball with a small initial velocity
        if not ball_active:
            ball_active = True
            ball_velocity = np.array([1, -1], dtype=np.float64)  # Small initial jump
            print("Ball activated by keyboard")

cap.release()
cv2.destroyAllWindows()