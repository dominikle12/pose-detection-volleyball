import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import torch
import time

from src import model
from src import util_fast
from src.body import Body

KEYPOINT_CONFIDENCE_THRESHOLD = 0.3  # Adjust this value as needed

body_estimation = Body('model/body_coco.pth')

print(f"Torch device: {torch.cuda.get_device_name()}")

# Ball properties
ball_pos = np.array([320, 240], dtype=np.float64)  # Ball position as float64
ball_velocity = np.array([1, 1], dtype=np.float64)  # Ball velocity as float64
ball_radius = 20
gravity = 0.2
elasticity = 0.8  # Bounce factor

# Add at the top with other constants
friction = 0.99  # Slight air resistance

# Add before updating ball position
ball_velocity *= friction

cap = cv2.VideoCapture(0)

# Reduce camera resolution
cap.set(3, 320)  # Half the resolution
cap.set(4, 240)

# Only run pose detection every few frames
pose_skip_frames = 5
frame_count = 0
last_valid_keypoints = None

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

# Improved wall collision function
def check_wall_collision(canvas_width, canvas_height):
    global ball_pos, ball_velocity
    collision_happened = False
    
    # Save original velocity magnitude for comparison
    original_magnitude = np.linalg.norm(ball_velocity)
    
    # Check for horizontal wall collisions
    if ball_pos[0] - ball_radius < 0:
        # Push ball away from wall
        ball_pos[0] = ball_radius + 2  
        # Reverse x velocity and apply proper elasticity (damping)
        ball_velocity[0] = abs(ball_velocity[0]) * 0.8  # Positive elasticity < 1
        collision_happened = True
    elif ball_pos[0] + ball_radius > canvas_width:
        ball_pos[0] = canvas_width - ball_radius - 2
        ball_velocity[0] = -abs(ball_velocity[0]) * 0.8  # Negative direction, positive elasticity
        collision_happened = True
    
    # Check for vertical wall collisions 
    if ball_pos[1] - ball_radius < 0:
        ball_pos[1] = ball_radius + 2
        ball_velocity[1] = abs(ball_velocity[1]) * 0.8
        collision_happened = True
    elif ball_pos[1] + ball_radius > canvas_height:
        ball_pos[1] = canvas_height - ball_radius - 2
        ball_velocity[1] = -abs(ball_velocity[1]) * 0.8
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

while True:
    start_time = time.time()  # Start time for FPS calculation
    ret, oriImg = cap.read()
    # Flip the image horizontally for a more intuitive interaction
    oriImg = cv2.flip(oriImg, 1)
    frame_count += 1
    if frame_count % pose_skip_frames == 0:
        candidate, subset = body_estimation(oriImg)
        if len(candidate) > 0 and subset.shape[0] > 0:
            last_valid_keypoints = (candidate, subset)
    elif last_valid_keypoints is not None:
        candidate, subset = last_valid_keypoints
    else:
        continue
    canvas = copy.deepcopy(oriImg)
    canvas = util_fast.draw_bodypose(canvas, candidate, subset)
    
    # Check if wrist and elbow are detected
    if candidate.shape[0] > 4:
        elbow_x, elbow_y, elbow_conf, _ = candidate[3]
        wrist_x, wrist_y, wrist_conf, _ = candidate[4]

        # Only consider keypoints above the confidence threshold
        if elbow_conf > KEYPOINT_CONFIDENCE_THRESHOLD and wrist_conf > KEYPOINT_CONFIDENCE_THRESHOLD:
            # Your existing collision detection code here            
            # Calculate the distance from the ball to the line segment between elbow and wrist
            segment_start = np.array([elbow_x, elbow_y])
            segment_end = np.array([wrist_x, wrist_y])
            distance = point_to_segment_distance(ball_pos, segment_start, segment_end)

            print('Distance from ball to line segment: ', distance)
            if distance < ball_radius + 10:  # If the ball is close to the segment
                print("CLOSE!")
                
                # Calculate the normal vector to the segment
                segment_vector = segment_end - segment_start
                segment_length = np.linalg.norm(segment_vector)
                
                if segment_length > 0:  # Prevent division by zero
                    segment_unit_vector = segment_vector / segment_length
                    normal_vector = np.array([-segment_unit_vector[1], segment_unit_vector[0]])  # Perpendicular to segment
                    
                    # Calculate proper reflection vector
                    dot_product = np.dot(ball_velocity, normal_vector)
                    reflection = ball_velocity - 2 * dot_product * normal_vector
                    
                    # Apply reflection with elasticity
                    ball_velocity = reflection * -elasticity
                    
                    # Add some "hit force" based on arm direction
                    arm_force = 3.0  # Adjustable force multiplier
                    ball_velocity += segment_unit_vector * arm_force
                    
                    # Move ball slightly away from arm to prevent multiple collisions
                    ball_pos += normal_vector * 2
    print('Velocity: ', ball_velocity)
    # Update ball position
    # Add before updating ball position
    ball_velocity[1] += gravity  # Apply gravity to y-velocity
    ball_velocity *= friction    # Apply friction
    ball_pos += ball_velocity 

    max_velocity = 10.0
    current_magnitude = np.linalg.norm(ball_velocity)
    if current_magnitude > max_velocity:
        ball_velocity = (ball_velocity / current_magnitude) * max_velocity
      
    check_wall_collision(canvas.shape[1], canvas.shape[0])
    
    # Draw ball with a dynamic color based on velocity
    velocity_magnitude = np.linalg.norm(ball_velocity)
    color_intensity = min(255, int(velocity_magnitude * 20))
    ball_color = (0, color_intensity, 255 - color_intensity)
    cv2.circle(canvas, (int(ball_pos[0]), int(ball_pos[1])), ball_radius, ball_color, -1)
    
    # Draw keypoint indices on the canvas
    for idx, point in enumerate(candidate):
        x, y, confidence, _ = point
        if confidence > 0:  # Only draw if the keypoint has a significant confidence
            cv2.putText(canvas, str(idx), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    
    # Calculate FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    
    # Display FPS on screen
    cv2.putText(canvas, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('demo', canvas)  # Display the processed video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
