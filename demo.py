import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import time
from src import model
from src import util_fast
from src.body import Body

# Initialize the body estimation model
body_estimation = Body('model/body_coco.pth')


# Ball properties
ball_pos = np.array([320, 240], dtype=np.float64)  # Ball position as float64
ball_velocity = np.array([1, 1], dtype=np.float64)  # Ball velocity as float64
ball_radius = 20
gravity = 0.5  # Simulated gravity
elasticity = -0.7  # Bounce factor

# Load test image
test_image = 'images/demo.jpg'
oriImg = cv2.imread(test_image)  # B,G,R order

# Perform body estimation
candidate, subset = body_estimation(oriImg)

# Create a copy of the image for drawing
canvas = copy.deepcopy(oriImg)
canvas = util_fast.draw_bodypose(canvas, candidate, subset)

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

# Ensure both elbow and wrist are detected (index 3 and 4)
if candidate.shape[0] > 4:
    elbow_x, elbow_y, elbow_conf, _ = candidate[3]
    wrist_x, wrist_y, wrist_conf, _ = candidate[4]

if elbow_conf > 0 and wrist_conf > 0:
    # Calculate the distance from the ball to the line segment between elbow and wrist
    segment_start = np.array([elbow_x, elbow_y])
    segment_end = np.array([wrist_x, wrist_y])
    distance = point_to_segment_distance(ball_pos, segment_start, segment_end)

    print('Distance from ball to line segment: ', distance)
    if distance < ball_radius + 10:  # If the ball is close to the segment
        print("CLOSE!")
        
        # Calculate the normal vector to the segment (direction opposite to the line)
        segment_vector = segment_end - segment_start
        segment_length = np.linalg.norm(segment_vector)
        segment_unit_vector = segment_vector / segment_length
        normal_vector = np.array([-segment_unit_vector[1], segment_unit_vector[0]])  # Perpendicular to segment
        
        # Reflect the ball's velocity along this normal vector
        ball_velocity = -np.dot(ball_velocity, segment_unit_vector) * segment_unit_vector + ball_velocity

# Update ball position
ball_pos += ball_velocity * 5  # Multiply velocity to simulate faster movement

# Ensure the ball stays inside the image bounds
ball_pos[0] = np.clip(ball_pos[0], ball_radius, oriImg.shape[1] - ball_radius)
ball_pos[1] = np.clip(ball_pos[1], ball_radius, oriImg.shape[0] - ball_radius)

# Draw the ball on the image
cv2.circle(canvas, (int(ball_pos[0]), int(ball_pos[1])), ball_radius, (0, 0, 255), -1)

# Draw keypoint indices on the canvas
for idx, point in enumerate(candidate):
    x, y, confidence, _ = point
    if confidence > 0:  # Only draw if the keypoint has a significant confidence
        cv2.putText(canvas, str(idx), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

# Display the image with keypoints and the ball
plt.imshow(canvas[:, :, [2, 1, 0]])  # Convert from BGR to RGB for display
plt.axis('off')  # Turn off axis
plt.show()
