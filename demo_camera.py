import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import torch
import time

from src import model
from src import util_fast
from src.body import Body

body_estimation = Body('model/body_coco.pth')

print(f"Torch device: {torch.cuda.get_device_name()}")

# Ball properties
ball_pos = np.array([320, 240], dtype=np.float64)  # Ball position as float64
ball_velocity = np.array([1, 1], dtype=np.float64)  # Ball velocity as float64
ball_radius = 20
gravity = 0.5  # Simulated gravity
elasticity = -0.7  # Bounce factor


cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
while True:
    start_time = time.time()  # Start time for FPS calculation
    ret, oriImg = cap.read()
    candidate, subset = body_estimation(oriImg)
    canvas = copy.deepcopy(oriImg)
    canvas = util_fast.draw_bodypose(canvas, candidate, subset)
    

    if candidate.shape[0] > 4:  # Ensure both elbow and wrist are detected (index 3 and 4)
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
