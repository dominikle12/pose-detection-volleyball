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
ball_pos = np.array([320, 240])  # Initial position at screen center
ball_velocity = np.array([0, 0])  # Initial velocity
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
    
    # Extract right wrist position (COCO keypoint index 4)
    if candidate.shape[0] > 0 and candidate.shape[0] > 4:
        wrist_x, wrist_y = candidate[4][:2]
        if wrist_x > 0 and wrist_y > 0:
            distance = np.linalg.norm(ball_pos - np.array([wrist_x, wrist_y]))
            print('Distance from ball: ', distance)
            if distance < ball_radius + 10:  # If hand is close to ball
                print("CLOSE!")

    
    # Bounce off walls
    if ball_pos[0] - ball_radius < 0 or ball_pos[0] + ball_radius > 640:
        ball_velocity[0] *= elasticity
        ball_pos[0] = np.clip(ball_pos[0], ball_radius, 640 - ball_radius)
    if ball_pos[1] + ball_radius > 480:
        ball_velocity[1] *= elasticity
        ball_pos[1] = 480 - ball_radius
    
    # Draw ball
    cv2.circle(canvas, (int(ball_pos[0]), int(ball_pos[1])), ball_radius, (0, 0, 255), -1)
    cv2.addText
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
