import numpy as np
import math
import cv2
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Cache for colors to avoid recreation
LIMB_COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], 
               [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], 
               [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255], 
               [255, 0, 255], [255, 0, 170], [255, 0, 85]]

# Cache for limb sequence to avoid recreation
LIMB_SEQ = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
           [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
           [1, 16], [16, 18], [3, 17], [6, 18]]

# Cache for hand edges
HAND_EDGES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10],
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

def padRightDownCorner(img, stride, padValue):
    """Optimized padding function with pre-calculations and reduced operations"""
    h, w = img.shape[0], img.shape[1]
    
    # Calculate padding directly
    pad_down = 0 if (h % stride == 0) else stride - (h % stride)
    pad_right = 0 if (w % stride == 0) else stride - (w % stride)
    
    # Skip padding if not necessary
    if pad_down == 0 and pad_right == 0:
        return img, [0, 0, 0, 0]
    
    # Only pad what's needed
    if pad_down > 0:
        pad_down_const = np.full((pad_down, w, 3), padValue, dtype=img.dtype)
        img_padded = np.vstack((img, pad_down_const))
    else:
        img_padded = img
    
    if pad_right > 0:
        pad_right_const = np.full((img_padded.shape[0], pad_right, 3), padValue, dtype=img.dtype)
        img_padded = np.hstack((img_padded, pad_right_const))
    
    return img_padded, [0, 0, pad_down, pad_right]

def transfer(model, model_weights):
    """Transfer caffe model to pytorch which will match the layer name"""
    transfered_model_weights = {}
    state_dict_keys = model.state_dict().keys()
    
    # Build a dictionary for faster lookups
    model_weights_dict = {k: v for k, v in model_weights.items()}
    
    for weights_name in state_dict_keys:
        key = '.'.join(weights_name.split('.')[1:])
        if key in model_weights_dict:
            transfered_model_weights[weights_name] = model_weights_dict[key]
        else:
            print(f"Warning: {key} not found in model weights")
    
    return transfered_model_weights

def draw_bodypose(canvas, candidate, subset):
    """Optimized body pose drawing function"""
    # Skip drawing if no people detected
    if len(subset) == 0 or len(candidate) == 0:
        return canvas
    
    stickwidth = 4
    
    # Create a copy of the canvas only once
    overlay = canvas.copy()
    
    # Draw keypoints first (all at once for each person)
    for n in range(len(subset)):
        for i in range(18):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = int(candidate[index][0]), int(candidate[index][1])
            cv2.circle(canvas, (x, y), 4, LIMB_COLORS[i], thickness=-1)
    
    # Now draw limbs
    for n in range(len(subset)):
        for i in range(17):
            index = subset[n][np.array(LIMB_SEQ[i]) - 1]
            if -1 in index:
                continue
            
            # Get coordinates more efficiently
            Y = candidate[index.astype(int), 0].astype(int)
            X = candidate[index.astype(int), 1].astype(int)
            mX, mY = np.mean(X), np.mean(Y)
            
            # Use a simpler line drawing when possible
            if stickwidth <= 1:
                cv2.line(overlay, (Y[0], X[0]), (Y[1], X[1]), LIMB_COLORS[i], 2)
                continue
                
            # Calculate angle and length
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            
            # Create polygon points
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(overlay, polygon, LIMB_COLORS[i])
    
    # Blend the overlay with the original image once
    cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)
    return canvas

def draw_handpose_by_opencv(canvas, peaks, show_number=False):
    """Optimized hand pose drawing using OpenCV"""
    # Skip if no peaks
    if len(peaks) == 0:
        return canvas
    
    # Pre-calculate HSV colors to RGB once
    edge_colors = [matplotlib.colors.hsv_to_rgb([ie/float(len(HAND_EDGES)), 1.0, 1.0])*255 for ie in range(len(HAND_EDGES))]
    
    # Draw lines
    for ie, e in enumerate(HAND_EDGES):
        if np.sum(np.all(peaks[e], axis=1)==0)==0:
            x1, y1 = peaks[e[0]]
            x2, y2 = peaks[e[1]]
            cv2.line(canvas, (x1, y1), (x2, y2), edge_colors[ie], thickness=2)

    # Draw points
    for i, keypoint in enumerate(peaks):
        x, y = keypoint
        cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)
        if show_number:
            cv2.putText(canvas, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), lineType=cv2.LINE_AA)
    
    return canvas

def handDetect(candidate, subset, oriImg):
    """Optimized hand detector with faster calculations"""
    # Skip if no people detected
    if len(subset) == 0 or len(candidate) == 0:
        return []
    
    ratioWristElbow = 0.33
    detect_result = []
    image_height, image_width = oriImg.shape[0:2]
    
    for person in subset.astype(int):
        # Check for valid keypoints first (more efficient check)
        left_indices = person[[5, 6, 7]]
        right_indices = person[[2, 3, 4]]
        has_left = not np.any(left_indices == -1)
        has_right = not np.any(right_indices == -1)
        
        if not (has_left or has_right):
            continue
        
        hands = []
        
        # Process left hand
        if has_left:
            left_shoulder_index, left_elbow_index, left_wrist_index = left_indices
            x1, y1 = candidate[left_shoulder_index][:2]
            x2, y2 = candidate[left_elbow_index][:2]
            x3, y3 = candidate[left_wrist_index][:2]
            hands.append([x1, y1, x2, y2, x3, y3, True])
        
        # Process right hand
        if has_right:
            right_shoulder_index, right_elbow_index, right_wrist_index = right_indices
            x1, y1 = candidate[right_shoulder_index][:2]
            x2, y2 = candidate[right_elbow_index][:2]
            x3, y3 = candidate[right_wrist_index][:2]
            hands.append([x1, y1, x2, y2, x3, y3, False])
        
        # Calculate hand rectangles
        for x1, y1, x2, y2, x3, y3, is_left in hands:
            # Calculate wrist position relative to elbow
            dx_wrist_elbow = x3 - x2
            dy_wrist_elbow = y3 - y2
            
            # Calculate center of hand box
            x = x3 + ratioWristElbow * dx_wrist_elbow
            y = y3 + ratioWristElbow * dy_wrist_elbow
            
            # Calculate distances using squared distance first (faster than sqrt)
            dist_wrist_elbow_sq = dx_wrist_elbow**2 + dy_wrist_elbow**2
            dist_elbow_shoulder_sq = (x2 - x1)**2 + (y2 - y1)**2
            
            # Only compute sqrt if needed
            if 0.9**2 * dist_elbow_shoulder_sq > dist_wrist_elbow_sq:
                width = 1.5 * math.sqrt(0.9**2 * dist_elbow_shoulder_sq)
            else:
                width = 1.5 * math.sqrt(dist_wrist_elbow_sq)
            
            # Adjust to top-left corner
            half_width = width / 2
            x -= half_width
            y -= half_width
            
            # Clamp to image boundaries
            x = max(0, x)
            y = max(0, y)
            width = min(width, image_width - x, image_height - y)
            
            # Only add if big enough
            if width >= 20:
                detect_result.append([int(x), int(y), int(width), is_left])
    
    return detect_result

def npmax(array):
    """Optimized version of npmax"""
    # Flatten the array and find the max index
    flat_idx = array.argmax()
    
    # Convert flat index to 2D coordinates
    i, j = np.unravel_index(flat_idx, array.shape)
    return i, j