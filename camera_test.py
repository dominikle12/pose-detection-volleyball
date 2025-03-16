import cv2
import time

# Just test if we can show a window with the camera feed
cap = cv2.VideoCapture(0)
cap.set(3, 320)
cap.set(4, 240)

print("Starting camera test...")
print("Press Q to quit")

# Force window creation before loop
cv2.namedWindow("Camera Test", cv2.WINDOW_NORMAL)
print("Window created")

try:
    for i in range(100):  # Try just 100 frames
        print(f"Frame {i}")
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        cv2.imshow("Camera Test", frame)
        print(f"Displayed frame {i}")
        
        # Use a short wait time
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Q pressed, exiting")
            break
            
        time.sleep(0.1)  # Add a small delay
        
except Exception as e:
    print(f"Error: {e}")
finally:
    print("Releasing camera")
    cap.release()
    cv2.destroyAllWindows()
    print("Done")