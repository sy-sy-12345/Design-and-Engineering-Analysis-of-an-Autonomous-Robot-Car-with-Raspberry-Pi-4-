import cv2
import numpy as np
import time
from picamera2 import Picamera2

# 1. Initialize Camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (320, 240)})
picam2.configure(config)
picam2.start()

def nothing(x):
    pass

# 2. Create Window and Trackbars
cv2.namedWindow("Red_Threshold_Settings")
cv2.resizeWindow("Red_Threshold_Settings", 400, 400)

# Trackbars for Lower Red (using the 160-180 range as it's usually more vivid)
cv2.createTrackbar("Low_H", "Red_Threshold_Settings", 160, 180, nothing)
cv2.createTrackbar("Low_S", "Red_Threshold_Settings", 100, 255, nothing)
cv2.createTrackbar("Low_V", "Red_Threshold_Settings", 100, 255, nothing)

# Trackbars for Upper Red
cv2.createTrackbar("Up_H", "Red_Threshold_Settings", 180, 180, nothing)
cv2.createTrackbar("Up_S", "Red_Threshold_Settings", 255, 255, nothing)
cv2.createTrackbar("Up_V", "Red_Threshold_Settings", 255, 255, nothing)

print("--- CALIBRATION MODE ---")
print("1. Adjust sliders until your RED line is purely WHITE in the Mask window.")
print("2. Press 'q' to save and exit.")

try:
    while True:
# 1. Capture frame (comes in as RGB)
        frame_raw = picam2.capture_array()
        
        # 2. CONVERT RGB TO BGR (This makes Red look Red!)
        frame = cv2.cvtColor(frame_raw, cv2.COLOR_RGB2BGR)
        
        # 3. Convert BGR to HSV for your sliders
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Get current trackbar positions
        lh = cv2.getTrackbarPos("Low_H", "Red_Threshold_Settings")
        ls = cv2.getTrackbarPos("Low_S", "Red_Threshold_Settings")
        lv = cv2.getTrackbarPos("Low_V", "Red_Threshold_Settings")
        uh = cv2.getTrackbarPos("Up_H", "Red_Threshold_Settings")
        us = cv2.getTrackbarPos("Up_S", "Red_Threshold_Settings")
        uv = cv2.getTrackbarPos("Up_V", "Red_Threshold_Settings")
        
        # Define ranges
        lower_red = np.array([lh, ls, lv])
        upper_red = np.array([uh, us, uv])
        
        # Create mask
        mask = cv2.inRange(hsv, lower_red, upper_red)
        
        # Show windows
        cv2.imshow("Live Camera", frame)
        cv2.imshow("Red Mask Result", mask)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n--- YOUR CALIBRATED VALUES ---")
            print(f"LOWER_RED = np.array([{lh}, {ls}, {lv}])")
            print(f"UPPER_RED = np.array([{uh}, {us}, {uv}])")
            break


finally:
    picam2.stop()
    cv2.destroyAllWindows()