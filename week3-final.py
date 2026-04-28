import RPi.GPIO as GPIO
import time
import cv2
import numpy as np
from picamera2 import Picamera2, Preview
import threading

#####MOTOR
ENA=19
ENB=12
leftP=13
leftN=5
rightP=27
rightN=22
duty=23 #24 0.3

####PID
kp = 0.33 # 0.279 #0.26 #0.18 #0.182
kd = 0.11 #0.17589 #0.05
error=0
integral=0
lasterror=0
correction=0

#####TURN ANGLE
x=360
y=360
tl=(x+1.02095)/91.21629
tr=(y+1.58547)/92.20735

#####THREAD
'''
Two threads run in parallel:

Vision thread (symbol detection)
Motor thread (movement)

Locks prevent data conflicts
'''
running = True
frame_lock = threading.Lock() #Prevents BOTH threads from editing/reading at the same time
shared_frame = None
shared_motor = None
frame_count = 0
frame_lock_counter = threading.Lock()
action = None
action_start = 0
action_duration = 0
symbol_data = {
    "detected": False,
    "name": None,
    "direction": None,
    "junction": False,
    "next_turn": None,
    "lock": threading.Lock()
}

#####SYMBOL
direction = None

#####COLORLINE
LOWER_RED1 = np.array([0, 95, 95])
UPPER_RED1 = np.array([180, 255, 255])
LOWER_RED2 = np.array([170, 120, 70])
UPPER_RED2 = np.array([180, 255, 255])
LOWER_YELLOW = np.array([19, 98, 98])
UPPER_YELLOW = np.array([30, 255, 255])

#####SETUP MOTOR
GPIO.setmode(GPIO.BCM)
GPIO.setup([leftP,leftN,rightP,rightN,ENA,ENB],GPIO.OUT)

####SETUP PICAM
blur=None
picam2 = Picamera2()
config = picam2.create_preview_configuration()#main={"size": (320, 240)})
picam2.configure(config)
picam2.start()
picam2.set_controls({"FrameRate": 5000})

####START MOTOR
pwmLeft = GPIO.PWM(ENA,50)  # 100Hz frequency
pwmRight = GPIO.PWM(ENB,50)
pwmLeft.start(duty)
pwmRight.start(duty)

#####ORB
orb = cv2.ORB_create(nfeatures=500,scaleFactor=1.2,nlevels=10,edgeThreshold=15,patchSize=31,fastThreshold = 10)# ORB will detect up to 800 keypoints (important points) in the image #changed
bf = cv2.BFMatcher(cv2.NORM_HAMMING) # finds important points (keypoints) in the image & creates a descriptor (a fingerprint) for each keypoint.

#####SYMBOLS
symbol_files = {
    "Fingerprint": "fingerprint.png", #print fingerprint
    "Button": "pressbutton.png", #stop
    "QRcode": "qr.png", #print QR
    "Warning": "warning.png", #stop
    "Recycle": "recycle.png" #turn 360
}

symbols = {}
frame_count = 0
num_corners = 0
approx = None
symbol_memory = []
last_triggered_symbol = None
last_stop_time = 0
stop_cooldown = 3   # seconds

for name, path in symbol_files.items():
    img = cv2.imread(path, 0)
    if img is None:
        print(f"Error loading {path}")
        continue

    key, des = orb.detectAndCompute(img, None)

    symbols[name] = {
        "name": name,
        "image": img,
        "kp": key,
        "des": des,
        "shape": img.shape
    }

def move():
    GPIO.output(leftP,GPIO.HIGH)
    GPIO.output(leftN,GPIO.LOW)
    GPIO.output(rightP,GPIO.HIGH)
    GPIO.output(rightN,GPIO.LOW)
    
def right():
    GPIO.output(leftP,GPIO.HIGH)
    GPIO.output(leftN,GPIO.LOW)
    GPIO.output(rightP,GPIO.LOW)
    GPIO.output(rightN,GPIO.HIGH)
    
def left():
    GPIO.output(leftP,GPIO.LOW)
    GPIO.output(leftN,GPIO.HIGH)
    GPIO.output(rightP,GPIO.HIGH)
    GPIO.output(rightN,GPIO.LOW)

def stop():
    GPIO.output(leftP,GPIO.LOW)
    GPIO.output(leftN,GPIO.LOW)
    GPIO.output(rightP,GPIO.LOW)
    GPIO.output(rightN,GPIO.LOW)
    pwmLeft.ChangeDutyCycle(0)
    pwmRight.ChangeDutyCycle(0)

def arrowType(approx):
    try:
        hull = cv2.convexHull(approx, returnPoints=False)
        defects = cv2.convexityDefects(approx, hull)
    except cv2.error:
        defects = None  # fallback: just skip convexity defects
    far_points = []

    if defects is not None:
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            far = tuple(approx[f][0])
            far_points.append(far)
            
    M = cv2.moments(approx)

    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
    arrow_tip = None
    max_dist = 0

    for point in approx:
        px, py = point[0]

        dist = np.sqrt((px-cx)**2 + (py-cy)**2)

        if dist > max_dist:
            max_dist = dist
            arrow_tip = (px, py)
            
    if arrow_tip is None:
        return "Arrow"

    dx = arrow_tip[0] - cx
    dy = arrow_tip[1] - cy
    
    if abs(dx) > abs(dy):

        if dx < 0:
            direction = "Turn Right"
        else:
            direction = "Turn Left"

    else:

        if dy < 0:
            direction = "Reverse"
        else:
            direction = "Forward"
    
    return direction

def symbol_thread_func():
    while running:
        with frame_lock:
            if shared_frame is None: 
                time.sleep(0.01)
                continue
            local_frame = shared_frame.copy()

        gray_frame = cv2.cvtColor(local_frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_frame, 150, 255, cv2.THRESH_BINARY)
        edged = cv2.Canny(thresh, 30, 100)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Detect ORB features in scene
        kp_scene, des_scene = orb.detectAndCompute(gray_frame, None)

        if des_scene is not None:
            for temp in templates:

                # Safety check
                if temp['des'] is None:
                    continue

                # =========================
                # MATCH FEATURES (KNN)
                # =========================
                matches = bf.knnMatch(temp['des'], des_scene, k=2)

                good = []
                for pair in matches:
                    if len(pair) < 2:
                        continue
                    m, n = pair
                    if m.distance < 0.72 * n.distance:
                        good.append(m)

                # =========================
                # HOMOGRAPHY
                # =========================
                if len(good) > 20:
                    src_pts = np.float32(
                        [temp['kp'][m.queryIdx].pt for m in good]
                    ).reshape(-1, 1, 2)

                    dst_pts = np.float32(
                        [kp_scene[m.trainIdx].pt for m in good]
                    ).reshape(-1, 1, 2)

                    M, mask = cv2.findHomography(
                        src_pts, dst_pts, cv2.RANSAC, 5.0
                    )

                    inliers = mask.ravel().sum()
                    if inliers < 30:
                        continue

                    if M is not None:
                        h, w = temp['shape']

                        # Template corners
                        pts = np.float32([
                            [0, 0],
                            [0, h - 1],
                            [w - 1, h - 1],
                            [w - 1, 0]
                        ]).reshape(-1, 1, 2)

                        # Project onto scene
                        dst = cv2.perspectiveTransform(pts, M)

                        area = cv2.contourArea(dst)

                        if area < 1000 or area > 50000:
                            continue

                        # Draw bounding box
                        local_frame = cv2.polylines(
                            local_frame,
                            [np.int32(dst)],
                            True,
                            (0, 255, 0),
                            3,
                            cv2.LINE_AA
                        )

                        # Label position (top-left corner)
                        label_pos = np.int32(dst[0][0])

                        cv2.putText(
                            local_frame,
                            temp['name'],
                            (label_pos[0], label_pos[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2
                        )

                        if symbol_data.get("name") != temp["name"]:
                            print(f"[SYMBOL]: {temp['name']}")
                        with symbol_data["lock"]:
                            symbol_data["detected"] = True
                            symbol_data["name"] = temp["name"]
                            
                        # Stop checking other templates for this frame
                        break

        # --- ARROW DETECTION ---
        found_dir = None
        arrow_detected = False
        warning_detected = False
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 800:
                continue
            
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
  
            if peri > 0:
                circularity = (4 * np.pi * area) / (peri * peri)
                
                #print(f"{circularity}")
                # If circularity is close to 1, it's a circle!
                if 0.45 <= circularity <= 0.95:
                    warning_detected = True
                    #print(f"{circularity}")
                    break

            if len(approx) == 9 and area > 2500:
                found_dir = arrowType(approx)
                arrow_detected = True
                break

        with symbol_data["lock"]:
            if arrow_detected:
                symbol_data["detected"] = True
                symbol_data["name"] = "arrow"
                symbol_data["direction"] = found_dir
                symbol_data["junction"] = True
                print(f"[SYMBOL]: {symbol_data['name']} | [DIRECTION]: {symbol_data['direction']}")
        
            elif warning_detected:
                symbol_data["detected"] = True
                symbol_data["name"] = "Warning"
                symbol_data["direction"] = None
                print(f"[SYMBOL]: {symbol_data['name']}")
            
            if symbol_data["name"] == "QRcode" or symbol_data["name"] == "Fingerprint":
                symbol_data["name"] = None
                

def motor_thread_func():
    global symbol_data, action, action_start, action_duration, lasterror, duty, lasterror
    action = None

    while running:
        with frame_lock:
            if shared_motor is None: 
                time.sleep(0.01)
                continue
            local_frame = shared_motor.copy()

        # --- COLOR/LINE DETECTION LOGIC ---
        hsv = cv2.cvtColor(local_frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, LOWER_RED1, UPPER_RED1)
        mask2 = cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)
        mask3 = cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)
        red_mask = cv2.add(mask1, mask2)

        gray = cv2.cvtColor(local_frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, black_mask = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY_INV)

        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        black_contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        target_contour = None
        track_type = "black"
        
        if red_contours and max([cv2.contourArea(c) for c in red_contours]) > 480:
            target_contour = max(red_contours, key=cv2.contourArea)
            track_type = "red" 
        elif cv2.countNonZero(mask3) > 500:  # yellow detection
            yellow_contours, _ = cv2.findContours(mask3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if yellow_contours:
                target_contour = max(yellow_contours, key=cv2.contourArea)
                track_type = "yellow"
        elif black_contours:
            target_contour = max(black_contours, key=cv2.contourArea)
      
        # --- PID CALCULATION ---
        if target_contour is not None: 
            M = cv2.moments(target_contour)
            if M["m00"] > 0:
                if track_type == "red":
                    active_mask = red_mask
                    symbol_data["junction"] = True
                elif track_type == "yellow":
                    active_mask = mask3
                    symbol_data["junction"] = True
                else:
                    active_mask = black_mask

                mid = int(M["m10"] / M["m00"])
                h, w = active_mask.shape 
                c = w // 2
                                
                if action is None:
                    error = mid - c
                    correction = (kp * error) + (kd * (error - lasterror))

                    left_speed = duty + correction
                    right_speed = duty - correction
                    
                    '''
                    # Clamping and Steering
                    if (-5 <= left_speed <= 5):
                        left_speed = 5
                    if (-5 <= right_speed <= 5):
                        right_speed = 5
                    '''
                    left_speed = max(-100, min(100, left_speed))
                    right_speed = max(-100, min(100, right_speed))
                    
                    if right_speed < 0 and left_speed >= 0: right()
                    elif left_speed < 0 and right_speed >= 0: left()
                    else: move()

                    pwmLeft.ChangeDutyCycle(abs(left_speed))
                    pwmRight.ChangeDutyCycle(abs(right_speed))
                    lasterror = error 
            
        else: # Search Logic (Line Lost)
            if lasterror < 0:
                pwmLeft.ChangeDutyCycle(0) 
                pwmRight.ChangeDutyCycle(45)
            else:
                pwmLeft.ChangeDutyCycle(45)
                pwmRight.ChangeDutyCycle(0)

        # --- SYMBOL ACTION LOGIC ---
        # We wrap the check in a lock for thread safety
        with symbol_data["lock"]:
            is_detected = symbol_data["detected"]
            sym_name = symbol_data["name"]
            sym_dir = symbol_data["direction"]

        if is_detected and action is None:
            action_start = time.time()
            if sym_name == "Recycle":
                action = "left"
                action_duration = tl - 1
            elif sym_name in ["Warning"]:
                action = "stop"
                action_duration = 3
            elif sym_name in ["Button"]:
                duty+=4
                action = "stop"
                action_duration = 3
            elif sym_dir == "Turn Left":
                symbol_data["next_turn"] = "left"
                action_duration = 0.7
            elif sym_dir == "Turn Right":
                symbol_data["next_turn"] = "right"
                action_duration = 0.7
            elif sym_dir == "Forward":
                symbol_data["next_turn"] = "forward"
                action_duration = 1.0

        #print(f"Turn: {symbol_data["next_turn"]}, Junction: {symbol_data["junction"]}")

        if symbol_data["junction"] and symbol_data["next_turn"] is not None and action is None:
            action_start = time.time()

            if symbol_data["next_turn"] == "left":
                action = "left"
                action_duration = 0.7
            elif symbol_data["next_turn"] == "right":
                action = "right"
                action_duration = 0.7
            elif symbol_data["next_turn"] == "forward":
                action = "move"
                action_duration = 0.8
            symbol_data["next_turn"] = None
        
        if symbol_data["next_turn"] is None:
            symbol_data["junction"] = False

        # Execute Action Timers
        if action is not None:
            elapsed = time.time() - action_start
            if action == "left":
                left()
                pwmLeft.ChangeDutyCycle(duty)
                pwmRight.ChangeDutyCycle(duty + 35)
            elif action == "right":
                right()
                pwmLeft.ChangeDutyCycle(duty + 35)
                pwmRight.ChangeDutyCycle(duty)
            elif action == "stop":
                stop()
            elif action == "move":
                move()
                pwmLeft.ChangeDutyCycle(duty)
                pwmRight.ChangeDutyCycle(duty)

            if elapsed >= action_duration:
                duty=24
                action = None
                symbol_data["name"] = None
                with symbol_data["lock"]:
                    symbol_data["detected"] = False
                move()
                pwmLeft.ChangeDutyCycle(duty)
                pwmRight.ChangeDutyCycle(duty)



try:
    # Initialize the templates list for the symbol thread
    templates = list(symbols.values())
    
    t1 = threading.Thread(target=symbol_thread_func, daemon=True)
    t2 = threading.Thread(target=motor_thread_func, daemon=True)
    t1.start()
    t2.start()

    move()
    pwmLeft.ChangeDutyCycle(duty)
    pwmRight.ChangeDutyCycle(duty)

    frame_count = 0

    while True:
        frame = picam2.capture_array()

        frame_count += 1
        
        if frame_count % 2 != 0:
            continue

        with frame_lock:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            shared_frame = cv2.resize(frame.copy(),(320,240))
            shared_motor = frame[370:,:]

        with symbol_data["lock"]:
            current = symbol_data["name"]
            
        # =========================
        # DISPLAY
        # =========================
        cv2.imshow('Symbol', shared_frame)
        cv2.imshow("roi", shared_motor)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
finally:
    running = False
    stop()
    GPIO.cleanup()
    picam2.stop()