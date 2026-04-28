import RPi.GPIO as GPIO
import time
import cv2
import numpy as np
from picamera2 import Picamera2, Preview

#-motor
ENA=19
ENB=12
leftP=13
leftN=5
rightP=27
rightN=22

duty=35
on=True
blur=None
#-pid
kp = 0.18#18#0.181 #0.182 ###0.8
kd = 0.17589#0.1762#1758#1.55#0.10 #0.05   ##0.1758
error=0
integral=0
lasterror=0
correction=0


GPIO.setmode(GPIO.BCM)
GPIO.setup([leftP,leftN,rightP,rightN,ENA,ENB],GPIO.OUT)

#centre point of cam == 1296
picam2 = Picamera2() #initialize cam
#picam2.start_preview(Preview.QTGL) #show live cam window

preview_config = picam2.create_preview_configuration() #setup preview
picam2.configure(preview_config) #applies the config setup
picam2.start()

pwmLeft = GPIO.PWM(ENA,1000)  # 1kHz frequency
pwmRight = GPIO.PWM(ENB,1000)

pwmLeft.start(duty)
pwmRight.start(duty)



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

def detectcolor(roi_frame):
    # Convert the ROI frame to grayscale
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Threshold to find the black line
    _, mask = cv2.threshold(blur, 110, 255, cv2.THRESH_BINARY_INV)
    return mask

def findcenter(mask):
    mo = cv2.moments(mask)

    if mo['m00'] == 0:
        return None  #no line detected
    
    centerx = int(mo['m10'] / mo['m00'])

    return centerx


try:
    move()
    pwmLeft.ChangeDutyCycle(duty)
    pwmRight.ChangeDutyCycle(duty)
    
    while on:
        frame = picam2.capture_array()
        roi = frame[370:, :] 

        # 1. Faster Vision: Go straight to detectcolor without HSV conversion (threshold)
        mask = detectcolor(roi)
        
        # 4. Find the Center of the Line (Centroid)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #It finds all the shapes (outlines) in your image.
        
        if contours:
            # Find the biggest blob (the line)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Calculate Moments
            M = cv2.moments(largest_contour)
            
            if M['m00'] > 0: #if black detected
                # cx is the X-position of the line center
                mid = int(M['m10'] / M['m00'])
                #height, width
                h, w = mask.shape
                c=w//2#mask.shape[1]//2
                
                # Draw for debugging
                cv2.drawContours(roi, [largest_contour], -1, (90, 255, 0), 2) #(image, contours, contourIdx, color, thickness) *color follow BGR
                
                size=5
                # Vertical Line: (cx, cy-size) to (cx, cy+size)
                cv2.line(roi, (mid, 55-size), (mid, 55+size), (90, 255, 0), 2)
                # Horizontal Line: (cx-size, cy) to (cx+size, cy)
                cv2.line(roi, (mid - size, 55), (mid + size, 55), (90, 255, 0), 2)
                
                # show display
                cv2.imshow("ROI", roi)
                cv2.imshow("BLACKnWHITE", mask)
                if cv2.waitKey(1) == ord('q'):
                    break
                
                
                # 3. PID Correction
                error = mid - c
                correction = (kp * error) + (kd * (error - lasterror))
                #correction = max(-15, min(15, correction))                
                # 4. Standard Differential Steering
                left_speed = duty + correction
                right_speed = duty - correction
                # 5. Clamping
                left_speed = max(-100, min(100, left_speed))
                right_speed = max(-100, min(100, right_speed))
                # Apply to motors
                print("left: ",left_speed)
                print("right: ",right_speed)
                if right_speed<0 and left_speed>=0:
                    right()
                elif left_speed<0 and right_speed>=0:
                    left()
                else:
                    move()
                
                pwmLeft.ChangeDutyCycle(abs(left_speed))
                pwmRight.ChangeDutyCycle(abs(right_speed))
            
                lasterror = error # Save for the 'else' block
                
                
            else: #see all white
                # 6. FIXED SEARCH LOGIC
                # If negative error, line was on Left -> Spin Left
                if lasterror < 0:
                    pwmLeft.ChangeDutyCycle(0) 
                    pwmRight.ChangeDutyCycle(40)
                # If positive error, line was on Right -> Spin Right
                else:
                    pwmLeft.ChangeDutyCycle(40)
                    pwmRight.ChangeDutyCycle(0)
        

finally:
    stop()
    GPIO.cleanup()
    picam2.stop_preview()

