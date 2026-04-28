import RPi.GPIO as GPIO
import time

ENA=19
ENB=12
leftP=13
leftN=5
rightP=27
rightN=22
encoder_L=24
encoder_R=25
servo=18
distancePerPulse=0.010210176 #0.00903614458 #0.010210176
duty=45

#####
choice=2 #(0=forward same speed, 1=backward same speed, 2=rotation, 3=forward vary speed)
d=1
speed=0.29
isLeft=True #True
x=90
y=90
#####

tl=(x+1.02095)/91.21629
tr=(y+1.58547)/92.20735
t=(d+0.00884146)/0.411829 #duty==45
time1=d/speed

count_L=0
count_R=0

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup([leftP,leftN,rightP,rightN,ENA,ENB],GPIO.OUT)
GPIO.setup(encoder_R, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(encoder_L, GPIO.IN, pull_up_down=GPIO.PUD_UP)

pwmLeft = GPIO.PWM(ENA,100)
pwmRight = GPIO.PWM(ENB,100)

pwmLeft.start(duty)
pwmRight.start(duty)

def leftPulse(channel):
    global count_L
    count_L+=1

def rightPulse(channel):
    global count_R
    count_R+=1
 
GPIO.add_event_detect(encoder_L, GPIO.RISING, callback=leftPulse)
GPIO.add_event_detect(encoder_R, GPIO.RISING, callback=rightPulse)

def move():
    GPIO.output(leftP,GPIO.HIGH)
    GPIO.output(leftN,GPIO.LOW)
    GPIO.output(rightP,GPIO.HIGH)
    GPIO.output(rightN,GPIO.LOW)
    pwmLeft.ChangeDutyCycle(duty)
    pwmRight.ChangeDutyCycle(duty)
    
def reverse():
    GPIO.output(leftP,GPIO.LOW)
    GPIO.output(leftN,GPIO.HIGH)
    GPIO.output(rightP,GPIO.LOW)
    GPIO.output(rightN,GPIO.HIGH)
    pwmLeft.ChangeDutyCycle(duty)
    pwmRight.ChangeDutyCycle(duty)
    
def left():
    GPIO.output(leftP,GPIO.LOW)
    GPIO.output(leftN,GPIO.HIGH)
    GPIO.output(rightP,GPIO.HIGH)
    GPIO.output(rightN,GPIO.LOW)
    pwmLeft.ChangeDutyCycle(duty)
    pwmRight.ChangeDutyCycle(duty)
    
def right():
    GPIO.output(leftP,GPIO.HIGH)
    GPIO.output(leftN,GPIO.LOW)
    GPIO.output(rightP,GPIO.LOW)
    GPIO.output(rightN,GPIO.HIGH)
    pwmLeft.ChangeDutyCycle(duty)
    pwmRight.ChangeDutyCycle(duty)

def stop():
    GPIO.output(leftP,GPIO.LOW)
    GPIO.output(leftN,GPIO.LOW)
    GPIO.output(rightP,GPIO.LOW)
    GPIO.output(rightN,GPIO.LOW)
    pwmLeft.ChangeDutyCycle(0)
    pwmRight.ChangeDutyCycle(0)


try:
    if choice==0:
        move()
        time.sleep(t)
        distance=((count_L+count_R)/2)*distancePerPulse
        print("%.2fm"%(distance))
        print("\nl={0}, r={1}".format(count_L,count_R))
        
    elif choice==1:
        reverse()
        time.sleep(t)
        distance=((count_L+count_R)/2)*distancePerPulse
        print("%.2fm"%(distance))
        print("\nl={0}, r={1}".format(count_L,count_R))
        
    elif choice==2:
        if isLeft:
            left()
            time.sleep(tl)
        else:
            right()
            time.sleep(tr)
            
    elif choice==3:
        duty=(speed-0.04175)/0.00802143 #speed varying equation
        move()
        GPIO.sleep(time1)
        distance=((count_L+count_R)/2)*distancePerPulse
        print("%.2fm"%(distance))
        print("\nl={0}, r={1}".format(count_L,count_R))
        
    stop()

finally:
    GPIO.cleanup() #resets the pins to a safe, neutral state (usually high-impedance inputs), which prevents accidental short circuits if you start moving wires around after the script ends