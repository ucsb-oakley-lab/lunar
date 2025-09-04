from skyfield.api import load, Topos
from datetime import datetime, timedelta
import RPi.GPIO as GPIO
import time
import math
import threading
#import datetime

servo_pin = 24
def setup_servo(pin):
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(pin, GPIO.OUT)
    

    # Using frequency 50 Hz
    pwm = GPIO.PWM(pin, 50)
    pwm.start(0) 
    return pwm


def set_servo_angle(pwm, angle):
    duty_cycle = ((500 + (1500*angle))/20000)*100  # Map angle to servo range (2% to 12%)
    pwm.ChangeDutyCycle(duty_cycle)
      # Allow the servo to reach the position
    time.sleep(0.5) # Allow the servo to reach the position
    pwm.ChangeDutyCycle(0) # Stop sending signal to prevent jitter
    #threading.Timer(60, lambda: pwm.ChangeDutyCycle(0)).start()


def moon_placement_azimuth(year, month, day, hour, minute, second, pwm):
    planets = load('/home/moonlight/skyfield-data/de421.bsp')
    ts = load.timescale()
    local_time = datetime(year, month, day, hour, minute, second)

    # Convert local time to UTC (adjust for your timezone offset, if any)
    utc_time = local_time + timedelta(minutes=0)  # Adjust this based on your timezone
    t = ts.utc(utc_time.year, utc_time.month, utc_time.day, utc_time.hour, utc_time.minute, utc_time.second)

    # Observation location (latitude and longitude)
    latitude = 0
    longitude = -119.8610
    location = Topos(latitude_degrees=latitude, longitude_degrees=longitude)

    moon = planets['moon']
    earth = planets['earth']

    astrometric = earth + location
    


def find_zenith():
    i = 0
    #nonlocal local_time
    while True:
        i += 1
        current_time = local_time + timedelta(minutes=(5*i))
        utc_time = current_time
        t = ts.utc(utc_time.year, utc_time.month, utc_time.day, utc_time.hour, utc_time.minute, utc_time.second)

        
        moon_position = astrometric.at(t).observe(moon).apparent()
        alt,_, _ = moon_position.altaz()
        altitude = alt.degrees
        zenith_angle = 90 - altitude
        zen = math.floor(zenith_angle)
        
        print(f"Time: {current_time:} Zenith = {zenith_angle:.2f}°")
        print (f"Altitude = {altitude:.2f}°")

        
        angle_percent = (min(180, zen))/180
        print(f"Angle Percent: {angle_percent:.2f}")
        servo_angle = max(0, angle_percent)
        """ 
        #newval = math.floor(servo_angle)
        #print(f"Floored angle: {newval:}")
        
        set_servo_angle(pwm, (servo_angle))
        #time.sleep(60)  #dont need i think 
        """
        return  servo_angle
def update_servo():
	while True:
		time.sleep(5)
		set_servo_angle(pwm, servo_angle)
		

update_thread = threading.Thread(target=update_servo, daemon=True)
continuous_thread = threading.Thread(target=find_zenith, daemon=True)

update_thread.start()
continuous_thread.start()

while True:
	time.sleep(1)


try:
    servo_pwm = setup_servo(servo_pin)
    set_servo_angle(servo_pwm, 0)
    print("Enter date and time:\n")
    year = int(input("Year: "))
    month = int(input("Month: "))
    day = int(input("Day: "))
    hour = int(input("Hour: "))
    minute = int(input("Minute: "))
    second = int(input("Seconds: "))

    # Set up the servo
    #servo_pwm = setup_servo(servo_pin)

    # Run the Moon azimuth function
    moon_placement_azimuth(year, month, day, hour, minute, second, servo_pwm)
    

finally:
    # Clean up GPIO resources
    servo_pwm.stop()
    GPIO.cleanup()
