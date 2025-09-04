from rpi_hardware_pwm import HardwarePWM
import time
import threading

pwm = HardwarePWM(pwm_channel=0,hz=50)
pwm.start(0)

# duty cycle: duty_cycle = (10.1)*(angle/180) + 2.5
def update_pwm(angle):
    duty_cycle = 10.1 * (angle / 180) + 2.5
    pwm.change_duty_cycle(duty_cycle)
    print(f"Moved to {angle} (duty_cycle: {duty_cycle:.2f})")

def move_servo(start_angle, end_angle, step=1, delay=1):
    if start_angle < end_angle:
        angle_range = range(int(start_angle), int(end_angle) + 1, int(step))
    else: 
        angle_range = range(int(start_angle), int(end_angle) - 1, -int(step))
    
    for angle in angle_range:
        update_pwm(angle)
        time.sleep(delay)

repeat_count = 1
move_servo(0,100, step = 1, delay=0.05)

move_servo(100, 0, step = 1, delay=0.05)
