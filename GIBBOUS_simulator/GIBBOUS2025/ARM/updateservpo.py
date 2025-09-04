from rpi_hardware_pwm import HardwarePWM
import time

pwm = HardwarePWM(pwm_channel = 0, hz=50, chip=2)
pwm.start(0)
# duty cycle: 	180duty_cycle = (10.1)*(angle/180) + 2.5


try:
	while True:
		"""
		for duty in range (25, 91, 1):
			pwm.change_duty_cycle(duty /10)
			print(f"Duty Cycle: {duty / 10}%")
			time.sleep(1)
			"""
		user_input = input("Enter Angle (0 to 180):")
		try:
			angle = float(user_input)
			if 0 <= angle <= 180:
				#adjusted_angle = angle/180
				duty_cycle = (10.1)*(angle/180) + 2.5
				pwm.change_duty_cycle(duty_cycle)
				print(f"Duty Cycle: {duty_cycle}%")
		except ValueError:
			print("Invalid")
			
except KeyboardInterrupt:
	pass
	
finally:
	pwm.stop()

