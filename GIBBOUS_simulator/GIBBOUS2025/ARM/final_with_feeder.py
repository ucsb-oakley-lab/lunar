import time
import math
import datetime
import threading
from queue import Queue
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import sys
from waveshare_OLED import OLED_1in27_rgb
from PIL import Image, ImageDraw, ImageFont
from rpi_hardware_pwm import HardwarePWM

picdir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'pic')
libdir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'lib')
if os.path.exists(libdir):
    sys.path.append(libdir)

SUNSET_HOUR  = 18
SUNRISE_HOUR = 6
DEFAULT_LUNAR_CYCLE_LENGTH = 28
SHAKE_NUM = 5
SUN_COLOR = 'E56020'
LUNAR_PHASES = [
    'Full Moon',
    'Waning Gibbous',
    'Last Quarter',
    'Waning Crescent',
    'New Moon',
    'Waxing Crescent',
    'First Quarter',
    'Waxing Gibbous'
]

MoonPhaseLengthDays = {
    'Full Moon':       1,
    'Waning Gibbous':  6,
    'Last Quarter':    1,
    'Waning Crescent': 6,
    'New Moon':        1,
    'Waxing Crescent': 6,
    'First Quarter':   1,
    'Waxing Gibbous':  6
}

MoonPhaseChecker = {
    'Full Moon':       'x',
    'Waning Gibbous':  'o',
    'Last Quarter':    'x',
    'Waning Crescent': 'o',
    'New Moon':        'x',
    'Waxing Crescent': 'o',
    'First Quarter':   'x',
    'Waxing Gibbous':  'o'
}

DEFAULT_LATER_PER_DAY = (50 * 28) / 29

def set_servo_angle(angle):
    pwm = HardwarePWM(pwm_channel=0, hz=50)
    pwm.start(0)
    duty_cycle = 2.6 + 6.5 * (angle / 180.0)
    pwm.change_duty_cycle(duty_cycle)
    return duty_cycle


def set_feeder_angle(feeder_angle):
    #pwm_channel = 1 = pin 13? double check in config.txt file
    pwm = HardwarePWM(pwm_channel=1, hz=50)
    pwm.start(0)
    duty_cycle = 2.6 + 10.1 * (feeder_angle / 180.0)
    pwm.change_duty_cycle(duty_cycle)
    return duty_cycle

def move_feeder(start_angle, end_angle, delay, step=1):
    if start_angle < end_angle:
        angle_range = range(int(start_angle), int(end_angle) + 1, int(step))
    else:
        angle_range = range(int(start_angle), int(end_angle) - 1, -int(step))
    
    for angle in angle_range:
        set_feeder_angle(angle)
        print(f"Moving feeder to {angle}°")
        time.sleep(delay)

def drop_feeder():
    move_feeder(0, 120, step=1, delay=0.05)

def reset_feeder():
    move_feeder(120, 0, step=1, delay=0.05)

def shake_feeder():
    #drop_feeder()
    # Shake the feeder by moving it back and forth
    for _ in range(SHAKE_NUM):
        move_feeder(120, 80, step=5, delay=100)
        move_feeder(80, 120, step=5, delay=100)
    reset_feeder()
    
def get_num_phases(target_cycle_length):
    scalar = target_cycle_length / DEFAULT_LUNAR_CYCLE_LENGTH
    scaled_phases = {}

    for phase in LUNAR_PHASES:
        base_length = MoonPhaseLengthDays[phase]
        if scalar >= 1:
            if MoonPhaseChecker[phase] == 'x':
                new_length = math.floor(scalar * base_length)
            else:
                new_length = math.ceil(scalar * base_length)
        else:
            # scalar < 1
            if MoonPhaseChecker[phase] == 'x':
                new_length = math.ceil(scalar * base_length)
            else:
                new_length = math.floor(scalar * base_length)

        scaled_phases[phase] = new_length

    new_total = sum(scaled_phases.values())

    # Adjust if sum doesn't match target
    if new_total != target_cycle_length:
        if new_total > target_cycle_length:
            # Reduce days from 'o' phases
            for phase in MoonPhaseChecker:
                if MoonPhaseChecker[phase] == 'o' and scaled_phases[phase] > 0:
                    scaled_phases[phase] -= 1
                    new_total = sum(scaled_phases.values())
                    if new_total == target_cycle_length:
                        break
        else:
            # Add days to 'o' phases
            for phase in MoonPhaseChecker:
                if MoonPhaseChecker[phase] == 'o':
                    scaled_phases[phase] += 1
                    new_total = sum(scaled_phases.values())
                    if new_total == target_cycle_length:
                        break

    return scaled_phases, new_total


def calculate_moonrise_times(target_cycle_length):
    scaled_phases, new_total_days = get_num_phases(target_cycle_length)
    scale_factor = float(target_cycle_length) / 29.5
    kickback = DEFAULT_LATER_PER_DAY / scale_factor
    base_time_minutes = SUNSET_HOUR * 60

    phase_sequence = []
    for phase_name in LUNAR_PHASES:
        count = scaled_phases[phase_name]
        phase_sequence.extend([phase_name] * count)

    results = []
    last_new_moon_day = None

    for day in range(new_total_days):
        this_phase = phase_sequence[day]
        if this_phase == "New Moon":
            moonrise_time = None
            moonset_time  = None
            last_new_moon_day = day
        else:
            if last_new_moon_day is None:
                # Pre-New Moon
                offset_minutes = int(round(day * kickback))
                total_rise_minutes = base_time_minutes + offset_minutes
                rise_hour   = (total_rise_minutes // 60) % 24
                rise_minute = total_rise_minutes % 60
                moonrise_time = datetime.time(rise_hour, rise_minute)
                moonset_time  = datetime.time(6, 0)
            else:
                # Post-New Moon
                moonrise_time = datetime.time(18, 0)
                days_since_new_moon = day - last_new_moon_day
                offset_after_new_moon = int(round(days_since_new_moon * kickback))
                total_set_minutes = (SUNSET_HOUR * 60) + offset_after_new_moon
                set_hour   = (total_set_minutes // 60) % 24
                set_minute = total_set_minutes % 60
                moonset_time = datetime.time(set_hour, set_minute)

        # Calculate total_visibility in seconds
        if moonrise_time and moonset_time:
            today = datetime.date.today()
            rise_dt = datetime.datetime.combine(today, moonrise_time)
            set_dt  = datetime.datetime.combine(today, moonset_time)
            if set_dt < rise_dt:
                set_dt += datetime.timedelta(days=1)
            total_vis = (set_dt - rise_dt).total_seconds()
        else:
            total_vis = 0

        phase_angle = set_moon_phase_angle(day, new_total_days)

        results.append({
            'day': day,
            'phase': this_phase,
            'moonrise_time': moonrise_time,
            'moonset_time':  moonset_time,
            'total_visibility': total_vis,
            'phase_angle': phase_angle
        })

    return results


def set_moon_phase_angle(day, cycle_length):
    if day < 0:
        day = 0
    if day > cycle_length:
        day = cycle_length
    y = (day + cycle_length / 2.0) % cycle_length
    return 180 * (1.0 - abs(1.0 - 2.0 * y / cycle_length))


def find_schedule_entry_for_time(schedule, cycle_start_date, sim_time):
    for entry in schedule:
        day_offset = entry['day']
        mr = entry['moonrise_time']
        ms = entry['moonset_time']
        if not mr or not ms:
            continue

        rise_dt = datetime.datetime.combine(cycle_start_date + datetime.timedelta(days=day_offset), mr)
        set_dt  = datetime.datetime.combine(cycle_start_date + datetime.timedelta(days=day_offset), ms)
        if set_dt <= rise_dt:
            set_dt += datetime.timedelta(days=1)

        if rise_dt <= sim_time < set_dt:
            return entry
    return None


def calculate_current_altitude(schedule_entry, specific_time, cycle_start_date):
    if schedule_entry['phase'] == 'New Moon':
        return -1

    mr = schedule_entry['moonrise_time']
    ms = schedule_entry['moonset_time']
    if not mr or not ms:
        return 0

    entry_date = cycle_start_date + datetime.timedelta(days=schedule_entry['day'])
    moonrise_dt = datetime.datetime.combine(entry_date, mr)
    moonset_dt  = datetime.datetime.combine(entry_date, ms)
    if moonset_dt <= moonrise_dt:
        moonset_dt += datetime.timedelta(days=1)

    if specific_time < moonrise_dt or specific_time > moonset_dt:
        return 0

    time_since_rise = (specific_time - moonrise_dt).total_seconds()
    total_vis = (moonset_dt - moonrise_dt).total_seconds()
    progress = time_since_rise / total_vis
    return 90.0 * (1.0 - np.cos(np.pi * progress))


def plot_moon_phase_angle(schedule):
    days = [entry['day'] for entry in schedule]
    angle  = [entry['phase_angle'] for entry in schedule]

    plt.figure(figsize=(8,4))
    plt.plot(days, angle, marker='o', color='blue')
    plt.title("Moon Phase Angle Over Lunar Month")
    plt.xlabel("Day in Lunar Month")
    plt.ylabel("Phase Angle")
    plt.ylim(0, 190.05)  # small padding above
    plt.grid(True, alpha=0.3)
    plt.show(block=False)


def plot_moon_schedule_times(schedule):
    def to_decimal_hour(t):
        return t.hour + t.minute / 60.0 if t else None

    days       = []
    rise_hours = []
    set_hours  = []

    for entry in schedule:
        day_label = entry['day'] + 1
        days.append(day_label)
        mr = entry['moonrise_time']
        ms = entry['moonset_time']
        rise_hours.append(to_decimal_hour(mr))
        set_hours.append(to_decimal_hour(ms))

    plt.figure(figsize=(10,5))
    plt.plot(days, rise_hours, marker='o', label='Moonrise', color='blue')
    plt.plot(days, set_hours,  marker='o', label='Moonset',  color='red')
    plt.title('Moonrise and Moonset Times')
    plt.xlabel('Day in Lunar Cycle')
    plt.ylabel('Time of Day (Hours)')
    plt.xticks(range(1, max(days)+1))
    plt.yticks(range(0, 25, 2))
    plt.ylim(0, 24)
    plt.grid(True)
    plt.legend()
    plt.show(block=False)


def plot_moon_schedule_phases(schedule):
    days = []
    phase_indices = []

    for entry in schedule:
        day_label = entry['day'] + 1
        days.append(day_label)
        phase_name = entry['phase']
        if phase_name in LUNAR_PHASES:
            p_idx = LUNAR_PHASES.index(phase_name)
        else:
            p_idx = -1
        phase_indices.append(p_idx)

    plt.figure(figsize=(10,4))
    plt.scatter(days, phase_indices, marker='o', color='green')
    plt.yticks(range(len(LUNAR_PHASES)), LUNAR_PHASES)
    plt.xlabel("Day in Lunar Cycle")
    plt.ylabel("Lunar Phase")
    plt.title("Lunar Phase by Day")
    plt.grid(True)
    plt.show(block=False)


def plot_hourly_altitude(schedule_entry, cycle_start_date, marker_interval=60):
    mr = schedule_entry['moonrise_time']
    ms = schedule_entry['moonset_time']
    if not mr or not ms:
        print("No moon visibility for this day.")
        return

    entry_date = cycle_start_date + datetime.timedelta(days=schedule_entry['day'])
    moonrise_dt = datetime.datetime.combine(entry_date, mr)
    moonset_dt  = datetime.datetime.combine(entry_date, ms)
    if moonset_dt <= moonrise_dt:
        moonset_dt += datetime.timedelta(days=1)

    time_points = []
    altitudes   = []

    current = moonrise_dt
    while current <= moonset_dt:
        alt = calculate_current_altitude(schedule_entry, current, cycle_start_date)
        time_points.append(current)
        altitudes.append(alt)
        current += datetime.timedelta(minutes=marker_interval)

    plt.figure(figsize=(10,5))
    plt.plot(time_points, altitudes, color='purple')
    plt.scatter(time_points, altitudes, color='red', s=20)
    plt.title(f"Hourly Altitude (Day {schedule_entry['day']} - {schedule_entry['phase']})")
    plt.xlabel("Time")
    plt.ylabel("Altitude (degrees)")

    hours = mdates.HourLocator()
    fmt   = mdates.DateFormatter('%H:%M')
    plt.gca().xaxis.set_major_locator(hours)
    plt.gca().xaxis.set_major_formatter(fmt)
    plt.gcf().autofmt_xdate()

    plt.ylim(0, 200)
    plt.grid(True, alpha=0.3)
    plt.show(block=False)


def prompt_int_with_skip(prompt, current_value):
    while True:
        val = input(f"{prompt} (current={current_value}): ").strip()
        if not val:  # skip
            return None
        try:
            return int(val)
        except ValueError:
            print("Invalid input. Please enter a valid integer or press Enter to keep current value.")


def prompt_float_with_skip(prompt, current_value):
    while True:
        val = input(f"{prompt} (current={current_value}): ").strip()
        if not val:  # skip
            return None
        try:
            return float(val)
        except ValueError:
            print("Invalid input. Please enter a valid float or press Enter to keep current value.")

def prompt_hex_with_skip(prompt, current_value):
    while True:
        val = input(f"{prompt} (current={current_value}): ").strip()
        if not val:  # skip
            return None
        try:
            return decimal_to_hex(int(val, 16))
        except ValueError:
            print("Invalid input. Please enter a valid hex value or press Enter to keep current value.")

def prompt_time_with_skip(prompt, current_time):
    while True:
        time_str = input(f"{prompt} (current={current_time.strftime('%H:%M')}): ").strip()
        if not time_str:  # Keep current value
            return None
            
        try:
            hours, minutes = map(int, time_str.split(':'))
            if not (0 <= hours < 24 and 0 <= minutes < 60):
                raise ValueError
            return datetime.time(hours, minutes)
        except (ValueError, IndexError):
            print("Invalid time format. Please use HH:MM (24-hour format)")

def decimal_to_hex(decimal):
    hex_value = hex(decimal)[2:].upper()
    return hex_value.zfill(6)

def user_input_thread(command_queue, state):
    while True:
        valid_commands = ["pt", "pp", "pang", "pa", "start", "change", "status", "q", ""]
        cmd = input().strip().lower()
        if cmd not in valid_commands:
            print("Unknown command. Valid commands:\n"
                  " pt, pp, pang, pa, start, change, status, q")
            continue

        if cmd == "pa":
            day_str = input("Enter a day index [0, N-1] to plot the altitude: ").strip()
            if not day_str.isdigit():
                print("Invalid day index. Command aborted.")
                continue
            command_queue.put(('pa', day_str))

        elif cmd == "change":
            print("[User Input Thread] Changing user options...")
            new_cycle_length = prompt_int_with_skip("Enter new lunar cycle length", state['user_cycle_length'])
            new_speed = prompt_float_with_skip("Enter new speed factor", state['speed_factor'])
            new_day_length = prompt_float_with_skip("Enter new real-time seconds for 24-hour sim-day", state['day_length_in_real_seconds'])
            new_hex_color = prompt_hex_with_skip("Enter new hex color", state['hex_color'])
            #FEEDER
            new_feed_drop = prompt_time_with_skip("Enter new feeder drop time", state['feed_drop'])
            new_feed_reset = prompt_time_with_skip("Enter new feeder reset time", state['feed_reset'])

            command_queue.put(('change', (new_cycle_length, new_speed, new_day_length, new_hex_color, new_feed_drop, new_feed_reset)))

        elif cmd == "status":
            command_queue.put(('status', None))

        else:
            command_queue.put((cmd, None))

        if cmd == 'q':
            break

def apply_brightness_to_hex(hex_color, brightness):

    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    r = int(r * brightness)
    g = int(g * brightness)
    b = int(b * brightness)

    new_hex = f"{r:02X}{g:02X}{b:02X}"
    return new_hex

def simulation_loop(schedule, cycle_start_date, user_cycle_length,
                   update_interval_minutes, speed_factor,
                   day_length_in_real_seconds, hex_color, feed_drop, feed_reset, stop_event,):

    real_secs_per_sim_minute = day_length_in_real_seconds / (24 * 60)
    simulation_time = cycle_start_date
    cycle_end_time  = cycle_start_date + datetime.timedelta(days=user_cycle_length)

    disp = OLED_1in27_rgb.OLED_1in27_rgb()
    #logging.info("\r1.27inch RGB OLED initializing...")
    disp.Init()
    disp.clear()

    print("\n[Simulation Thread] Started.")
    while not stop_event.is_set():
        if simulation_time >= cycle_end_time:
            print("[Simulation Thread] Reached end of the simulation.")
            break

        current_entry = find_schedule_entry_for_time(
            schedule, cycle_start_date, simulation_time
        )

        if current_entry is not None:
            altitude_deg = calculate_current_altitude(
                current_entry, simulation_time, cycle_start_date
            )
            phase_angle = current_entry.get('phase_angle', 0.0)

            print(f"[Sim {simulation_time.strftime('%Y-%m-%d %H:%M')}] "
                  f"Day {current_entry['day']} - Phase: {current_entry['phase']} "
                  f"- Altitude: {altitude_deg:.1f}° "
                  f"- Phase Angle: {phase_angle:.2f}")

            dc = set_servo_angle(altitude_deg)
            print(f"Servo DC: {dc:.2f}% ")
        else:
            # No moon entry in schedule => check day/night
            if SUNRISE_HOUR <= simulation_time.hour < SUNSET_HOUR:
                altitude_deg = 90
                print(f"[Sim {simulation_time.strftime('%Y-%m-%d %H:%M')}] "
                      f"Sun is out (alt={altitude_deg}°).")
                dc = set_servo_angle(altitude_deg)
                print(f"Servo DC: {dc:.2f}% ")
                phase_angle = 0  # no moon
            else:
                altitude_deg = 0
                print(f"[Sim {simulation_time.strftime('%Y-%m-%d %H:%M')}] "
                      "Moon not visible (alt=0).")
                dc = set_servo_angle(altitude_deg)
                print(f"Servo DC: {dc:.2f}% ")
                phase_angle = 0  # no moon

        
        '''today = simulation_time.date()
        drop_dt = datetime.datetime.combine(today, feed_drop)

        if feed_reset <= feed_drop:
            # reset is after midnight → tomorrow
            reset_dt = datetime.datetime.combine(today + datetime.timedelta(days=1),
                                         feed_reset)
        else:
            reset_dt = datetime.datetime.combine(today, feed_reset)

        # 2) trigger the drop
        if simulation_time == drop_dt:
            print("Feeder dropped.")
            # drop_feeder()

        # 3) during the feeding windowss
        if drop_dt <= simulation_time < reset_dt:
            print("Feeding!.")

    # 4) trigger the reset
        if simulation_time == reset_dt:
            print("Done Feeding!")
            # shake_feeder()
            # reset_feeder()
            print("Feeder Reset!")
        # 5) calculate the color
        '''
        # each tick in simulation_loop:

        today = simulation_time.date()
        one_minute = datetime.timedelta(seconds=60)

        drop_minus_1min_dt = datetime.datetime.combine(today, feed_drop) - one_minute
        drop_minus_1min = drop_minus_1min_dt.time()
        
        reset_minus_1min_dt = datetime.datetime.combine(today, feed_reset) - one_minute
        reset_minus_1min = reset_minus_1min_dt.time()

        # 1) did we hit the exact drop moment?
        if drop_minus_1min <= simulation_time.time() <= feed_drop:
            print("Feeder dropping...")

        if simulation_time.time() == feed_drop:
            drop_feeder()

        # 2) during the feeding window
        # – if reset is after drop (e.g. 08:00 → 12:00), window is [drop, reset)
        #– if reset ≤ drop (e.g. 21:00 → 01:00), window is [drop…24:00) ∪ [00:00…reset)
        if feed_reset > feed_drop:
            feeding = feed_drop <= simulation_time.time() < feed_reset
        else:
            feeding = (simulation_time.time() >= feed_drop) or (simulation_time.time() < feed_reset)

        if feeding:
            print("Feeding! 🥩")
        
        if reset_minus_1min <= simulation_time.time() <= feed_reset:
            #print(reset_minus_1min)
            print("Done Feeding! 🦴 Feeder Resetting... 🫨")

    # 3) did we hit the exact reset moment?
        if simulation_time.time() == feed_reset:
            
            shake_feeder()
            reset_feeder()
        



        if SUNRISE_HOUR <= simulation_time.hour < SUNSET_HOUR:
            base_hex_color = SUN_COLOR
            brightness = 1.0
        else:
            base_hex_color = hex_color
            brightness = phase_angle / 180.0

        dimmed_hex = apply_brightness_to_hex(base_hex_color, brightness)
        hex_color_h = '#' + dimmed_hex
        print(f"Screen color = {hex_color_h} (brightness={brightness:.2f})")

        # Draw the color
        from PIL import Image
        image = Image.new('RGB', (disp.width, disp.height), hex_color_h)
        disp.ShowImage(disp.getbuffer(image))

        # Sleep and increment simulation time
        real_time_to_sleep = (update_interval_minutes * real_secs_per_sim_minute) / speed_factor
        time.sleep(real_time_to_sleep)
        simulation_time += datetime.timedelta(minutes=update_interval_minutes)

    print("[Simulation Thread] Exiting...")

def handle_command(cmd, arg, stop_event, state):
    if cmd == 'pt':
        print("Plotting Moonrise/Moonset times...")
        plot_moon_schedule_times(state['moon_schedule'])

    elif cmd == 'pp':
        print("Plotting Moon schedule phases...")
        plot_moon_schedule_phases(state['moon_schedule'])

    elif cmd == 'pang':
        print("Plotting Moon Phase Angles...")
        plot_moon_phase_angle(state['moon_schedule'])

    elif cmd == 'pa':
        if arg and arg.isdigit():
            day_idx = int(arg)
            if 0 <= day_idx < len(state['moon_schedule']):
                print(f"Plotting altitude for day {day_idx}...")
                plot_hourly_altitude(state['moon_schedule'][day_idx],
                                     state['cycle_start_date'],
                                     marker_interval=30)
            else:
                print("Invalid day index!")
        else:
            print("Please enter a valid integer for the day index.")

    elif cmd == 'start':
        if not state['simulation_started']:
            print("[Main Thread] Starting simulation now...")
            state['simulation_started'] = True

            sim_thread = threading.Thread(
                target=simulation_loop,
                args=(
                    state['moon_schedule'],
                    state['cycle_start_date'],
                    state['user_cycle_length'],
                    1,  # update_interval_minutes
                    state['speed_factor'],
                    state['day_length_in_real_seconds'],
                    state['hex_color'],
                    state['feed_drop'],
                    state['feed_reset'],
                    stop_event
                ),
                daemon=True
            )
            sim_thread.start()
            state['simulation_thread'] = sim_thread
        else:
            print("[Main Thread] Simulation is already running.")

    elif cmd == 'change':
        new_cycle_length, new_speed, new_day_length, new_hex_color, new_feed_drop, new_feed_reset = arg
        # If the user typed nothing, it's None -> keep old value
        if new_cycle_length is not None:
            state['user_cycle_length'] = new_cycle_length
        if new_speed is not None:
            state['speed_factor'] = new_speed
        if new_day_length is not None:
            state['day_length_in_real_seconds'] = new_day_length
        if new_hex_color is not None:
            state['hex_color'] = new_hex_color

        ## FEEDER
        if new_feed_drop is not None:
            state['feed_drop'] = new_feed_drop
        if new_feed_reset is not None: 
            state['feed_reset'] = new_feed_reset    

        # Recompute schedule
        state['moon_schedule'] = calculate_moonrise_times(state['user_cycle_length'])
        state['cycle_start_date'] = datetime.datetime.now()


        print("[Main Thread] Options updated. You can plot again or type 'start' to run simulation.")

    elif cmd == 'status':
        print("[Main Thread] Current Simulation Parameters:")
        print(f"  Cycle Length      : {state['user_cycle_length']}")
        print(f"  Speed Factor      : {state['speed_factor']}")
        print(f"  Day Length (s)    : {state['day_length_in_real_seconds']}")
        print(f"  Hex Color         : {state['hex_color']}")
        print(f"  Start Date        : {state['cycle_start_date']}")
        print(f"  Sim Started?      : {state['simulation_started']}")
        print(f"  Feeder Drop Time  : {state['feed_drop']}")
        print(f"  Feeder Reset Time : {state['feed_reset']}")   

    elif cmd == 'q':
        print("[Main Thread] User requested quit.")
        stop_event.set()

    elif cmd == '':
        pass
    else:
        print(f"[Main Thread] Unknown command: {cmd}")


def main():
    user_cycle_length = DEFAULT_LUNAR_CYCLE_LENGTH
    speed_factor = 0.1
    day_length_in_real_seconds = 86400.0
    hex_color = 'FF0000'  # Default color (red)
    feed_drop = datetime.time(SUNSET_HOUR, 0, 0)#place holder
    feed_reset = datetime.time((SUNSET_HOUR + 2), 0, 0) # place holder

    # Build initial schedule
    moon_schedule = calculate_moonrise_times(user_cycle_length)
    cycle_start_date = datetime.datetime.now()


    # Shared state dictionary
    state = {
        'moon_schedule': moon_schedule,
        'cycle_start_date': cycle_start_date,
        'user_cycle_length': user_cycle_length,
        'speed_factor': speed_factor,
        'day_length_in_real_seconds': day_length_in_real_seconds,
        'hex_color': hex_color,
        'feed_drop': feed_drop,
        'feed_reset': feed_reset,
        'simulation_thread': None,
        'simulation_started': False,
    }

    # Create queue and threads
    command_queue = Queue()
    stop_event = threading.Event()

    # Pass state to user_input_thread so it can show current values in prompts
    input_thread = threading.Thread(
        target=user_input_thread,
        args=(command_queue, state),
        daemon=True
    )
    input_thread.start()

    print("[Main Thread] Ready. Type commands in the console:\n"
          "  pt       -> plot moonrise/moonset times\n"
          "  pp       -> plot moon schedule phases\n"
          "  pang     -> plot moon phase angles\n"
          "  pa       -> plot hourly altitude for a specific day\n"
          "  change   -> change any of the options\n"
          "  status   -> display current parameters\n"  
          "  start    -> start the simulation\n"
          "  q        -> quit the program\n")

    while not stop_event.is_set():
        while not command_queue.empty():
            cmd, arg = command_queue.get()
            handle_command(cmd, arg, stop_event, state)

        plt.pause(0.01)
        time.sleep(0.1)

    if state['simulation_thread'] is not None:
        state['simulation_thread'].join()

    print("[Main Thread] Exiting.")


if __name__ == "__main__":
    main()
