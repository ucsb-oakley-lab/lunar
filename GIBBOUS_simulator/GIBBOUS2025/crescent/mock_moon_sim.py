import datetime
import math
import threading
import time
from queue import Queue

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- Simulation Parameters ---
SUNSET_HOUR = 18
SUNRISE_HOUR = 6
DEFAULT_LUNAR_CYCLE_LENGTH = 28
SUN_COLOR = 'FFF5E1'  # soft warm white for sunlight
MOON_COLOR = 'F8F7F4'  # pale neutral moonlight (~4300K)
BRIGHTNESS_GAMMA = 1.0  # >1.0 makes full moon relatively brighter
REALTIME_HOUR_SECONDS = 60  # seconds per simulated hour when real-time mode ON (set to 3600 for true wall-clock)

LUNAR_PHASES = [
    'Full Moon', 'Waning Gibbous', 'Last Quarter', 'Waning Crescent',
    'New Moon', 'Waxing Crescent', 'First Quarter', 'Waxing Gibbous'
]

MoonPhaseLengthDays = {
    'Full Moon': 1, 'Waning Gibbous': 6, 'Last Quarter': 1,
    'Waning Crescent': 6, 'New Moon': 1, 'Waxing Crescent': 6,
    'First Quarter': 1, 'Waxing Gibbous': 6
}

MoonPhaseChecker = {
    'Full Moon': 'x', 'Waning Gibbous': 'o', 'Last Quarter': 'x',
    'Waning Crescent': 'o', 'New Moon': 'x', 'Waxing Crescent': 'o',
    'First Quarter': 'x', 'Waxing Gibbous': 'o'
}

DEFAULT_LATER_PER_DAY = (50 * 28) / 29


def get_num_phases(target_cycle_length):
    scalar = target_cycle_length / DEFAULT_LUNAR_CYCLE_LENGTH
    scaled_phases = {}
    for phase in LUNAR_PHASES:
        base_length = MoonPhaseLengthDays[phase]
        if scalar >= 1:
            new_length = math.floor(scalar * base_length) if MoonPhaseChecker[phase] == 'x' else math.ceil(scalar * base_length)
        else:
            new_length = math.ceil(scalar * base_length) if MoonPhaseChecker[phase] == 'x' else math.floor(scalar * base_length)
        scaled_phases[phase] = new_length

    new_total = sum(scaled_phases.values())
    if new_total != target_cycle_length:
        adjust = 1 if new_total < target_cycle_length else -1
        for phase in MoonPhaseChecker:
            if MoonPhaseChecker[phase] == 'o':
                scaled_phases[phase] += adjust
                new_total = sum(scaled_phases.values())
                if new_total == target_cycle_length:
                    break
    return scaled_phases, new_total


def set_moon_phase_angle(day, cycle_length):
    y = (day + cycle_length / 2.0) % cycle_length
    return 180 * (1.0 - abs(1.0 - 2.0 * y / cycle_length))


def calculate_moonrise_times(target_cycle_length):
    scaled_phases, new_total_days = get_num_phases(target_cycle_length)
    kickback = DEFAULT_LATER_PER_DAY / (target_cycle_length / 29.5)
    base_time_minutes = SUNSET_HOUR * 60
    phase_sequence = []
    for phase_name in LUNAR_PHASES:
        phase_sequence.extend([phase_name] * scaled_phases[phase_name])

    results = []
    last_new_moon_day = None
    for day in range(new_total_days):
        this_phase = phase_sequence[day]
        if this_phase == "New Moon":
            moonrise_time = None
            moonset_time = None
            last_new_moon_day = day
        else:
            if last_new_moon_day is None:
                offset_minutes = int(round(day * kickback))
                total_rise_minutes = base_time_minutes + offset_minutes
                moonrise_time = datetime.time((total_rise_minutes // 60) % 24, total_rise_minutes % 60)
                moonset_time = datetime.time(6, 0)
            else:
                moonrise_time = datetime.time(18, 0)
                days_since_new_moon = day - last_new_moon_day
                offset_after_new_moon = int(round(days_since_new_moon * kickback))
                total_set_minutes = (SUNSET_HOUR * 60) + offset_after_new_moon
                moonset_time = datetime.time((total_set_minutes // 60) % 24, total_set_minutes % 60)

        if moonrise_time and moonset_time:
            today = datetime.date.today()
            rise_dt = datetime.datetime.combine(today, moonrise_time)
            set_dt = datetime.datetime.combine(today, moonset_time)
            if set_dt < rise_dt:
                set_dt += datetime.timedelta(days=1)
            total_vis = (set_dt - rise_dt).total_seconds()
        else:
            total_vis = 0

        results.append({
            'day': day,
            'phase': this_phase,
            'moonrise_time': moonrise_time,
            'moonset_time': moonset_time,
            'total_visibility': total_vis,
            'phase_angle': set_moon_phase_angle(day, new_total_days)
        })
    return results


def apply_brightness_to_hex(hex_color, brightness):
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    r = int(r * brightness)
    g = int(g * brightness)
    b = int(b * brightness)
    return f"#{r:02X}{g:02X}{b:02X}"


def simulation_loop(schedule, cycle_start_date, user_cycle_length, speed_factor, hex_color, stop_event, real_time_mode=False):
    fig, ax = plt.subplots(figsize=(3, 3))
    square = ax.imshow(np.zeros((50, 50, 3)), vmin=0, vmax=1)
    plt.axis("off")
    plt.title("OLED Simulation (Laptop)")
    plt.show(block=False)

    sim_time = cycle_start_date
    cycle_end_time = cycle_start_date + datetime.timedelta(days=user_cycle_length)

    print(f"[Simulation] Starting from {cycle_start_date}, running for {user_cycle_length} days. "
          f"Mode: {'Real-time' if real_time_mode else f'{speed_factor}x speed'}")

    while not stop_event.is_set() and sim_time < cycle_end_time:
        entry = schedule[sim_time.day % len(schedule)]
        is_daytime = SUNRISE_HOUR <= sim_time.hour < SUNSET_HOUR

        if is_daytime:
            phase_name = "Daylight"
            brightness = 1.0
            color_hex = "#" + SUN_COLOR
        else:
            phase_name = entry['phase']
            raw_brightness = entry['phase_angle'] / 180.0 if entry['moonrise_time'] and entry['moonset_time'] else 0
            brightness = raw_brightness ** BRIGHTNESS_GAMMA
            color_hex = apply_brightness_to_hex(hex_color, brightness)

        print(f"[Sim {sim_time.strftime('%Y-%m-%d %H:%M')}] {phase_name} (brightness={brightness:.2f})")

        rgb = tuple(int(color_hex[i:i + 2], 16) / 255. for i in (1, 3, 5))
        square.set_data(np.ones((50, 50, 3)) * rgb)
        plt.pause(0.01)

        sim_time += datetime.timedelta(hours=1)
        if real_time_mode:
            time.sleep(REALTIME_HOUR_SECONDS)
        else:
            time.sleep(0.05 / speed_factor)

    print("[Simulation] Finished or interrupted.")


def simulation_loop_oled(schedule, cycle_start_date, user_cycle_length, speed_factor, hex_color, stop_event, real_time_mode=False):
    try:
        from waveshare_OLED import OLED_1in27_rgb
        from PIL import Image
    except ImportError:
        print("OLED libraries not available. Are you running this on the Pi?")
        return

    disp = OLED_1in27_rgb.OLED_1in27_rgb()
    disp.Init()
    disp.clear()

    sim_time = cycle_start_date
    cycle_end_time = cycle_start_date + datetime.timedelta(days=user_cycle_length)

    print(f"[OLED Simulation] Starting from {cycle_start_date}, running for {user_cycle_length} days. "
          f"Mode: {'Real-time' if real_time_mode else f'{speed_factor}x speed'}")

    while not stop_event.is_set() and sim_time < cycle_end_time:
        entry = schedule[sim_time.day % len(schedule)]
        is_daytime = SUNRISE_HOUR <= sim_time.hour < SUNSET_HOUR

        if is_daytime:
            phase_name = "Daylight"
            brightness = 1.0
            color_hex = "#" + SUN_COLOR
        else:
            phase_name = entry['phase']
            raw_brightness = entry['phase_angle'] / 180.0 if entry['moonrise_time'] and entry['moonset_time'] else 0
            brightness = raw_brightness ** BRIGHTNESS_GAMMA
            color_hex = apply_brightness_to_hex(hex_color, brightness)

        print(f"[OLED {sim_time.strftime('%Y-%m-%d %H:%M')}] {phase_name} (brightness={brightness:.2f})")

        try:
            image = Image.new('RGB', (disp.width, disp.height), color_hex)
            disp.ShowImage(disp.getbuffer(image))
        except Exception as e:
            print(f"Error writing to OLED: {e}")
            break

        sim_time += datetime.timedelta(hours=1)
        if real_time_mode:
            time.sleep(REALTIME_HOUR_SECONDS)
        else:
            time.sleep(0.05 / speed_factor)

    print("[OLED Simulation] Finished or interrupted.")


def plot_moon_schedule_times(schedule):
    def to_decimal_hour(t):
        return t.hour + t.minute / 60.0 if t else None

    days, rise_hours, set_hours = [], [], []
    for entry in schedule:
        days.append(entry['day'] + 1)
        rise_hours.append(to_decimal_hour(entry['moonrise_time']))
        set_hours.append(to_decimal_hour(entry['moonset_time']))

    plt.figure(figsize=(10, 5))
    plt.plot(days, rise_hours, marker='o', label='Moonrise', color='blue')
    plt.plot(days, set_hours, marker='o', label='Moonset', color='red')
    plt.title('Moonrise and Moonset Times')
    plt.xlabel('Day in Lunar Cycle')
    plt.ylabel('Time of Day (Hours)')
    plt.xticks(range(1, max(days) + 1))
    plt.yticks(range(0, 25, 2))
    plt.ylim(0, 24)
    plt.legend()
    plt.grid()
    plt.show(block=False)


def plot_moon_phase_angle(schedule):
    days = [entry['day'] for entry in schedule]
    angle = [entry['phase_angle'] for entry in schedule]
    plt.figure(figsize=(8, 4))
    plt.plot(days, angle, marker='o', color='blue')
    plt.title("Moon Phase Angle Over Lunar Month")
    plt.xlabel("Day in Lunar Month")
    plt.ylabel("Phase Angle")
    plt.ylim(0, 190)
    plt.grid(True, alpha=0.3)
    plt.show(block=False)


def plot_moon_schedule_phases(schedule):
    days, phase_indices = [], []
    for entry in schedule:
        days.append(entry['day'] + 1)
        phase_indices.append(LUNAR_PHASES.index(entry['phase']))
    plt.figure(figsize=(10, 4))
    plt.scatter(days, phase_indices, marker='o', color='green')
    plt.yticks(range(len(LUNAR_PHASES)), LUNAR_PHASES)
    plt.xlabel("Day in Lunar Cycle")
    plt.ylabel("Lunar Phase")
    plt.title("Lunar Phase by Day")
    plt.grid(True)
    plt.show(block=False)


def handle_command(cmd, arg, stop_event, state):
    if cmd == 'pt':
        plot_moon_schedule_times(state['moon_schedule'])
    elif cmd == 'pp':
        plot_moon_schedule_phases(state['moon_schedule'])
    elif cmd == 'pang':
        plot_moon_phase_angle(state['moon_schedule'])
    elif cmd == 'start':
        print("[Main Thread] Starting laptop simulation...")
        simulation_loop(
            state['moon_schedule'], state['cycle_start_date'], state['user_cycle_length'],
            state['speed_factor'], state['hex_color'], stop_event, state['real_time_mode']
        )
    elif cmd == 'startoled':
        print("[Main Thread] Starting OLED simulation...")
        simulation_loop_oled(
            state['moon_schedule'], state['cycle_start_date'], state['user_cycle_length'],
            state['speed_factor'], state['hex_color'], stop_event, state['real_time_mode']
        )
    elif cmd == 'realtime':
        state['real_time_mode'] = not state['real_time_mode']
        mode = "ON (wall-clock)" if state['real_time_mode'] else "OFF (accelerated)"
        print(f"[Main Thread] Real-time mode is now {mode}.")
    elif cmd == 'status':
        print(f"Cycle Length: {state['user_cycle_length']}, "
              f"Speed: {state['speed_factor']}, "
              f"Moon Hex: {state['hex_color']}, "
              f"Gamma: {BRIGHTNESS_GAMMA}, "
              f"Real-time: {state['real_time_mode']}")
    elif cmd == 'change':
        try:
            new_cycle_length = input(f"New cycle length (current={state['user_cycle_length']}): ").strip()
            if new_cycle_length:
                state['user_cycle_length'] = int(new_cycle_length)
                state['moon_schedule'] = calculate_moonrise_times(state['user_cycle_length'])
            new_speed = input(f"New speed factor (current={state['speed_factor']}): ").strip()
            if new_speed:
                state['speed_factor'] = float(new_speed)
            new_hex = input(f"New hex color (current={state['hex_color']}): ").strip()
            if new_hex:
                state['hex_color'] = new_hex.upper()
        except ValueError:
            print("Invalid input — values not updated.")
    elif cmd == 'q':
        stop_event.set()
    else:
        print("Unknown command. Try: pt, pp, pang, change, realtime, start, startoled, status, q")


def main():
    cycle_length = DEFAULT_LUNAR_CYCLE_LENGTH
    speed_factor = 10.0
    hex_color = MOON_COLOR
    moon_schedule = calculate_moonrise_times(cycle_length)
    cycle_start_date = datetime.datetime.now()

    state = {
        'moon_schedule': moon_schedule,
        'cycle_start_date': cycle_start_date,
        'user_cycle_length': cycle_length,
        'speed_factor': speed_factor,
        'hex_color': hex_color,
        'real_time_mode': False,
        'simulation_thread': None
    }

    stop_event = threading.Event()

    print("[Main Thread] Ready. Type commands in the console:\n"
          "  pt        -> plot moonrise/moonset times\n"
          "  pp        -> plot moon schedule phases\n"
          "  pang      -> plot moon phase angles\n"
          "  change    -> change simulation parameters\n"
          "  realtime  -> toggle real-time mode (1 hour = "
          f"{REALTIME_HOUR_SECONDS} sec)\n"
          "  start     -> start simulation (laptop window)\n"
          "  startoled -> start simulation (real OLED on Pi)\n"
          "  status    -> show current parameters\n"
          "  q         -> quit\n")
    while not stop_event.is_set():
        cmd = input("> ").strip().lower()
        handle_command(cmd, None, stop_event, state)

    if state['simulation_thread']:
        state['simulation_thread'].join()

    print("[Main Thread] Exiting.")


if __name__ == "__main__":
    main()

