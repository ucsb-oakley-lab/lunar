#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lunar Simulator using astronomical ephemerides (Skyfield).

Modes
  1) screen – visualize Moon altitude, illuminated fraction, and a brightness proxy.
  2) oled   – drive a Waveshare RGB OLED to simulate sunlight/moonlight.

Extras
  * OLED speed-up: --rate (sim seconds per real second) and --sim-start (local ISO).
  * SCREEN controls: --date YYYY-MM-DD, --hours (can exceed 24), --step-min sampling.

INSTALL (screen mode)
  python -m pip install skyfield numpy matplotlib pillow pytz

EXTRA (Raspberry Pi OLED mode)
  - Install your Waveshare OLED Python lib (e.g., `waveshare_OLED`).
  - Enable SPI/I2C as required by the panel.

Examples
  # Plot today (defaults to Bocas del Toro)
  python lunar_simulator.py screen

  # Plot a specific date for 36 hours sampled every 10 min
  python lunar_simulator.py screen --date 2025-10-03 --hours 36 --step-min 10

  # OLED, real-time updates each minute
  python lunar_simulator.py oled --interval 60

  # OLED, SPEED-UP: 1 real sec = 1 simulated hour, start at local 18:00, night only
  python lunar_simulator.py oled --rate 3600 --sim-start 2025-09-16T18:00 --interval 1 --duration 600 --night-only
"""

from __future__ import annotations

import argparse
from argparse import RawTextHelpFormatter, ArgumentDefaultsHelpFormatter
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Tuple

from PIL import Image

# Optional plotting (screen mode)
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
except Exception:
    plt = None
    mdates = None

# Timezones
try:
    import pytz
except Exception:
    raise SystemExit("Please `pip install pytz`.")

# Skyfield ephemerides
try:
    from skyfield.api import load, wgs84
    from skyfield import almanac
except Exception as e:
    raise SystemExit(f"Please `pip install skyfield`. Error: {e}")

# Optional OLED (Waveshare)
OLED_AVAILABLE = False
try:
    from waveshare_OLED import OLED_1in27_rgb  # adjust for your panel if different
    OLED_AVAILABLE = True
except Exception:
    OLED_AVAILABLE = False


# ---------------------------- Help / CLI plumbing ---------------------------- #

class HelpFormatter(ArgumentDefaultsHelpFormatter, RawTextHelpFormatter):
    """Show defaults and preserve newlines/indentation in help text."""
    pass

EPILOG_TOP = (
    "Examples:\n"
    "  # Plot today (defaults to Bocas del Toro)\n"
    "  python lunar_simulator.py screen\n\n"
    "  # Plot a specific date for 36 hours sampled every 10 min\n"
    "  python lunar_simulator.py screen --date 2025-10-03 --hours 36 --step-min 10\n\n"
    "  # OLED fast-forward demo (1s = 1h), start at local 18:00, night-only\n"
    "  python lunar_simulator.py oled --rate 3600 --sim-start 2025-09-16T18:00 --interval 1 --duration 600 --night-only\n\n"
    "Tip: for mode-specific options, run:\n"
    "  python lunar_simulator.py screen -h\n"
    "  python lunar_simulator.py oled -h\n"
)

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ephemeris-based lunar simulator (screen / oled)",
        formatter_class=HelpFormatter,
        epilog=EPILOG_TOP,
    )
    sub = p.add_subparsers(dest='mode', required=True)

    # Common options (shared by both subcommands)
    def add_common(sp):
        sp.add_argument('--lat', type=float, default=9.34,
                        help='Latitude in degrees (default=9.34 for Bocas del Toro, Panama).')
        sp.add_argument('--lon', type=float, default=-82.24,
                        help='Longitude in degrees (default=-82.24 for Bocas del Toro, Panama).')
        sp.add_argument('--tz', type=str, default='America/Panama',
                        help='IANA timezone for local timestamps.')

        sp.add_argument('--moon-color', type=str, default='F8F7F4',
                        help='Moonlight base hex color (no #); actual OLED color scales with brightness.')
        sp.add_argument('--sun-color', type=str, default='FFF5E1',
                        help='Solid color used during daytime (unless --night-only).')

        sp.add_argument('--gamma', type=float, default=1.6,
                        help='Exponent in proxy: brightness = illum_frac * sin(alt)^gamma.')
        sp.add_argument('--min-nit', type=float, default=0.0,
                        help='Brightness floor at night (0..1).')
        sp.add_argument('--max-nit', type=float, default=1.0,
                        help='Brightness ceiling (0..1).')
        sp.add_argument('--night-only', action='store_true',
                        help='Keep OLED off during daylight (screen mode still plots).')

    # screen subcommand
    sp1 = sub.add_parser(
        'screen',
        help='Plot altitude/phase/brightness for a chosen span',
        formatter_class=HelpFormatter,
        epilog="Tip: return to top-level help with: python lunar_simulator.py -h",
    )
    add_common(sp1)
    sp1.add_argument('--date', type=str, default=None,
                     help='Start date (local tz) in YYYY-MM-DD; default: today.')
    sp1.add_argument('--hours', type=float, default=24.0,
                     help='Number of hours to plot from local midnight (can exceed 24).')
    sp1.add_argument('--step-min', type=int, default=5,
                     help='Sampling step in minutes.')

    # oled subcommand
    sp2 = sub.add_parser(
        'oled',
        help='Drive a Waveshare RGB OLED to simulate moonlight/sunlight',
        formatter_class=HelpFormatter,
        epilog="Tip: return to top-level help with: python lunar_simulator.py -h",
    )
    add_common(sp2)
    sp2.add_argument('--interval', type=float, default=60.0,
                     help='Update cadence in seconds (can be fractional, e.g., 0.5).')
    sp2.add_argument('--duration', type=int, default=0,
                     help='Run time in real seconds; 0 = run until interrupted.')
    sp2.add_argument('--rate', type=float, default=0.0,
                     help='Speed-up: simulated_seconds = real_seconds × rate. '
                          '0 = real time, 3600 = 1 real sec = 1 simulated hour, 900 = 15 min/sec.')
    sp2.add_argument('--sim-start', type=str, default=None,
                     help='Local start datetime ISO (YYYY-MM-DDTHH:MM) for the simulated clock when --rate>0. '
                          'Interpreted in --tz; default: now.')

    return p


# ---------------------------- Config dataclass ---------------------------- #

@dataclass
class Config:
    # Location / time
    lat: float
    lon: float
    tz: str

    # Screen-mode sampling
    screen_date: Optional[str] = None     # YYYY-MM-DD (local)
    screen_hours: float = 24.0            # hours to plot (can exceed 24)
    screen_step_min: int = 5              # sampling step in minutes

    # OLED loop
    interval_s: float = 60.0              # update cadence in seconds (can be fractional)
    duration_s: int = 0                   # 0 = run forever (oled mode)

    # Colors / brightness model
    moon_color_hex: str = "F8F7F4"        # neutral moonlight-like
    sun_color_hex: str = "FFF5E1"         # warm sunlight-like
    night_only: bool = False
    gamma: float = 1.6
    min_nit: float = 0.0                  # floor brightness at night (0..1)
    max_nit: float = 1.0                  # cap brightness (0..1)

    # OLED simulation acceleration: simulated_seconds = real_seconds * sim_rate
    sim_rate: float = 0.0                 # 0 = real time; 3600 = 1 hr sim per 1 sec real
    sim_start_iso: Optional[str] = None   # local start, e.g., 2025-09-16T18:00


# ---------------------------- Utility helpers ---------------------------- #

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def hex_to_rgb(hex6: str) -> Tuple[int, int, int]:
    hex6 = hex6.strip().lstrip('#')
    if len(hex6) != 6:
        raise ValueError("Hex color must be 6 chars, e.g. F8F7F4")
    r = int(hex6[0:2], 16)
    g = int(hex6[2:4], 16)
    b = int(hex6[4:6], 16)
    return r, g, b

def apply_brightness(rgb: Tuple[int, int, int], brightness: float) -> Tuple[int, int, int]:
    brightness = clamp01(brightness)
    r, g, b = rgb
    return (
        int(round(r * brightness)),
        int(round(g * brightness)),
        int(round(b * brightness)),
    )

def solid_image(rgb: Tuple[int, int, int], size: Tuple[int, int]) -> Image.Image:
    return Image.new('RGB', size, rgb)


# --------------------- Ephemeris and brightness model --------------------- #

class Ephemeris:
    def __init__(self, lat: float, lon: float, tzname: str):
        self.ts  = load.timescale()
        # Skyfield will download to cache once (e.g., ~/.skyfield) then load locally
        self.eph = load('de421.bsp')
        self.earth = self.eph['earth']
        self.sun   = self.eph['sun']
        self.moon  = self.eph['moon']
        self.topo  = wgs84.latlon(latitude_degrees=lat, longitude_degrees=lon)
        self.tz    = pytz.timezone(tzname)

    def now_ts(self) -> datetime:
        return datetime.now(tz=self.tz)

    def alt_phase_fraction(self, dt_local: datetime) -> Tuple[float, float, float]:
        """Return (altitude_deg, illuminated_fraction_0_1, elongation_deg)."""
        t = self.ts.from_datetime(dt_local.astimezone(pytz.UTC))
        observer = self.earth + self.topo

        app_moon = observer.at(t).observe(self.moon).apparent()
        alt, az, distance = app_moon.altaz()
        alt_deg = alt.degrees

        frac = almanac.fraction_illuminated(self.eph, 'moon', t)

        app_sun = observer.at(t).observe(self.sun).apparent()
        elong = app_moon.separation_from(app_sun)

        return alt_deg, float(frac), elong.degrees

    def is_daytime(self, dt_local: datetime) -> bool:
        t = self.ts.from_datetime(dt_local.astimezone(pytz.UTC))
        observer = self.earth + self.topo
        app_sun = observer.at(t).observe(self.sun).apparent()
        alt, _, _ = app_sun.altaz()
        return alt.degrees > 0.0


def moon_brightness_proxy(alt_deg: float, illum_frac: float, gamma: float,
                          min_nit: float, max_nit: float) -> float:
    """Simple proxy: brightness = illum_frac * sin(alt)^gamma, clamped to [min_nit, max_nit]."""
    alt_rad = math.radians(max(0.0, alt_deg))
    alt_term = math.sin(alt_rad) ** max(0.0, gamma)
    raw = illum_frac * alt_term
    return clamp01(min(max(raw, min_nit), max_nit))


# --------------------------- Screen mode logic --------------------------- #

def run_screen(cfg: Config) -> None:
    if plt is None:
        raise SystemExit("Matplotlib not available. `pip install matplotlib`.")

    eph = Ephemeris(cfg.lat, cfg.lon, cfg.tz)

    # Determine starting local midnight for the selected day
    if cfg.screen_date:
        try:
            base = datetime.strptime(cfg.screen_date, "%Y-%m-%d")
        except ValueError:
            raise SystemExit("Invalid --date. Use YYYY-MM-DD.")
        local_midnight = eph.tz.localize(base.replace(hour=0, minute=0, second=0, microsecond=0))
    else:
        now = eph.now_ts()
        local_midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)

    total_minutes = int(round(cfg.screen_hours * 60.0))
    step_min = max(1, int(cfg.screen_step_min))
    times = [local_midnight + timedelta(minutes=m) for m in range(0, total_minutes + 1, step_min)]

    altitudes, illum, bright = [], [], []
    for dt in times:
        alt, frac, _ = eph.alt_phase_fraction(dt)
        altitudes.append(alt)
        illum.append(frac)
        b = moon_brightness_proxy(alt, frac, cfg.gamma, cfg.min_nit, cfg.max_nit)
        if eph.is_daytime(dt) and cfg.night_only:
            b = 0.0
        bright.append(b)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(times, altitudes, label='Moon Altitude (deg)')
    ax1.set_ylabel('Altitude (deg)')
    ax1.set_xlabel('Local time')

    ax2 = ax1.twinx()
    ax2.plot(times, illum, label='Illuminated Fraction', linestyle='--')
    ax2.plot(times, bright, label='Brightness Proxy', linestyle=':')
    ax2.set_ylabel('Fraction / Brightness (0..1)')

    # Time axis formatting that works for multi-day spans
    if mdates:
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax1.xaxis.set_major_formatter(mdates.AutoDateFormatter(ax1.xaxis.get_major_locator()))
        fig.autofmt_xdate()

    lines, labels = [], []
    for ax in (ax1, ax2):
        L = ax.get_legend_handles_labels()
        lines += L[0]
        labels += L[1]
    ax1.legend(lines, labels, loc='upper left')

    plt.title(f"Moon altitude, phase, brightness — {cfg.tz}  "
              f"(start={times[0].strftime('%Y-%m-%d %H:%M')}, hours={cfg.screen_hours:g})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------- OLED mode logic ---------------------------- #

def run_oled(cfg: Config) -> None:
    eph = Ephemeris(cfg.lat, cfg.lon, cfg.tz)

    disp = None
    size = (128, 128)
    if OLED_AVAILABLE:
        try:
            disp = OLED_1in27_rgb.OLED_1in27_rgb()
            disp.Init()
            disp.clear()
            size = (disp.width, disp.height)
        except Exception as e:
            print(f"[WARN] OLED init failed ({e}). Falling back to console-only mode.")
            disp = None
    else:
        print("[INFO] Waveshare OLED library not found. Running headless.")

    moon_rgb = hex_to_rgb(cfg.moon_color_hex)
    sun_rgb = hex_to_rgb(cfg.sun_color_hex)

    # Speed-up mapping for a simulated clock
    if cfg.sim_start_iso:
        try:
            start_local = datetime.fromisoformat(cfg.sim_start_iso)
            if start_local.tzinfo is None:
                # interpret as local time in target tz
                start_local = eph.tz.localize(start_local)
            else:
                start_local = start_local.astimezone(eph.tz)
        except Exception as e:
            print(f"[WARN] Could not parse --sim-start '{cfg.sim_start_iso}': {e}. Using current time.")
            start_local = eph.now_ts()
    else:
        start_local = eph.now_ts()

    t0 = time.time()
    while True:
        if cfg.sim_rate and cfg.sim_rate > 0.0:
            elapsed = time.time() - t0
            sim_now = start_local + timedelta(seconds=elapsed * cfg.sim_rate)
        else:
            sim_now = eph.now_ts()

        alt, frac, phase_angle = eph.alt_phase_fraction(sim_now)
        daytime = eph.is_daytime(sim_now)

        if daytime and cfg.night_only:
            rgb = (0, 0, 0)
            brightness = 0.0
            role = 'day (off)'
        elif daytime:
            rgb = sun_rgb
            brightness = 1.0
            role = 'sunlight'
        else:
            brightness = moon_brightness_proxy(alt, frac, cfg.gamma, cfg.min_nit, cfg.max_nit)
            rgb = apply_brightness(moon_rgb, brightness)
            role = 'moonlight'

        print(f"[{sim_now.strftime('%Y-%m-%d %H:%M:%S %Z')}] role={role} "
              f"alt={alt:5.1f}° illum={frac:0.3f} phase≈{phase_angle:5.1f}° "
              f"-> brightness={brightness:0.3f} RGB={rgb} (rate={cfg.sim_rate}x)")

        if disp is not None:
            img = solid_image(rgb, size)
            try:
                disp.ShowImage(disp.getbuffer(img))
            except Exception as e:
                print(f"[WARN] OLED draw failed: {e}")

        if cfg.duration_s > 0 and (time.time() - t0) >= cfg.duration_s:
            break
        time.sleep(max(0.05, float(cfg.interval_s)))  # allow fractional intervals

    print("Done.")


# --------------------------------- main ---------------------------------- #

def main(argv=None):
    import sys
    parser = build_parser()
    argv = sys.argv[1:] if argv is None else argv

    # If no args, show top-level help (includes Tip in epilog)
    if not argv:
        parser.print_help()
        return

    args = parser.parse_args(argv)

    cfg = Config(
        lat=args.lat,
        lon=args.lon,
        tz=args.tz,

        screen_date=getattr(args, 'date', None),
        screen_hours=getattr(args, 'hours', 24.0),
        screen_step_min=getattr(args, 'step_min', 5),

        interval_s=getattr(args, 'interval', 60.0),
        duration_s=getattr(args, 'duration', 0),
        moon_color_hex=args.moon_color,
        sun_color_hex=args.sun_color,
        night_only=args.night_only,
        gamma=args.gamma,
        min_nit=args.min_nit,
        max_nit=args.max_nit,
        sim_rate=getattr(args, 'rate', 0.0),
        sim_start_iso=getattr(args, 'sim_start', None),
    )

    if args.mode == 'screen':
        run_screen(cfg)
    elif args.mode == 'oled':
        run_oled(cfg)
    else:
        raise SystemExit('Unknown mode')

if __name__ == '__main__':
    main()

