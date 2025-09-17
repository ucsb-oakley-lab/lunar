#!/usr/bin/env python3
"""
Lunar Simulator using astronomical ephemerides (Skyfield).

Modes:
  1) screen  – visualize real moon altitude/phase and a brightness proxy over a day.
  2) oled    – drive a Waveshare RGB OLED to simulate sunlight/moonlight.

NEW: speed-up simulation for OLED mode via --rate and optional --sim-start.

INSTALL (screen mode):
  python -m pip install skyfield numpy matplotlib pillow pytz

EXTRA (Raspberry Pi OLED mode):
  - Install your Waveshare OLED Python lib (e.g., `waveshare_OLED`).
  - Enable SPI/I2C as required by the panel.

USAGE
  # Screen visualization for today (local tz)
  python lunar_ephemeris_simulator_v2.py screen --lat 34.414 --lon -119.848 --tz America/Los_Angeles

  # OLED, real-time updates each minute
  python lunar_ephemeris_simulator_v2.py oled --lat 34.414 --lon -119.848 --tz America/Los_Angeles --interval 60

  # OLED, SPEED-UP: 1 real sec = 1 simulated hour, start at local 18:00
  python lunar_ephemeris_simulator_v2.py oled --lat 34.414 --lon -119.848 --tz America/Los_Angeles \
      --rate 3600 --sim-start 2025-09-16T18:00 --interval 1 --duration 720

NOTES
  * Brightness model: brightness = phase_fraction * max(sin(alt), 0)^gamma
    (photometric proxy, not calibrated lux). Tweak --gamma, --min-nit, --max-nit.
  * Daytime: fills with SUN_COLOR unless --night-only.
  * Skyfield computes true topocentric alt/az and illuminated fraction.
"""

from __future__ import annotations
import argparse
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Tuple

import numpy as np
from PIL import Image

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
except Exception:
    plt = None
    mdates = None

try:
    import pytz
except Exception as e:
    raise SystemExit("Please `pip install pytz`.")

try:
    from skyfield.api import load, wgs84
    from skyfield import almanac
except Exception as e:
    raise SystemExit("Please `pip install skyfield`. Error: %s" % e)

OLED_AVAILABLE = False
try:
    from waveshare_OLED import OLED_1in27_rgb  # adjust to your panel
    OLED_AVAILABLE = True
except Exception:
    OLED_AVAILABLE = False

# ---------------------------- Config dataclass ---------------------------- #

@dataclass
class Config:
    lat: float
    lon: float
    tz: str
    interval_s: int = 60  # update cadence in seconds (oled loop)
    duration_s: int = 0   # 0 = run forever (oled mode)
    moon_color_hex: str = "F8F7F4"  # neutral moonlight-like
    sun_color_hex: str = "FFF5E1"   # warm sunlight-like
    night_only: bool = False
    gamma: float = 1.6
    min_nit: float = 0.0  # floor brightness at night (0..1)
    max_nit: float = 1.0  # cap brightness (0..1)
    # Simulation acceleration: simulated_seconds = real_seconds * sim_rate
    sim_rate: float = 0.0               # 0 = real time; 3600 = 1 hr sim per 1 sec real
    sim_start_iso: Optional[str] = None # local start, e.g., 2025-09-16T18:00

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
        self.eph = load('de421.bsp')
        self.earth = self.eph['earth']
        self.sun   = self.eph['sun']
        self.moon  = self.eph['moon']
        self.topo  = wgs84.latlon(latitude_degrees=lat, longitude_degrees=lon)
        self.tz    = pytz.timezone(tzname)
    def now_ts(self) -> datetime:
        return datetime.now(tz=self.tz)

    def alt_phase_fraction(self, dt_local: datetime) -> Tuple[float, float, float]:
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
        sun = self.eph['sun']
        app = self.observer.at(t).observe(sun).apparent()
        alt, _, _ = app.altaz()
        return alt.degrees > 0.0
    def is_daytime(self, dt_local: datetime) -> bool:
        t = self.ts.from_datetime(dt_local.astimezone(pytz.UTC))
        observer = self.earth + self.topo
        app_sun = observer.at(t).observe(self.sun).apparent()
        alt, _, _ = app_sun.altaz()
        return alt.degrees > 0.0

def moon_brightness_proxy(alt_deg: float, illum_frac: float, gamma: float, min_nit: float, max_nit: float) -> float:
    alt_rad = math.radians(max(0.0, alt_deg))
    alt_term = math.sin(alt_rad) ** max(0.0, gamma)
    raw = illum_frac * alt_term
    return clamp01(min(max(raw, min_nit), max_nit))

# --------------------------- Screen mode logic --------------------------- #

def run_screen(cfg: Config) -> None:
    if plt is None:
        raise SystemExit("Matplotlib not available. `pip install matplotlib`.")
    eph = Ephemeris(cfg.lat, cfg.lon, cfg.tz)
    now = eph.now_ts()
    local_midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    times = [local_midnight + timedelta(minutes=m) for m in range(0, 24*60, 5)]
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
    if mdates:
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        fig.autofmt_xdate()
    lines, labels = [], []
    for ax in (ax1, ax2):
        L = ax.get_legend_handles_labels()
        lines += L[0]
        labels += L[1]
    ax1.legend(lines, labels, loc='upper left')
    plt.title(f"Moon altitude, phase, brightness — {cfg.tz}")
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

    # Speed-up mapping
    if cfg.sim_start_iso:
        try:
            start_local = datetime.fromisoformat(cfg.sim_start_iso)
            if start_local.tzinfo is None:
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

        print(f"[{sim_now.strftime('%Y-%m-%d %H:%M:%S %Z')}] role={role} alt={alt:5.1f}° illum={frac:0.3f} "
              f"phase≈{phase_angle:5.1f}° -> brightness={brightness:0.3f} RGB={rgb} (rate={cfg.sim_rate}x)")

        if disp is not None:
            img = solid_image(rgb, size)
            try:
                disp.ShowImage(disp.getbuffer(img))
            except Exception as e:
                print(f"[WARN] OLED draw failed: {e}")

        if cfg.duration_s > 0 and (time.time() - t0) >= cfg.duration_s:
            break
        time.sleep(max(0.1, cfg.interval_s))

    print("Done.")

# ------------------------------- CLI entry ------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ephemeris-based lunar simulator (screen / oled)")
    sub = p.add_subparsers(dest='mode', required=True)

    def add_common(sp):
        sp.add_argument('--lat', type=float, default=9.34, help='Latitude in degrees (default=9.34 for Bocas del Toro, Panama)')
        sp.add_argument('--lon', type=float, default=-82.24,help='Longitude in degrees (default=-82.24 for Bocas del Toro, Panama)')
        sp.add_argument('--tz', type=str, default='America/Panama', help='Timezone (default=America/Panama)')
        sp.add_argument('--moon-color', type=str, default='F8F7F4', help='Moonlight base hex color (no #)')
        sp.add_argument('--sun-color', type=str, default='FFF5E1', help='Sunlight base hex color (no #)')
        sp.add_argument('--gamma', type=float, default=1.6, help='Altitude gamma; higher makes zenith much brighter')
        sp.add_argument('--min-nit', type=float, default=0.0, help='Floor brightness at night (0..1)')
        sp.add_argument('--max-nit', type=float, default=1.0, help='Ceiling brightness (0..1)')
        sp.add_argument('--night-only', action='store_true', help='Turn off display during day')

    sp1 = sub.add_parser('screen', help='Plot altitude/phase/brightness for the current day')
    add_common(sp1)

    sp2 = sub.add_parser('oled', help='Drive a Waveshare RGB OLED to simulate moonlight')
    add_common(sp2)
    sp2.add_argument('--interval', type=int, default=60, help='Update interval (seconds)')
    sp2.add_argument('--duration', type=int, default=0, help='Run time (seconds); 0 = forever')
    sp2.add_argument('--rate', type=float, default=0.0, help='Sim speed multiplier: simulated_seconds = real_seconds * rate; 0 = real time')
    sp2.add_argument('--sim-start', type=str, default=None, help='Local start datetime ISO, e.g., 2025-09-16T18:00')

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = Config(
        lat=args.lat,
        lon=args.lon,
        tz=args.tz,
        interval_s=getattr(args, 'interval', 60),
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

