"""
Simulasi Orbit Satelit - Autopilot Hohmann + Phasing + Launch from Ground
Fitur: UI lengkap, grafik energi, thrust manual, autopilot otomatis,
       waktu simulasi, indikator arah burn, toggle Rendezvous/Flyby,
       Lock Zoom, Kembali ke Bumi aman (target 4000 km) dengan koreksi agresif,
       dan status kepulangan flyby. Pendaratan manual setelah orbit tercapai.

================================================================================
  MANUAL SHORTCUT KEYBOARD
================================================================================

  ── 1. KONTROL SIMULASI UTAMA ───────────────────────────────────────────────
  Space       Toggle Jalankan / Pause
  R           Reset simulasi
  M           Tembak Manuver
  G           Toggle Ghost On/Off
  O           Toggle Garis Orbit
  L           Toggle Lock Zoom

  ── 2. THRUSTER MANUAL ───────────────────────────────────────────────────────
  Z (tahan)   BURN aktif (tekan=mulai, lepas=berhenti)
  , (koma)    Arah burn -5 derajat
  . (titik)   Arah burn +5 derajat
  [ (kurung)  Kurangi delta-v rate (-10 m/s)
  ] (kurung)  Tambah delta-v rate (+10 m/s)
  1           Preset Prograde   (0 derajat)
  2           Preset Normal     (90 derajat)
  3           Preset Retrograde (180 derajat)
  4           Preset Radial-in  (270 derajat)

  ── 3. ZOOM & PAN CANVAS ────────────────────────────────────────────────────
  + / =       Zoom in
  -           Zoom out
  Panah atas/bawah/kiri/kanan  Pan canvas (geser tampilan)
  BackSpace   Reset zoom (sama seperti double-click)

  ── 4. SLIDER PARAMETER ──────────────────────────────────────────────────────
  Q / W       Altitude awal  -/+
  A / S       Eksentrisitas awal  -/+
  Y / U       Altitude manuver  -/+
  H / J       Eksentrisitas manuver  -/+
  D (tahan)   dt simulasi -1 (berulang selama ditahan)
  F (tahan)   dt simulasi +1 (berulang selama ditahan)
  Ctrl+,      Kecepatan sim -1
  Ctrl+.      Kecepatan sim +1

  ── 5. AUTOPILOT & NAVIGASI ─────────────────────────────────────────────────
  N           Autopilot ke Bulan
  P           Autopilot ke Mars
  X           Batal Autopilot
  B           Toggle Mode Rendezvous/Flyby
  K           Kembali ke Bumi

  ── 6. LAUNCH ────────────────────────────────────────────────────────────────
  V           Launch to 325 km (hanya aktif saat satelit di permukaan & sim berhenti)
  C           Batal Launch

  ── 7. ZOOM CEPAT, GRAFIK & RIWAYAT ─────────────────────────────────────────
  5           Toggle Fokus ke Satelit (sama seperti klik satelit di kanvas)
  6           Zoom ke Bulan
  7           Zoom ke Mars
  8           Buka Grafik Energi (jendela live KE/PE/TE)
  9           Hapus Riwayat orbit
  0           Toggle Gravitasi Bulan ON/OFF

================================================================================
"""

import tkinter as tk
import math
import numpy as np
from scipy.integrate import solve_ivp
from astropy.time import Time
from astropy.coordinates import get_body, solar_system_ephemeris
import astropy.units as u
solar_system_ephemeris.set('builtin')

# ======================================
# KONSTANTA FISIKA
# ======================================
G          = 6.67e-11
M          = 5.97e24       # massa Bumi (kg) — posisi tetap di origin
R_BUMI     = 6.371e6
OMEGA_BUMI = 7.2921e-5
V_ROT      = OMEGA_BUMI * R_BUMI
ALT_TARGET_RETURN = 4000  # km (orbit kembali, aman)

# ── Bulan ──────────────────────────────────────────────────────────
M_BULAN    = 7.342e22      # massa Bulan (kg)
R_BULAN    = 1.7374e6      # radius Bulan (m)

# ── Satelit ────────────────────────────────────────────────────────
M_SATELIT  = 1000.0        # massa satelit (kg)  — untuk info, tidak mempengaruhi lintasan

def v_circular(r):
    return math.sqrt(G * M / r)

# ======================================
# WARNA
# ======================================
BG       = "#0d0d1a"
OCEAN    = "#1a3a5c"
COAST    = "#4a9edd"
LAND     = "#2d6e3a"
GRAY     = "#334466"
TEXT_COL = "#ccddff"
WHITE    = "#ffffff"
CYAN     = "#00cccc"
ORANGE   = "#ff8800"
LIME     = "#44ff44"
RED      = "#ff3333"
YELLOW   = "#ffff00"
PURPLE   = "#cc44ff"
PINK     = "#ff44aa"
AP_COLOR = "#ffaa44"
RETURN_COLOR = "#66ddff"
LAUNCH_COLOR = "#88ff88"

ORBIT_COLORS = [ORANGE, CYAN, LIME, PURPLE, PINK, YELLOW, "#ff6644", "#44ffcc"]

# ======================================
# FISIKA (RK4, orbital_elements, dll)
# ======================================

# ── Gravitasi Bulan (diakses oleh fungsi ODE) ──────────────────────
# Diupdate setiap langkah dari _sim_step agar posisi bulan konsisten
_moon_grav_x  = 0.0   # posisi Bulan saat ini (m) – komponen x
_moon_grav_y  = 0.0   # posisi Bulan saat ini (m) – komponen y
_moon_grav_on = False  # True = gravitasi Bulan aktif


def rk4_step(x, y, vx, vy, dt):
    state = np.array([x, y, vx, vy], dtype=np.float64)
    def deriv_np(s):
        r3 = (s[0]*s[0] + s[1]*s[1]) ** 1.5
        ax = -G*M*s[0]/r3
        ay = -G*M*s[1]/r3
        if _moon_grav_on:
            dx = s[0] - _moon_grav_x
            dy = s[1] - _moon_grav_y
            rm3 = (dx*dx + dy*dy) ** 1.5
            if rm3 > 1e6:
                ax += -G*M_BULAN*dx/rm3
                ay += -G*M_BULAN*dy/rm3
        return np.array([s[2], s[3], ax, ay])
    k1 = deriv_np(state)
    k2 = deriv_np(state + 0.5*dt*k1)
    k3 = deriv_np(state + 0.5*dt*k2)
    k4 = deriv_np(state +     dt*k3)
    result = state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return float(result[0]), float(result[1]), float(result[2]), float(result[3])

_MAX_DT = 5.0

def _ode_gravity(t, state):
    x, y, vx, vy = state
    r3 = (x*x + y*y) ** 1.5
    ax = -G*M*x/r3
    ay = -G*M*y/r3
    if _moon_grav_on:
        dx = x - _moon_grav_x
        dy = y - _moon_grav_y
        rm3 = (dx*dx + dy*dy) ** 1.5
        if rm3 > 1e6:
            ax += -G*M_BULAN*dx/rm3
            ay += -G*M_BULAN*dy/rm3
    return [vx, vy, ax, ay]

def rk4_step_safe(x, y, vx, vy, dt):
    try:
        sol = solve_ivp(
            _ode_gravity,
            [0.0, dt],
            [x, y, vx, vy],
            method='DOP853',
            rtol=1e-9,
            atol=1e-6,
            dense_output=False,
        )
        if sol.success:
            s = sol.y[:, -1]
            return float(s[0]), float(s[1]), float(s[2]), float(s[3])
    except Exception:
        pass
    n = max(1, math.ceil(abs(dt) / _MAX_DT))
    sub_dt = dt / n
    for _ in range(n):
        x, y, vx, vy = rk4_step(x, y, vx, vy, sub_dt)
    return x, y, vx, vy

def orbital_elements(x, y, vx, vy):
    pos = np.array([x, y], dtype=np.float64)
    vel = np.array([vx, vy], dtype=np.float64)
    r   = float(np.linalg.norm(pos))
    v   = float(np.linalg.norm(vel))
    E   = 0.5*v*v - G*M/r
    a   = -G*M/(2*E) if E != 0 else float('inf')
    hz  = float(x*vy - y*vx)
    ex  = (vy*hz - G*M*x/r)/(G*M)
    ey  = (-vx*hz - G*M*y/r)/(G*M)
    e   = float(np.hypot(ex, ey))
    T   = 2*math.pi*math.sqrt(abs(a)**3/(G*M)) if a > 0 else float('inf')
    alt = (r - R_BUMI)/1000
    return {'a':a, 'e':e, 'E':E, 'T':T, 'alt':alt, 'r':r, 'v':v, 'h':hz}

def periapsis_apoapsis(el):
    a = el['a']; e = el['e']
    if a <= 0 or e >= 1.0:
        r = el['r']
        return r, (r-R_BUMI)/1000, None, None
    r_p = a * (1 - e)
    r_a = a * (1 + e)
    return r_p, (r_p-R_BUMI)/1000, r_a, (r_a-R_BUMI)/1000

def compute_orbit_velocity(alt_km, ecc, direction=1):
    r_p = R_BUMI + alt_km * 1000.0
    r_p = max(r_p, R_BUMI)
    if ecc < 1.0:
        a = r_p / (1.0 - ecc)
        v = math.sqrt(G*M * (2.0/r_p - 1.0/a))
    else:
        a = r_p / (ecc - 1.0)
        v = math.sqrt(G*M * (2.0/r_p + 1.0/a))
    return r_p, 0.0, 0.0, direction * v

def predict_orbit_full(x, y, vx, vy, n_pts=120):
    r  = math.sqrt(x*x + y*y)
    v2 = vx*vx + vy*vy
    E  = 0.5*v2 - G*M/r
    if E < 0:
        a  = -G*M / (2*E)
        hz = x*vy - y*vx
        ex = (vy*hz - G*M*x/r)/(G*M)
        ey = (-vx*hz - G*M*y/r)/(G*M)
        emag = math.sqrt(ex*ex + ey*ey)
        emag = max(emag, 1e-12)
        if emag >= 1.0:
            pts_x = [x]; pts_y = [y]
            cx2, cy2, cvx2, cvy2 = x, y, vx, vy
            for _ in range(250):
                cx2, cy2, cvx2, cvy2 = rk4_step_safe(cx2, cy2, cvx2, cvy2, 15)
                r2 = math.sqrt(cx2*cx2 + cy2*cy2)
                if r2 < R_BUMI * 0.99 or r2 > R_BUMI * 50:
                    break
                pts_x.append(cx2); pts_y.append(cy2)
            return pts_x, pts_y
        if emag < 1e-6:
            enx = x / r; eny = y / r
        else:
            enx = ex/emag; eny = ey/emag
        nu  = np.linspace(0, 2*math.pi, n_pts + 1)
        denom = 1 + emag*np.cos(nu)
        valid = denom > 1e-6
        nu = nu[valid]; denom = denom[valid]
        rc  = a*(1 - emag*emag) / denom
        pts_x = (rc*(enx*np.cos(nu) - eny*np.sin(nu))).tolist()
        pts_y = (rc*(eny*np.cos(nu) + enx*np.sin(nu))).tolist()
        return pts_x, pts_y
    else:
        pts_x = [x]; pts_y = [y]
        cx, cy, cvx, cvy = x, y, vx, vy
        for _ in range(250):
            cx, cy, cvx, cvy = rk4_step_safe(cx, cy, cvx, cvy, 15)
            r2 = math.sqrt(cx*cx + cy*cy)
            if r2 < R_BUMI * 0.99 or r2 > R_BUMI * 50:
                break
            pts_x.append(cx); pts_y.append(cy)
        return pts_x, pts_y

def predict_orbit(x, y, vx, vy, dt=10, max_pts=800):
    pts_x = [x]; pts_y = [y]
    cx, cy, cvx, cvy = x, y, vx, vy
    crossed = False
    for _ in range(max_pts):
        cx, cy, cvx, cvy = rk4_step_safe(cx, cy, cvx, cvy, dt)
        r = math.sqrt(cx*cx + cy*cy)
        if r < R_BUMI * 0.99:
            break
        pts_x.append(cx); pts_y.append(cy)
        if len(pts_x) > 20:
            prev_y = pts_y[-2]
            if not crossed and prev_y < 0 and cy >= 0:
                crossed = True
            elif crossed and prev_y < 0 and cy >= 0:
                break
    return pts_x, pts_y

# ======================================
# CANVAS HELPERS
# ======================================
def w2c(x, y, cx, cy, scale):
    return cx + x*scale, cy - y*scale

def draw_circle_pts(canvas, cx, cy, scale, r, color, width=1, fill="", dash=None):
    angles = np.linspace(0, 2*math.pi, 181)
    xs = r * np.cos(angles)
    ys = r * np.sin(angles)
    px = cx + xs * scale
    py = cy - ys * scale
    pts = np.column_stack([px, py]).flatten().tolist()
    if dash:
        canvas.create_line(pts, fill=color, width=width, dash=dash, smooth=False)
    else:
        canvas.create_polygon(pts, outline=color, fill=fill, width=width, smooth=False)

def draw_arrow_c(canvas, x0, y0, x1, y1, color, width=2):
    canvas.create_line(x0, y0, x1, y1, fill=color, width=width,
                       arrow=tk.LAST, arrowshape=(10,12,4))

def draw_path_pts(canvas, xs, ys, cx, cy, scale, color, width=1, dash=None, smooth=False):
    if len(xs) < 2:
        return
    arr_x = np.asarray(xs, dtype=np.float64)
    arr_y = np.asarray(ys, dtype=np.float64)
    px = cx + arr_x * scale
    py = cy - arr_y * scale
    pts = np.column_stack([px, py]).flatten().tolist()
    kw = dict(fill=color, width=width, smooth=smooth)
    if dash:
        kw['dash'] = dash
    canvas.create_line(pts, **kw)

# ======================================
# ORBIT RECORD
# ======================================
class OrbitRecord:
    def __init__(self, color, label):
        self.color   = color
        self.label   = label
        self.trail_x = []
        self.trail_y = []
        self.line_id = None

    def add(self, x, y):
        self.trail_x.append(x)
        self.trail_y.append(y)
        if len(self.trail_x) > 1500:
            self.trail_x = self.trail_x[-1200:]
            self.trail_y = self.trail_y[-1200:]

# ======================================
# KELAS UTAMA: OrbitApp
# ======================================
class OrbitApp:
    def __init__(self, root):
        self.root = root
        root.title("Simulasi Orbit Satelit - Autopilot + Kembali ke Bumi (aman)")
        root.configure(bg=BG)
        root.geometry("1400x860")

        self.init_alt_var = tk.DoubleVar(value=0)
        self.init_ecc_var = tk.DoubleVar(value=0.0)
        self.man_alt_var  = tk.DoubleVar(value=800)
        self.man_ecc_var  = tk.DoubleVar(value=0.3)
        self.dt_var       = tk.DoubleVar(value=5)
        self.speed_var    = tk.DoubleVar(value=3)
        self._dt_label_var = tk.StringVar(value="5s")

        self.sat_x = self.sat_y = self.sat_vx = self.sat_vy = 0.0
        self._apply_initial_orbit()

        self.orbits    = []
        self.orbit_idx = 0
        self._new_orbit_segment()

        self.pending_maneuver = None
        self.maneuver_flash   = 0
        self.ghost_x  = []; self.ghost_y  = []
        self.cur_ox   = []; self.cur_oy   = []
        self.show_ghost = True
        self.show_orbit_line = True
        self.earth_angle = 0.0

        self.MOON_R_VIS  = R_BUMI * 0.27
        self.MARS_R_VIS  = R_BUMI * 0.53
        self.MOON_PERIOD = 27.3 * 86400
        self.MARS_PERIOD = 779.9 * 86400

        self.sim_epoch = Time.now()
        moon_x, moon_y, moon_r = self._astropy_body_xy('moon', 0.0)
        mars_x, mars_y, mars_r = self._astropy_body_xy('mars', 0.0)

        self.MOON_DIST  = moon_r
        self.MARS_DIST  = mars_r
        self.moon_angle = math.atan2(moon_y, moon_x)
        self.mars_angle = math.atan2(mars_y, mars_x)

        self._astropy_cache_time = -1e9
        self._astropy_cache_interval = 3600.0
        self._moon_x = moon_x; self._moon_y = moon_y
        self._mars_x = mars_x; self._mars_y = mars_y

        self.zoom_level  = 1.0
        self.pan_cx_off  = 0.0
        self.pan_cy_off  = 0.0
        self._pan_start  = None
        self.follow_satellite = False
        self._sat_px = self._sat_py = None
        self.running  = False
        self.after_id = None
        self.sim_dt   = 5
        self.steps_per_frame = 3
        self._was_above_surface = False
        self._current_dt_sub = 0.0

        self.energy_history = []
        self.sim_time = 0.0
        self.maneuver_times = []
        self._energy_rv_history = []

        self.perm_trail_x = []
        self.perm_trail_y = []
        self.perm_line_id = None

        self.thrust_dv_rate   = tk.DoubleVar(value=50.0)
        self.thrust_angle_deg = tk.DoubleVar(value=0.0)
        self._thrust_active   = False
        self._thrust_after    = None
        self.total_dv_applied = 0.0
        self._locked_thrust_bx = None
        self._locked_thrust_by = None

        self._dt_repeat_dir   = 0      # -1, 0, +1 : arah tahan tombol D/F (dt sim)
        self._dt_repeat_after = None   # id job .after() untuk repeat dt

        self.autopilot_active   = False
        self.autopilot_target   = None
        self.autopilot_phase    = None
        self.autopilot_r_target = 0.0
        self.autopilot_burn2_done = False
        self.autopilot_intercept_angle = 0.0
        self.autopilot_transfer_time = 0.0
        self.autopilot_a_transfer = 0.0
        self.autopilot_delta_theta_req = 0.0
        self.autopilot_wait_start_time = 0.0
        self.autopilot_burn1_applied = False
        self._ap_prev_vr = 0.0
        self._ap_frame_cnt = 0
        self._ap_phase_cache_time = -1e9
        self._ap_phase_cache_req  = 0.0
        self._ap_prev_err = None
        self._ap_err_history = []

        self.autopilot_mode = "Rendezvous"
        self.mode_btn = None
        self.flyby_returned = False

        self.return_to_earth_active = False
        self.return_phase = None
        self.return_a_transfer = 0.0
        self.return_transfer_time = 0.0
        self.return_btn = None
        self.return_burn3_done = False
        self._return_correction_done = False
        # Burn3 gradual state
        self._burn3_dvx_rem = 0.0   # sisa Δv komponen x yang belum diberikan
        self._burn3_dvy_rem = 0.0   # sisa Δv komponen y yang belum diberikan
        self._burn3_dv_rate = 50.0  # m/s per detik (thrust rate burn3)

        self.lock_zoom = False
        self._locked_cx = 0.0
        self._locked_cy = 0.0
        self._locked_scale = 1.0

        # ── Gravitasi Bulan ──────────────────────────────────────────
        self.moon_grav_on = False    # aktif/nonaktif lewat tombol 0

        # Autopilot launch
        self.autopilot_launch_active = False
        self.launch_phase = None
        self.launch_target_alt = 325.0
        self.launch_dv_rate = 80.0
        self.launch_alt_threshold = 100.0
        self.launch_btn = None
        self.launch_cancel_btn = None
        self.launch_alt_peak = 0.0
        self.launch_time_since_peak = 0.0

        self._cache_frame = 0
        self._cached_cur_ox = []
        self._cached_cur_oy = []
        self._cached_ghost_x = []
        self._cached_ghost_y = []
        self._orbit_cache_valid = False
        self._energy_dirty = True

        self._static_objects = []
        self._orbit_lines = {}

        self._build_ui()

        self.root.update_idletasks()
        self._update_ghost()
        self._draw_all()

        self.c_orbit.bind("<Configure>",  lambda e: self._draw_all())
        self.c_energy.bind("<Configure>", lambda e: self._draw_all())
        self.c_info.bind("<Configure>",   lambda e: self._draw_all())

        self._setup_keyboard_shortcuts()

    # ---------- KEYBOARD SHORTCUTS ----------
    def _setup_keyboard_shortcuts(self):
        r = self.root

        # ── 1. Kontrol Simulasi Utama ──────────────────────────────────────
        r.bind("<space>",    lambda e: self._toggle_run())
        r.bind("<r>",        lambda e: self._reset())
        r.bind("<R>",        lambda e: self._reset())
        r.bind("<m>",        lambda e: self._fire_maneuver())
        r.bind("<M>",        lambda e: self._fire_maneuver())
        r.bind("<g>",        lambda e: self._toggle_ghost())
        r.bind("<G>",        lambda e: self._toggle_ghost())
        r.bind("<o>",        lambda e: self._toggle_orbit_line())
        r.bind("<O>",        lambda e: self._toggle_orbit_line())
        r.bind("<l>",        lambda e: self._toggle_lock_zoom())
        r.bind("<L>",        lambda e: self._toggle_lock_zoom())

        # ── 2. Thruster Manual ─────────────────────────────────────────────
        r.bind("<KeyPress-z>",   self._kb_burn_press)
        r.bind("<KeyPress-Z>",   self._kb_burn_press)
        r.bind("<KeyRelease-z>", self._kb_burn_release)
        r.bind("<KeyRelease-Z>", self._kb_burn_release)

        def change_angle(delta):
            v = self.thrust_angle_deg.get()
            self.thrust_angle_deg.set((v + delta) % 360)
            self._update_ghost()
            if not self.running: self._draw_all()

        def change_rate(delta):
            v = self.thrust_dv_rate.get()
            self.thrust_dv_rate.set(max(1, min(2000, round(v + delta, 1))))

        r.bind("<comma>",        lambda e: change_angle(-5))
        r.bind("<period>",       lambda e: change_angle(+5))
        r.bind("<bracketleft>",  lambda e: change_rate(-10))
        r.bind("<bracketright>", lambda e: change_rate(+10))

        def set_preset(ang):
            self.thrust_angle_deg.set(ang)
            self._update_ghost()
            if not self.running: self._draw_all()

        r.bind("<Key-1>", lambda e: set_preset(0))
        r.bind("<Key-2>", lambda e: set_preset(90))
        r.bind("<Key-3>", lambda e: set_preset(180))
        r.bind("<Key-4>", lambda e: set_preset(270))

        # ── 2b. Zoom Cepat & Grafik & Riwayat ───────────────────────────────
        def toggle_focus_satellite():
            self.follow_satellite = not self.follow_satellite
            self.status_var.set("🎯 Fokus satelit " + ("ON" if self.follow_satellite else "OFF"))
            if not self.running: self._draw_all()

        r.bind("<Key-5>", lambda e: toggle_focus_satellite())
        r.bind("<Key-6>", lambda e: self._zoom_to_moon())
        r.bind("<Key-7>", lambda e: self._zoom_to_mars())
        r.bind("<Key-8>", lambda e: self._open_energy_example_window())
        r.bind("<Key-9>", lambda e: self._clear_history())
        r.bind("<Key-0>", lambda e: self._toggle_moon_gravity())

        # ── 3. Zoom & Pan ──────────────────────────────────────────────────
        PAN_STEP = 40  # piksel per tekanan panah

        def fake_scroll(factor):
            class _Ev: pass
            ev = _Ev()
            ev.num   = 4 if factor > 1 else 5
            ev.delta = 1 if factor > 1 else -1
            c = self.c_orbit
            ev.x = (c.winfo_width()  or 800) // 2
            ev.y = (c.winfo_height() or 600) // 2
            self._on_scroll(ev)

        r.bind("<plus>",        lambda e: fake_scroll(1.2))
        r.bind("<equal>",       lambda e: fake_scroll(1.2))
        r.bind("<minus>",       lambda e: fake_scroll(1/1.2))

        def pan(dx, dy):
            if self.lock_zoom:
                self._locked_cx += dx
                self._locked_cy += dy
            else:
                self.pan_cx_off += dx
                self.pan_cy_off += dy
            if not self.running: self._draw_all()

        r.bind("<Up>",          lambda e: pan(0, +PAN_STEP))
        r.bind("<Down>",        lambda e: pan(0, -PAN_STEP))
        r.bind("<Left>",        lambda e: pan(+PAN_STEP, 0))
        r.bind("<Right>",       lambda e: pan(-PAN_STEP, 0))
        r.bind("<BackSpace>",   lambda e: (self._reset_zoom(),
                                           self._draw_all() if not self.running else None))

        # ── 4. Slider Parameter ────────────────────────────────────────────
        def adj(var, delta, lo, hi, step, cmd=None):
            cur = var.get()
            var.set(max(lo, min(hi, round(cur + delta * step, 10))))
            if cmd: cmd()

        r.bind("<q>", lambda e: adj(self.init_alt_var, -1, 0, 2000, 10,
                                    self._on_init_change))
        r.bind("<Q>", lambda e: adj(self.init_alt_var, -1, 0, 2000, 10,
                                    self._on_init_change))
        r.bind("<w>", lambda e: adj(self.init_alt_var, +1, 0, 2000, 10,
                                    self._on_init_change))
        r.bind("<W>", lambda e: adj(self.init_alt_var, +1, 0, 2000, 10,
                                    self._on_init_change))
        r.bind("<a>", lambda e: adj(self.init_ecc_var, -1, 0.0, 0.95, 0.01,
                                    self._on_init_change))
        r.bind("<A>", lambda e: adj(self.init_ecc_var, -1, 0.0, 0.95, 0.01,
                                    self._on_init_change))
        r.bind("<s>", lambda e: adj(self.init_ecc_var, +1, 0.0, 0.95, 0.01,
                                    self._on_init_change))
        r.bind("<S>", lambda e: adj(self.init_ecc_var, +1, 0.0, 0.95, 0.01,
                                    self._on_init_change))
        r.bind("<y>", lambda e: adj(self.man_alt_var,  -1, 160, 5000, 10,
                                    self._on_man_change))
        r.bind("<Y>", lambda e: adj(self.man_alt_var,  -1, 160, 5000, 10,
                                    self._on_man_change))
        r.bind("<u>", lambda e: adj(self.man_alt_var,  +1, 160, 5000, 10,
                                    self._on_man_change))
        r.bind("<U>", lambda e: adj(self.man_alt_var,  +1, 160, 5000, 10,
                                    self._on_man_change))
        r.bind("<h>", lambda e: adj(self.man_ecc_var,  -1, 0.0, 0.95, 0.01,
                                    self._on_man_change))
        r.bind("<H>", lambda e: adj(self.man_ecc_var,  -1, 0.0, 0.95, 0.01,
                                    self._on_man_change))
        r.bind("<j>", lambda e: adj(self.man_ecc_var,  +1, 0.0, 0.95, 0.01,
                                    self._on_man_change))
        r.bind("<J>", lambda e: adj(self.man_ecc_var,  +1, 0.0, 0.95, 0.01,
                                    self._on_man_change))

        # ── 4b. dt Simulasi (tahan D/F untuk ubah berulang) & Kecepatan Sim ──
        def step_dt(delta):
            cur = self.dt_var.get()
            new = max(1, min(100000, cur + delta))
            self.dt_var.set(new)
            self._on_dt_change()

        r.bind("<KeyPress-d>",   lambda e: self._kb_dt_press(-1))
        r.bind("<KeyPress-D>",   lambda e: self._kb_dt_press(-1))
        r.bind("<KeyRelease-d>", lambda e: self._kb_dt_release())
        r.bind("<KeyRelease-D>", lambda e: self._kb_dt_release())
        r.bind("<KeyPress-f>",   lambda e: self._kb_dt_press(+1))
        r.bind("<KeyPress-F>",   lambda e: self._kb_dt_press(+1))
        r.bind("<KeyRelease-f>", lambda e: self._kb_dt_release())
        r.bind("<KeyRelease-F>", lambda e: self._kb_dt_release())
        self._step_dt_fn = step_dt

        r.bind("<Control-comma>",  lambda e: adj(self.speed_var, -1, 1, 20, 1,
                                             self._on_speed_change))
        r.bind("<Control-period>", lambda e: adj(self.speed_var, +1, 1, 20, 1,
                                              self._on_speed_change))

        # ── 5. Autopilot & Navigasi ────────────────────────────────────────
        r.bind("<n>", lambda e: self._autopilot_to_moon())
        r.bind("<N>", lambda e: self._autopilot_to_moon())
        r.bind("<p>", lambda e: self._autopilot_to_mars())
        r.bind("<P>", lambda e: self._autopilot_to_mars())
        r.bind("<x>", lambda e: self._autopilot_cancel())
        r.bind("<X>", lambda e: self._autopilot_cancel())
        r.bind("<b>", lambda e: self._toggle_autopilot_mode())
        r.bind("<B>", lambda e: self._toggle_autopilot_mode())
        r.bind("<k>", lambda e: self._start_return_to_earth())
        r.bind("<K>", lambda e: self._start_return_to_earth())

        # ── 6. Launch ──────────────────────────────────────────────────────
        r.bind("<v>", lambda e: self._autopilot_launch())
        r.bind("<V>", lambda e: self._autopilot_launch())
        r.bind("<c>", lambda e: self._cancel_launch())
        r.bind("<C>", lambda e: self._cancel_launch())

    def _kb_burn_press(self, event=None):
        """Keyboard Z press – mulai BURN (tidak repeat jika sudah aktif)."""
        if not self._thrust_active:
            self._on_burn_press()

    def _kb_burn_release(self, event=None):
        """Keyboard Z release – hentikan BURN."""
        if self._thrust_active:
            self._on_burn_release()

    def _toggle_moon_gravity(self):
        """Toggle efek gravitasi Bulan terhadap satelit (shortcut: 0)."""
        global _moon_grav_on
        self.moon_grav_on = not self.moon_grav_on
        _moon_grav_on = self.moon_grav_on
        state = "ON 🌕" if self.moon_grav_on else "OFF"
        self.status_var.set("Gravitasi Bulan: %s  (tekan 0 untuk toggle)" % state)
        if hasattr(self, 'moon_grav_btn'):
            if self.moon_grav_on:
                self.moon_grav_btn.config(text="🌕 Grav Bulan: ON  [0]",
                                          bg="#0a2a0a", fg=LIME)
            else:
                self.moon_grav_btn.config(text="🌕 Grav Bulan: OFF [0]",
                                          bg="#0a1a0a", fg="#aaffaa")
        if not self.running:
            self._draw_all()

    def _kb_dt_press(self, direction):
        """Keyboard D/F press – ubah dt sim ±1, berulang selama tombol ditahan."""
        if self._dt_repeat_dir == direction:
            return  # sudah berjalan ke arah ini (hindari restart akibat OS key-repeat)
        self._dt_repeat_dir = direction
        if self._dt_repeat_after is not None:
            try: self.root.after_cancel(self._dt_repeat_after)
            except Exception: pass
            self._dt_repeat_after = None
        self._dt_repeat_tick()

    def _kb_dt_release(self):
        """Keyboard D/F release – hentikan perubahan dt berulang."""
        self._dt_repeat_dir = 0
        if self._dt_repeat_after is not None:
            try: self.root.after_cancel(self._dt_repeat_after)
            except Exception: pass
            self._dt_repeat_after = None

    def _dt_repeat_tick(self):
        if self._dt_repeat_dir == 0:
            self._dt_repeat_after = None
            return
        self._step_dt_fn(self._dt_repeat_dir)
        self._dt_repeat_after = self.root.after(60, self._dt_repeat_tick)

    # ---------- ASTROPY HELPERS ----------
    def _astropy_body_xy(self, body_name, sim_time_seconds):
        t = self.sim_epoch + sim_time_seconds * u.second
        body = get_body(body_name, t).gcrs
        xyz  = body.represent_as('cartesian')
        x = float(xyz.x.to(u.m).value)
        y = float(xyz.y.to(u.m).value)
        r = float(math.sqrt(x*x + y*y))
        return x, y, r

    def _update_astropy_bodies(self, sim_time):
        if sim_time - self._astropy_cache_time < self._astropy_cache_interval:
            return
        self._astropy_cache_time = sim_time

        moon_x, moon_y, moon_r = self._astropy_body_xy('moon', sim_time)
        mars_x, mars_y, mars_r = self._astropy_body_xy('mars', sim_time)

        self._moon_x = moon_x; self._moon_y = moon_y
        self._mars_x = mars_x; self._mars_y = mars_y
        self.MOON_DIST  = moon_r
        self.MARS_DIST  = mars_r
        self.moon_angle = math.atan2(moon_y, moon_x)
        self.mars_angle = math.atan2(mars_y, mars_x)

    # ---------- INITIAL ORBIT ----------
    def _apply_initial_orbit(self):
        alt = self.init_alt_var.get()
        ecc = min(0.95, max(0.0, self.init_ecc_var.get()))
        if alt <= 0:
            self.sat_x  = R_BUMI
            self.sat_y  = 0.0
            self.sat_vx = 0.0
            self.sat_vy = 200.0
            self._was_above_surface = False
        else:
            x, y, vx, vy = compute_orbit_velocity(alt, ecc)
            self.sat_x  = x;  self.sat_y  = y
            self.sat_vx = vx; self.sat_vy = vy
            self._was_above_surface = True

    # ---------- UI BUILD ----------
    def _build_ui(self):
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=0)
        self.root.rowconfigure(2, weight=0)
        self.root.rowconfigure(3, weight=0)
        self.root.columnconfigure(0, weight=1)

        top = tk.Frame(self.root, bg=BG)
        top.grid(row=0, column=0, sticky='nsew', padx=4, pady=(4,2))
        top.rowconfigure(0, weight=1)
        top.columnconfigure(0, weight=3)
        top.columnconfigure(1, weight=1)

        self.c_orbit = tk.Canvas(top, bg=BG, highlightthickness=1, highlightbackground=GRAY)
        self.c_orbit.grid(row=0, column=0, sticky='nsew', padx=(0,4))
        self.c_orbit.bind("<MouseWheel>",      self._on_scroll)
        self.c_orbit.bind("<Button-4>",        self._on_scroll)
        self.c_orbit.bind("<Button-5>",        self._on_scroll)
        self.c_orbit.bind("<ButtonPress-1>",   self._on_pan_start)
        self.c_orbit.bind("<B1-Motion>",       self._on_pan_move)
        self.c_orbit.bind("<ButtonRelease-1>", self._on_pan_end)
        self.c_orbit.bind("<Double-Button-1>", self._on_zoom_reset)

        right = tk.Frame(top, bg=BG)
        right.grid(row=0, column=1, sticky='nsew')
        right.rowconfigure(0, weight=1)
        right.rowconfigure(1, weight=2)
        right.columnconfigure(0, weight=1)

        self.c_energy = tk.Canvas(right, bg=BG, highlightthickness=1, highlightbackground=GRAY)
        self.c_energy.grid(row=0, column=0, sticky='nsew', pady=(0,3))

        self.c_info = tk.Canvas(right, bg=BG, highlightthickness=1, highlightbackground=GRAY)
        self.c_info.grid(row=1, column=0, sticky='nsew')

        ctrl = tk.Frame(self.root, bg=BG)
        ctrl.grid(row=1, column=0, sticky='ew', padx=4, pady=(0,2))

        def section_label(parent, text, row, col, colspan=5):
            tk.Label(parent, text=text, bg=BG, fg=YELLOW,
                     font=("Courier",7,'bold')).grid(
                     row=row, column=col, columnspan=colspan, sticky='w', padx=2)

        section_label(ctrl, "── ORBIT AWAL ──────────────", 0, 0)
        self._slider(ctrl, "Altitude awal (km)", self.init_alt_var,
                     0, 2000, 1, col=0, step=10, command=self._on_init_change)
        self._slider(ctrl, "Eksentrisitas awal", self.init_ecc_var,
                     0.0, 0.95, 2, col=0, step=0.01, resolution=0.01,
                     command=self._on_init_change, color_var=CYAN)

        section_label(ctrl, "── MANUVER ──────────────────", 0, 5)
        self._slider(ctrl, "Altitude manuver (km)", self.man_alt_var,
                     160, 5000, 1, col=5, step=10, command=self._on_man_change)
        self._slider(ctrl, "Eksentrisitas manuver", self.man_ecc_var,
                     0.0, 0.95, 2, col=5, step=0.01, resolution=0.01,
                     command=self._on_man_change, color_var=ORANGE)

        section_label(ctrl, "── SIM ─────────────────────", 0, 10)
        self._slider(ctrl, "dt (s)",        self.dt_var,    1, 100000, 1, col=10,
                     step=100, command=self._on_dt_change, display_var=self._dt_label_var,
                     length=200)
        self._slider(ctrl, "Kecepatan sim", self.speed_var, 1, 20,     2, col=10,
                     step=1, command=self._on_speed_change, length=200)

        btn_f = tk.Frame(ctrl, bg=BG)
        btn_f.grid(row=0, column=15, rowspan=3, padx=(8,4), sticky='ns')

        col_L = tk.Frame(btn_f, bg=BG)
        col_L.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0,2))
        col_R = tk.Frame(btn_f, bg=BG)
        col_R.pack(side=tk.LEFT, fill=tk.BOTH)

        def btn(text, bg_c, fg_c, cmd, parent=None):
            p = parent if parent else col_L
            b = tk.Button(p, text=text, bg=bg_c, fg=fg_c,
                          width=22, font=("Courier",7),
                          wraplength=180, command=cmd)
            b.pack(fill=tk.X, pady=1)
            return b

        btn("▶ / ⏸  Jalankan",   "#1a3a1a", LIME,      self._toggle_run,        col_L)
        btn("🚀 Tembak Manuver",  "#1a1a3a", CYAN,      self._fire_maneuver,     col_L)
        btn("↺  Reset",           "#2a1a1a", RED,       self._reset,             col_L)
        btn("👁 Ghost On/Off",    "#1a1a2e", YELLOW,    self._toggle_ghost,      col_L)
        btn("〇 Garis Orbit",     "#1a1a2e", CYAN,      self._toggle_orbit_line, col_L)
        btn("🗑 Hapus Riwayat",   "#1a1a1a", GRAY,      self._clear_history,     col_L)
        btn("🔒 Lock Zoom",      "#1a1a3a", "#ffaa44", self._toggle_lock_zoom,  col_L)
        self.launch_btn = btn("🚀 Launch to 325 km", "#1a3a1a", LAUNCH_COLOR,
                              self._autopilot_launch, col_L)
        self.launch_cancel_btn = btn("✖ Batal Launch", "#3a1a1a", RED,
                                     self._cancel_launch, col_L)
        self.launch_cancel_btn.config(state=tk.DISABLED)

        btn("🌕 Zoom ke Bulan",        "#1a1a2e", "#aabbdd", self._zoom_to_moon,               col_R)
        btn("🔴 Zoom ke Mars",         "#2a1a0a", "#ff9966", self._zoom_to_mars,               col_R)
        btn("📊 Grafik Energi",        "#1a3a3a", "#88ff88", self._open_energy_example_window, col_R)
        btn("🌕 Auto → Bulan",         "#0a1a2a", "#aaccff", self._autopilot_to_moon,          col_R)
        btn("🔴 Auto → Mars",          "#1a0800", "#ff7744", self._autopilot_to_mars,          col_R)
        btn("✖ Batal Autopilot",       "#1a0000", RED,       self._autopilot_cancel,           col_R)

        self.mode_btn = btn("🔄 Mode: Rendezvous", "#1a2a1a", "#88ff88",
                            self._toggle_autopilot_mode, col_R)

        self.return_btn = btn("🔄 Kembali Bumi (orbit)", "#1a2a3a", "#66ddff",
                              self._start_return_to_earth, col_R)

        self.moon_grav_btn = btn("🌕 Grav Bulan: OFF [0]", "#0a1a0a", "#aaffaa",
                                 self._toggle_moon_gravity, col_R)

        # Row 2: THRUSTER manual
        thr = tk.Frame(self.root, bg="#0a0a18",
                       highlightthickness=1, highlightbackground="#223355")
        thr.grid(row=2, column=0, sticky='ew', padx=4, pady=(0,2))

        tk.Label(thr, text="── THRUSTER MANUAL ─────────────────────────────────────────────────────",
                 bg="#0a0a18", fg="#ff9900", font=("Courier",7,'bold')).grid(
                 row=0, column=0, columnspan=20, sticky='w', padx=4, pady=(2,0))

        tk.Label(thr, text="Δv/s (m/s):", bg="#0a0a18", fg=TEXT_COL,
                 font=("Courier",7), width=12, anchor='w').grid(row=1, column=0, padx=(6,1))
        def do_rate_step(d):
            v = self.thrust_dv_rate.get()
            self.thrust_dv_rate.set(max(1, min(2000, round(v + d*10, 1))))
        tk.Button(thr, text="◀", bg="#1a1a2a", fg=CYAN, width=2,
                  font=("Courier",7), relief=tk.FLAT,
                  command=lambda: do_rate_step(-1)).grid(row=1, column=1, padx=0)
        tk.Scale(thr, from_=1, to=2000, orient=tk.HORIZONTAL, variable=self.thrust_dv_rate,
                 bg="#0a0a18", fg=WHITE, troughcolor=GRAY, highlightthickness=0,
                 length=140, resolution=1, showvalue=False
                 ).grid(row=1, column=2, padx=1)
        tk.Button(thr, text="▶", bg="#1a1a2a", fg=CYAN, width=2,
                  font=("Courier",7), relief=tk.FLAT,
                  command=lambda: do_rate_step(+1)).grid(row=1, column=3, padx=0)
        tk.Label(thr, textvariable=self.thrust_dv_rate, bg="#0a0a18", fg=ORANGE,
                 width=5, font=("Courier",7), anchor='w').grid(row=1, column=4, padx=(2,10))

        tk.Label(thr, text="Arah burn:", bg="#0a0a18", fg=TEXT_COL,
                 font=("Courier",7), width=10, anchor='w').grid(row=1, column=5, padx=(6,1))
        def do_angle_step(d):
            v = self.thrust_angle_deg.get()
            self.thrust_angle_deg.set((v + d*5) % 360)
            self._update_ghost()
            if not self.running: self._draw_all()
        tk.Button(thr, text="◀", bg="#1a1a2a", fg=CYAN, width=2,
                  font=("Courier",7), relief=tk.FLAT,
                  command=lambda: do_angle_step(-1)).grid(row=1, column=6, padx=0)
        tk.Scale(thr, from_=0, to=359, orient=tk.HORIZONTAL, variable=self.thrust_angle_deg,
                 bg="#0a0a18", fg=WHITE, troughcolor=GRAY, highlightthickness=0,
                 length=160, resolution=1, showvalue=False,
                 command=lambda _: (self._update_ghost(), self._draw_all() if not self.running else None)
                 ).grid(row=1, column=7, padx=1)
        tk.Button(thr, text="▶", bg="#1a1a2a", fg=CYAN, width=2,
                  font=("Courier",7), relief=tk.FLAT,
                  command=lambda: do_angle_step(+1)).grid(row=1, column=8, padx=0)

        self._angle_label_var = tk.StringVar(value="0° [Prograde]")
        tk.Label(thr, textvariable=self._angle_label_var, bg="#0a0a18", fg=LIME,
                 width=18, font=("Courier",7), anchor='w').grid(row=1, column=9, padx=(2,10))

        def _update_angle_label(*_):
            a = self.thrust_angle_deg.get() % 360
            if   a < 22.5  or a >= 337.5: name = "Prograde ▶"
            elif a < 67.5:                 name = "Prograde+Normal"
            elif a < 112.5:                name = "Normal (atas)"
            elif a < 157.5:                name = "Retrograde+Normal"
            elif a < 202.5:                name = "Retrograde ◀"
            elif a < 247.5:                name = "Retro+Radial-in"
            elif a < 292.5:                name = "Radial-in ▼"
            else:                          name = "Prograde+Radial-in"
            self._angle_label_var.set("%.0f° [%s]" % (a, name))
        self.thrust_angle_deg.trace_add('write', _update_angle_label)
        _update_angle_label()

        def preset_frame():
            pf = tk.Frame(thr, bg="#0a0a18")
            pf.grid(row=1, column=10, padx=(4,2))
            presets = [
                ("PRO", 0,   "#1a3a1a", LIME),
                ("NOR", 90,  "#1a2a3a", CYAN),
                ("RET", 180, "#2a1a1a", RED),
                ("RAD", 270, "#2a2a1a", YELLOW),
            ]
            for i, (lbl, ang, bg_c, fg_c) in enumerate(presets):
                def make_cmd(a=ang):
                    def cmd():
                        self.thrust_angle_deg.set(a)
                        self._update_ghost()
                        if not self.running: self._draw_all()
                    return cmd
                tk.Button(pf, text=lbl, bg=bg_c, fg=fg_c, width=4,
                          font=("Courier",7,'bold'), relief=tk.FLAT,
                          command=make_cmd()).grid(row=0, column=i, padx=1)
        preset_frame()

        self._burn_btn = tk.Button(thr, text="🔥 BURN\n(tahan)",
                                   bg="#220000", fg=RED, width=10,
                                   font=("Courier",8,'bold'), relief=tk.RAISED)
        self._burn_btn.grid(row=1, column=11, padx=(14,4), ipady=2)
        self._burn_btn.bind("<ButtonPress-1>",   self._on_burn_press)
        self._burn_btn.bind("<ButtonRelease-1>", self._on_burn_release)

        self.total_dv_var = tk.StringVar(value="Total Δv: 0 m/s")
        tk.Label(thr, textvariable=self.total_dv_var, bg="#0a0a18", fg=YELLOW,
                 font=("Courier",7), width=18, anchor='w').grid(row=1, column=12, padx=(4,4))

        self.status_var = tk.StringVar(value="🚀 Mode PELUNCURAN: Alt=0 km → Jalankan → satelit naik dari permukaan → BURN prograde untuk orbit!")
        tk.Label(self.root, textvariable=self.status_var,
                 bg="#0a0a14", fg=TEXT_COL, font=("Courier",8),
                 anchor='w').grid(row=3, column=0, sticky='ew', padx=4, pady=(0,2))

    # ---------- SLIDER ----------
    def _slider(self, parent, label, var, lo, hi, row, col=0,
                step=1, command=None, resolution=1, color_var=TEXT_COL,
                display_var=None, length=110):
        tk.Label(parent, text=label, bg=BG, fg=TEXT_COL,
                 width=18, anchor='w',
                 font=("Courier",7)).grid(row=row, column=col, padx=(4,1), pady=2)
        def do_step(delta):
            cur = var.get()
            new = round(cur + delta * step, 10)
            new = max(lo, min(hi, new))
            var.set(new)
            if command: command()
        tk.Button(parent, text="◀", bg="#1a1a2a", fg=CYAN, width=2,
                  font=("Courier",7), relief=tk.FLAT,
                  command=lambda: do_step(-1)).grid(row=row, column=col+1, padx=0)
        cmd = (lambda _: command()) if command else None
        tk.Scale(parent, from_=lo, to=hi, orient=tk.HORIZONTAL,
                 variable=var, bg=BG, fg=WHITE, troughcolor=GRAY,
                 highlightthickness=0, length=length, resolution=resolution,
                 showvalue=False, command=cmd
                 ).grid(row=row, column=col+2, padx=1)
        tk.Button(parent, text="▶", bg="#1a1a2a", fg=CYAN, width=2,
                  font=("Courier",7), relief=tk.FLAT,
                  command=lambda: do_step(+1)).grid(row=row, column=col+3, padx=0)
        tk.Label(parent, textvariable=display_var if display_var else var, bg=BG, fg=color_var,
                 width=6, font=("Courier",7),
                 anchor='w').grid(row=row, column=col+4, padx=(2,6))

    # ---------- ORBIT MANAGEMENT ----------
    def _new_orbit_segment(self):
        color = ORBIT_COLORS[self.orbit_idx % len(ORBIT_COLORS)]
        label = "Orbit #%d" % (self.orbit_idx + 1)
        rec   = OrbitRecord(color, label)
        self.orbits.append(rec)
        self.orbit_idx += 1
        return rec

    @property
    def current_orbit(self):
        return self.orbits[-1]

    # ---------- SLIDER CALLBACKS ----------
    def _on_init_change(self):
        if not self.running:
            self._apply_initial_orbit()
            self.orbits = []
            self.orbit_idx = 0
            self._new_orbit_segment()
            self.energy_history = []
            self._energy_rv_history = []
            self.sim_time = 0.0
            self.maneuver_times = []
            self.maneuver_flash = 0
            self.perm_trail_x = []
            self.perm_trail_y = []
            self._update_ghost()
            self._draw_all()
            self._show_orbit_type_status("awal")
            self._update_launch_button_state()

    def _on_man_change(self):
        self._update_ghost()
        if not self.running:
            self._draw_all()

    def _on_dt_change(self):
        self.sim_dt = max(1, int(self.dt_var.get()))
        global _MAX_DT
        _MAX_DT = max(5.0, float(self.sim_dt))
        v = self.sim_dt
        if v < 60:
            self._dt_label_var.set("%ds" % v)
        elif v < 3600:
            self._dt_label_var.set("%.1fm" % (v/60))
        else:
            self._dt_label_var.set("%.1fj" % (v/3600))

    def _on_speed_change(self):
        self.steps_per_frame = max(1, int(self.speed_var.get()))

    def _show_orbit_type_status(self, which="awal"):
        ecc = self.init_ecc_var.get() if which == "awal" else self.man_ecc_var.get()
        alt = self.init_alt_var.get() if which == "awal" else self.man_alt_var.get()
        if ecc < 0.05:
            otype = "SIRKULAR"
        elif ecc < 1.0:
            otype = "ELIPS (e=%.2f)" % ecc
        else:
            otype = "HIPERBOLA"
        self.status_var.set("Orbit %s: %s  |  Altitude: %.0f km" % (which, otype, alt))

    def _update_launch_button_state(self):
        alt = (math.sqrt(self.sat_x**2 + self.sat_y**2) - R_BUMI) / 1000
        if alt < 1.0 and not self.running and not self.autopilot_launch_active:
            self.launch_btn.config(state=tk.NORMAL)
            self.launch_cancel_btn.config(state=tk.DISABLED)
        else:
            self.launch_btn.config(state=tk.DISABLED)
            if self.autopilot_launch_active:
                self.launch_cancel_btn.config(state=tk.NORMAL)
            else:
                self.launch_cancel_btn.config(state=tk.DISABLED)

    # ================================================================
    # AUTOPILOT LAUNCH (dari permukaan ke 325 km) - DIPERBAIKI
    # ================================================================
    def _autopilot_launch(self):
        if self.autopilot_launch_active:
            return
        alt = (math.sqrt(self.sat_x**2 + self.sat_y**2) - R_BUMI) / 1000
        if alt > 1.0:
            self.status_var.set("⚠ Satelit tidak berada di permukaan (alt > 1 km).")
            return
        if self.running:
            self.status_var.set("⚠ Hentikan simulasi dulu sebelum launch.")
            return

        self.sat_x = R_BUMI
        self.sat_y = 0.0
        self.sat_vx = 0.0
        self.sat_vy = 200.0
        self._was_above_surface = True
        self.orbits = []
        self.orbit_idx = 0
        self._new_orbit_segment()
        self.energy_history = []
        self._energy_rv_history = []
        self.sim_time = 0.0
        self.maneuver_times = []
        self.perm_trail_x = []
        self.perm_trail_y = []
        self.total_dv_applied = 0.0
        self.total_dv_var.set("Total Δv: 0 m/s")

        self.autopilot_launch_active = True
        self.launch_phase = 'vertical'
        self.launch_target_alt = 325.0
        self.launch_alt_threshold = 100.0
        self.launch_dv_rate = 80.0
        self.launch_alt_peak = 0.0
        self.launch_time_since_peak = 0.0
        if hasattr(self, '_launch_prev_vr'):
            del self._launch_prev_vr
        self.status_var.set("🚀 Autopilot Launch aktif: fase VERTICAL (menuju 325 km)")
        self.launch_btn.config(state=tk.DISABLED)
        self.launch_cancel_btn.config(state=tk.NORMAL)

        if not self.running:
            self.running = True
            self._sim_step()

    def _cancel_launch(self):
        if self.autopilot_launch_active:
            self.autopilot_launch_active = False
            self.launch_phase = 'done'
            if hasattr(self, '_launch_prev_vr'):
                del self._launch_prev_vr
            self.status_var.set("✖ Launch dibatalkan oleh pengguna.")
            self.launch_btn.config(state=tk.DISABLED)
            self.launch_cancel_btn.config(state=tk.DISABLED)
            self._was_above_surface = True
            self._update_ghost()
            self._draw_all()

    def _launch_check(self):
        """Autopilot launch dari permukaan ke orbit sirkular 325 km.

        Strategi: 2 fase TANPA coast
          'vertical' – Thrust radial murni sampai 20 km
          'ascent'   – Thrust campuran radial+tangensial kontinu.
                       Semakin besar v_tan/v_circ_target, semakin murni tangensial.
                       Selesai saat alt >= 325 km DAN v_tan >= v_circ(325 km).
                       Tidak ada coast — periapsis tidak bisa jatuh ke bawah permukaan.
        """
        if not self.autopilot_launch_active:
            return

        r   = math.sqrt(self.sat_x**2 + self.sat_y**2)
        alt = (r - R_BUMI) / 1000.0
        vx, vy = self.sat_vx, self.sat_vy
        v   = math.sqrt(vx*vx + vy*vy)
        target_alt = self.launch_target_alt   # 325 km
        r_target   = R_BUMI + target_alt * 1e3
        dt  = self._current_dt_sub            # <= 1 detik

        # Unit vektor
        rx = self.sat_x / r;  ry = self.sat_y / r   # radial (ke atas)
        tx = -ry;              ty =  rx               # tangensial CCW (prograde orbit)

        # Kecepatan orbital sirkular di 325 km (tetap, tidak berubah)
        v_circ_target = math.sqrt(G * M / r_target)

        # ── Fase 1: VERTICAL (0 – 20 km) ──────────────────────────────────
        if self.launch_phase == 'vertical':
            dv = self.launch_dv_rate * dt
            self.sat_vx += dv * rx
            self.sat_vy += dv * ry
            self.total_dv_applied += dv
            self.total_dv_var.set("Total Dv: %.1f m/s" % self.total_dv_applied)
            if alt >= 20.0:
                self.launch_phase = 'ascent'
                self.status_var.set("Membangun kecepatan orbital...")

        # ── Fase 2: ASCENT — thrust kontinu tanpa coast ────────────────────
        elif self.launch_phase == 'ascent':
            # Komponen kecepatan tangensial saat ini
            v_tan = vx * tx + vy * ty

            # Proporsi tangensial meningkat seiring v_tan mendekati target
            ratio = min(1.0, max(0.0, v_tan / v_circ_target))
            # Di awal ratio kecil → banyak radial agar terus naik
            # Di akhir ratio besar → hampir murni tangensial
            radial_frac = (1.0 - ratio) * 0.45
            tang_frac   = 1.0 - radial_frac

            bx = radial_frac * rx + tang_frac * tx
            by = radial_frac * ry + tang_frac * ty
            bm = math.sqrt(bx*bx + by*by)
            if bm > 1e-9:
                bx /= bm; by /= bm

            dv = self.launch_dv_rate * dt
            self.sat_vx += dv * bx
            self.sat_vy += dv * by
            self.total_dv_applied += dv
            self.total_dv_var.set("Total Dv: %.1f m/s" % self.total_dv_applied)

            # Info orbit saat ini
            el = orbital_elements(self.sat_x, self.sat_y, self.sat_vx, self.sat_vy)
            _, alt_p, _, alt_a = periapsis_apoapsis(el)
            ap_str = "%.0f" % alt_a if alt_a is not None else "sub-orb"
            pe_str = "%.0f" % alt_p if alt_p is not None else "sub-orb"
            self.status_var.set(
                "ASCENT: alt=%.0f km  Pe=%s  Ap=%s  v_tan=%.0f/%.0f m/s" % (
                    alt, pe_str, ap_str, v_tan, v_circ_target))

            # ── Kondisi selesai: v_tan sudah >= v_circ target DAN alt >= 320 km
            if v_tan >= v_circ_target * 0.998 and alt >= target_alt - 10.0:
                # Snap posisi ke r_target persis (325 km), pertahankan arah radial
                self.sat_x = r_target * rx
                self.sat_y = r_target * ry
                # Snap kecepatan ke orbit sirkular sempurna di 325 km
                self.sat_vx = v_circ_target * tx
                self.sat_vy = v_circ_target * ty

                self.autopilot_launch_active = False
                self.launch_phase = 'done'
                self.status_var.set(
                    "✅ Launch selesai! alt=325 km  Pe=325 km  Ap=325 km  v=%.1f m/s" % v_circ_target)
                self.launch_btn.config(state=tk.DISABLED)
                self.launch_cancel_btn.config(state=tk.DISABLED)
                self.maneuver_times.append(self.sim_time)
                self.maneuver_flash = 12
                self._new_orbit_segment()
                self._update_ghost()
                self._was_above_surface = True

    # ---------- SIMULATION LOOP ----------
    def _toggle_run(self):
        if self.running:
            self.running = False
            if self.after_id:
                self.root.after_cancel(self.after_id)
                self.after_id = None
            self.status_var.set("Dijeda.")
        else:
            self.running = True
            self._sim_step()

    def _sim_step(self):
        if not self.running:
            return
        dt  = self.sim_dt
        spf = self.steps_per_frame

        # ── Sinkronkan posisi Bulan ke fungsi ODE global ─────────────
        global _moon_grav_x, _moon_grav_y, _moon_grav_on
        _moon_grav_x  = self._moon_x
        _moon_grav_y  = self._moon_y
        _moon_grav_on = self.moon_grav_on

        if self.pending_maneuver is not None:
            dvx, dvy = self.pending_maneuver
            self.sat_vx += dvx; self.sat_vy += dvy
            self.pending_maneuver = None
            self.maneuver_flash   = 12
            self.maneuver_times.append(self.sim_time)
            self._new_orbit_segment()
            self._update_ghost()

        delta_total = dt * spf

        # ===== PERBAIKAN: saat launch, paksa dt_sub kecil (≤ 1 detik) =====
        if self.autopilot_launch_active:
            # Maksimum langkah integrasi 1 detik untuk akurasi
            max_dt = 1.0
            n_sub = max(1, int(math.ceil(delta_total / max_dt)))
            dt_sub = delta_total / n_sub
            self._current_dt_sub = dt_sub

            for _ in range(n_sub):
                self.sat_x, self.sat_y, self.sat_vx, self.sat_vy = rk4_step_safe(
                    self.sat_x, self.sat_y, self.sat_vx, self.sat_vy, dt_sub)
                r = math.sqrt(self.sat_x**2 + self.sat_y**2)
                if r > R_BUMI * 1.001:
                    self._was_above_surface = True
                if self._was_above_surface and r < R_BUMI * 0.995:
                    self.running = False
                    self.status_var.set("💥 Satelit menabrak bumi! Tekan Reset.")
                    self._draw_all(); return
                self.current_orbit.add(self.sat_x, self.sat_y)
                self.perm_trail_x.append(self.sat_x)
                self.perm_trail_y.append(self.sat_y)
                if len(self.perm_trail_x) > 15000:
                    self.perm_trail_x = self.perm_trail_x[-12000:]
                    self.perm_trail_y = self.perm_trail_y[-12000:]
                self.sim_time += dt_sub

                self._update_astropy_bodies(self.sim_time)
                self.moon_angle = math.atan2(self._moon_y, self._moon_x)
                self.mars_angle = math.atan2(self._mars_y, self._mars_x)

                # Jalankan logika launch setiap sub-step
                self._launch_check()
                # Jika launch selesai, autopilot_launch_active menjadi False,
                # tetapi kita tetap lanjutkan loop sampai selesai (tidak masalah).

            r = math.sqrt(self.sat_x**2 + self.sat_y**2)
            v2 = self.sat_vx**2 + self.sat_vy**2
            v  = math.sqrt(v2)
            E  = 0.5*v2 - G*M/r
            self.energy_history.append((self.sim_time, E, self.current_orbit.color))
            self._energy_rv_history.append((r, v))
            self._energy_dirty = True
            if len(self.energy_history) > 3000:
                self.energy_history = self.energy_history[-3000:]
                self._energy_rv_history = self._energy_rv_history[-3000:]
            self.earth_angle = (self.earth_angle + OMEGA_BUMI * delta_total * 100) % (2*math.pi)

        elif self.autopilot_active or self.return_to_earth_active:
            max_dt_ap = 10.0
            n_sub = max(1, int(math.ceil(delta_total / max_dt_ap)))
            dt_sub = delta_total / n_sub
            self._current_dt_sub = dt_sub

            for _ in range(n_sub):
                self.sat_x, self.sat_y, self.sat_vx, self.sat_vy = rk4_step_safe(
                    self.sat_x, self.sat_y, self.sat_vx, self.sat_vy, dt_sub)
                r = math.sqrt(self.sat_x**2 + self.sat_y**2)
                if r > R_BUMI * 1.001:
                    self._was_above_surface = True
                if self._was_above_surface and r < R_BUMI * 0.995:
                    self.running = False
                    self.status_var.set("💥 Satelit menabrak bumi! Tekan Reset.")
                    self._draw_all(); return
                self.current_orbit.add(self.sat_x, self.sat_y)
                self.perm_trail_x.append(self.sat_x)
                self.perm_trail_y.append(self.sat_y)
                if len(self.perm_trail_x) > 15000:
                    self.perm_trail_x = self.perm_trail_x[-12000:]
                    self.perm_trail_y = self.perm_trail_y[-12000:]
                self.sim_time += dt_sub

                self._update_astropy_bodies(self.sim_time)
                self.moon_angle = math.atan2(self._moon_y, self._moon_x)
                self.mars_angle = math.atan2(self._mars_y, self._mars_x)

                if self.autopilot_active:
                    self._autopilot_check()
                if self.return_to_earth_active:
                    self._return_check()

            r = math.sqrt(self.sat_x**2 + self.sat_y**2)
            v2 = self.sat_vx**2 + self.sat_vy**2
            v  = math.sqrt(v2)
            E  = 0.5*v2 - G*M/r
            self.energy_history.append((self.sim_time, E, self.current_orbit.color))
            self._energy_rv_history.append((r, v))
            self._energy_dirty = True
            if len(self.energy_history) > 3000:
                self.energy_history = self.energy_history[-3000:]
                self._energy_rv_history = self._energy_rv_history[-3000:]
            self.earth_angle = (self.earth_angle + OMEGA_BUMI * delta_total * 100) % (2*math.pi)

        else:
            # Mode normal
            for _ in range(spf):
                self.sat_x, self.sat_y, self.sat_vx, self.sat_vy = rk4_step_safe(
                    self.sat_x, self.sat_y, self.sat_vx, self.sat_vy, dt)
                r = math.sqrt(self.sat_x**2 + self.sat_y**2)
                if r > R_BUMI * 1.001:
                    self._was_above_surface = True
                if self._was_above_surface and r < R_BUMI * 0.995:
                    self.running = False
                    self.status_var.set("💥 Satelit menabrak bumi! Tekan Reset.")
                    self._draw_all(); return
                self.current_orbit.add(self.sat_x, self.sat_y)
                self.perm_trail_x.append(self.sat_x)
                self.perm_trail_y.append(self.sat_y)
                if len(self.perm_trail_x) > 15000:
                    self.perm_trail_x = self.perm_trail_x[-12000:]
                    self.perm_trail_y = self.perm_trail_y[-12000:]
                self.sim_time += dt

            self.earth_angle = (self.earth_angle + OMEGA_BUMI * delta_total * 100) % (2*math.pi)
            self._update_astropy_bodies(self.sim_time)
            self.moon_angle = math.atan2(self._moon_y, self._moon_x)
            self.mars_angle = math.atan2(self._mars_y, self._mars_x)

            r = math.sqrt(self.sat_x**2 + self.sat_y**2)
            v2 = self.sat_vx**2 + self.sat_vy**2
            v  = math.sqrt(v2)
            E  = 0.5*v2 - G*M/r
            self.energy_history.append((self.sim_time, E, self.current_orbit.color))
            self._energy_rv_history.append((r, v))
            self._energy_dirty = True
            if len(self.energy_history) > 3000:
                self.energy_history = self.energy_history[-3000:]
                self._energy_rv_history = self._energy_rv_history[-3000:]

        if (self.autopilot_phase == 'done' and self.autopilot_mode == 'Flyby'
            and not self.flyby_returned and not self.return_to_earth_active):
            r_now = math.sqrt(self.sat_x**2 + self.sat_y**2)
            if r_now < R_BUMI * 1.1 and r_now > R_BUMI * 0.99:
                self.flyby_returned = True
                self.status_var.set("🛰️ Satelit kembali ke Bumi setelah flyby! (Perjalanan pulang selesai)")

        if self.maneuver_flash > 0:
            self.maneuver_flash -= 1

        self._cache_frame += 1
        if self._cache_frame % 5 == 0 or not self._orbit_cache_valid:
            try:
                self._cached_cur_ox, self._cached_cur_oy = predict_orbit_full(
                    self.sat_x, self.sat_y, self.sat_vx, self.sat_vy)
            except Exception:
                self._cached_cur_ox = []; self._cached_cur_oy = []
            if self.show_ghost:
                try:
                    dvx, dvy = self._compute_maneuver_dv()
                    nx = self.sat_vx + dvx; ny = self.sat_vy + dvy
                    self._cached_ghost_x, self._cached_ghost_y = predict_orbit_full(
                        self.sat_x, self.sat_y, nx, ny)
                except Exception:
                    self._cached_ghost_x = []; self._cached_ghost_y = []
            else:
                self._cached_ghost_x = []; self._cached_ghost_y = []
            self._orbit_cache_valid = True

        self.cur_ox = self._cached_cur_ox
        self.cur_oy = self._cached_cur_oy
        self.ghost_x = self._cached_ghost_x
        self.ghost_y = self._cached_ghost_y

        if self.follow_satellite:
            self._center_on_satellite()

        if self._cache_frame % 5 == 0:
            self._draw_all()

        interval = max(10, int(33 / self.steps_per_frame))
        self.after_id = self.root.after(interval, self._sim_step)

    def _center_on_satellite(self):
        c = self.c_orbit
        w = c.winfo_width(); h = c.winfo_height()
        cx, cy, scale = self._get_view()
        sx = cx + self.sat_x * scale
        sy = cy - self.sat_y * scale
        self.pan_cx_off += (w/2 - sx)
        self.pan_cy_off += (h/2 - sy)

    # ---------- MANEUVER ----------
    def _compute_maneuver_dv(self):
        r_now = math.sqrt(self.sat_x**2 + self.sat_y**2)
        r_now = max(r_now, R_BUMI)
        alt_km = self.man_alt_var.get()
        ecc    = min(0.95, max(0.0, self.man_ecc_var.get()))
        r_p = R_BUMI + alt_km * 1000.0
        r_p = max(R_BUMI, min(r_p, r_now * 0.9999))
        if ecc < 1.0:
            a = r_p / (1.0 - ecc)
            v2 = G*M * (2.0/r_now - 1.0/a)
            if v2 <= 0:
                v2 = G*M / r_now
        else:
            a = r_p / (ecc - 1.0)
            v2 = G*M * (2.0/r_now + 1.0/a)
        v_target = math.sqrt(v2)
        v_now = math.sqrt(self.sat_vx**2 + self.sat_vy**2)
        if v_now < 1:
            tx = -self.sat_y / r_now; ty = self.sat_x / r_now
        else:
            tx = self.sat_vx / v_now; ty = self.sat_vy / v_now
        delta_v_scalar = v_target - v_now
        return delta_v_scalar * tx, delta_v_scalar * ty

    def _update_ghost(self):
        if not self.running:
            try:
                self._cached_cur_ox, self._cached_cur_oy = predict_orbit_full(
                    self.sat_x, self.sat_y, self.sat_vx, self.sat_vy)
            except Exception:
                self._cached_cur_ox = []; self._cached_cur_oy = []
            if self.show_ghost:
                try:
                    dvx, dvy = self._compute_maneuver_dv()
                    nx = self.sat_vx + dvx; ny = self.sat_vy + dvy
                    self._cached_ghost_x, self._cached_ghost_y = predict_orbit_full(
                        self.sat_x, self.sat_y, nx, ny)
                except Exception:
                    self._cached_ghost_x = []; self._cached_ghost_y = []
            else:
                self._cached_ghost_x = []; self._cached_ghost_y = []
            self._orbit_cache_valid = True
            self.cur_ox = self._cached_cur_ox
            self.cur_oy = self._cached_cur_oy
            self.ghost_x = self._cached_ghost_x
            self.ghost_y = self._cached_ghost_y

    def _fire_maneuver(self):
        dvx, dvy = self._compute_maneuver_dv()
        dv_mag = math.sqrt(dvx**2 + dvy**2)
        self.pending_maneuver = (dvx, dvy)
        self.status_var.set("🚀 Manuver dijadwalkan! Δv=%.1f m/s" % dv_mag)
        if not self.running:
            self.sat_vx += dvx; self.sat_vy += dvy
            self.pending_maneuver = None
            self.maneuver_flash   = 12
            self.maneuver_times.append(self.sim_time)
            self._new_orbit_segment()
            self._update_ghost()
            self._draw_all()

    # ---------- ZOOM & PAN ----------
    def _get_view(self):
        c  = self.c_orbit
        w  = c.winfo_width()
        h  = c.winfo_height()
        if w < 2: w = 800
        if h < 2: h = 600

        if self.lock_zoom and hasattr(self, '_locked_scale'):
            return self._locked_cx, self._locked_cy, self._locked_scale

        max_r = max(R_BUMI * 2, math.sqrt(self.sat_x**2 + self.sat_y**2))
        base_scale = min(w, h) / 2 / (max_r * 1.3)
        scale = base_scale * self.zoom_level
        cx = w / 2 + self.pan_cx_off
        cy = h / 2 + self.pan_cy_off
        return cx, cy, scale

    def _reset_zoom(self):
        self.zoom_level = 1.0; self.pan_cx_off = 0.0; self.pan_cy_off = 0.0
        self.follow_satellite = False
        if self.lock_zoom:
            self.lock_zoom = False
            self.status_var.set("🔓 Lock zoom dibatalkan saat reset.")

    def _zoom_to_moon(self):
        c = self.c_orbit
        w = c.winfo_width() or 800
        h = c.winfo_height() or 600
        target_r = self.MOON_DIST * 1.25
        base_scale = min(w, h) / 2 / (R_BUMI * 2 * 1.3)
        needed_scale = min(w, h) / 2 / target_r
        self.zoom_level = needed_scale / base_scale
        self.pan_cx_off = 0.0
        self.pan_cy_off = 0.0
        self.follow_satellite = False
        self.status_var.set("🌕 Zoom ke Bulan — orbit Bulan terlihat (%.5fx)" % self.zoom_level)
        if not self.running:
            self._draw_all()

    def _zoom_to_mars(self):
        c = self.c_orbit
        w = c.winfo_width() or 800
        h = c.winfo_height() or 600
        target_r = self.MARS_DIST * 1.25
        base_scale = min(w, h) / 2 / (R_BUMI * 2 * 1.3)
        needed_scale = min(w, h) / 2 / target_r
        self.zoom_level = needed_scale / base_scale
        self.pan_cx_off = 0.0
        self.pan_cy_off = 0.0
        self.follow_satellite = False
        self.status_var.set("🔴 Zoom ke Mars — orbit Mars terlihat (%.7fx)" % self.zoom_level)
        if not self.running:
            self._draw_all()

    def _on_scroll(self, event):
        factor = 1.2 if (event.num == 4 or event.delta > 0) else 1/1.2
        if self.lock_zoom:
            # Zoom terhadap posisi kursor, dalam mode locked
            cx = self._locked_cx
            cy = self._locked_cy
            self._locked_cx    += (event.x - cx) * (1 - factor)
            self._locked_cy    += (event.y - cy) * (1 - factor)
            self._locked_scale  = max(1e-6, self._locked_scale * factor)
        else:
            # Ambil view saat ini untuk hitung anchor
            c  = self.c_orbit
            w  = c.winfo_width() or 800
            h  = c.winfo_height() or 600
            max_r      = max(R_BUMI * 2, math.sqrt(self.sat_x**2 + self.sat_y**2))
            base_scale = min(w, h) / 2 / (max_r * 1.3)
            scale_now  = base_scale * self.zoom_level
            cx = w / 2 + self.pan_cx_off
            cy = h / 2 + self.pan_cy_off
            # Pan offset agar titik di bawah kursor tetap di tempat
            self.pan_cx_off += (event.x - cx) * (1 - factor)
            self.pan_cy_off += (event.y - cy) * (1 - factor)
            self.zoom_level  = max(0.05, min(500.0, self.zoom_level * factor))
        if not self.running:
            self._draw_all()

    def _on_pan_start(self, event):
        if self._sat_px and self._sat_py:
            if math.sqrt((event.x-self._sat_px)**2+(event.y-self._sat_py)**2) <= 18:
                self.follow_satellite = not self.follow_satellite
                self.status_var.set("🎯 Fokus satelit " + ("ON" if self.follow_satellite else "OFF"))
                if not self.running: self._draw_all()
                return
        self.follow_satellite = False
        self._pan_start = (event.x, event.y)

    def _on_pan_move(self, event):
        if self._pan_start is None: return
        dx = event.x - self._pan_start[0]
        dy = event.y - self._pan_start[1]
        if self.lock_zoom:
            self._locked_cx += dx
            self._locked_cy += dy
        else:
            self.pan_cx_off += dx
            self.pan_cy_off += dy
        self._pan_start = (event.x, event.y)
        if not self.running:
            self._draw_all()

    def _on_pan_end(self, event):
        self._pan_start = None

    def _on_zoom_reset(self, event):
        self._reset_zoom()
        if not self.running:
            self._draw_all()

    # ---------- LOCK ZOOM TOGGLE ----------
    def _toggle_lock_zoom(self):
        if not self.lock_zoom:
            # Aktifkan lock: simpan view saat ini ke state locked
            cx, cy, scale = self._get_view()
            self._locked_cx    = cx
            self._locked_cy    = cy
            self._locked_scale = scale
            self.lock_zoom = True
            self.status_var.set("🔒 Zoom terkunci – tampilan tetap.")
        else:
            # Nonaktifkan lock: sinkronkan zoom_level & pan agar tampilan tidak melompat
            c  = self.c_orbit
            w  = c.winfo_width() or 800
            h  = c.winfo_height() or 600
            max_r      = max(R_BUMI * 2, math.sqrt(self.sat_x**2 + self.sat_y**2))
            base_scale = min(w, h) / 2 / (max_r * 1.3)
            # Hitung zoom_level yang menghasilkan scale yang sama dengan locked
            if base_scale > 1e-12:
                self.zoom_level = self._locked_scale / base_scale
            # Pan offset: posisi center world sama
            self.pan_cx_off = self._locked_cx - w / 2
            self.pan_cy_off = self._locked_cy - h / 2
            self.lock_zoom = False
            self.status_var.set("🔓 Zoom tidak terkunci – otomatis kembali.")
        if not self.running:
            self._draw_all()

    # ---------- DRAWING ----------
    def _draw_all(self):
        self._energy_dirty = True
        self._draw_orbit_canvas()
        self._draw_energy()
        self._draw_info()

    def _draw_orbit_canvas(self):
        c = self.c_orbit
        cx, cy, scale = self._get_view()
        W = c.winfo_width()
        H = c.winfo_height()
        if W < 2: W = 800
        if H < 2: H = 600

        c.delete("all")

        for mult in [1, 2, 3, 5, 8]:
            draw_circle_pts(c, cx, cy, scale, R_BUMI*mult, GRAY, width=1, dash=(2,8))
            lx, ly = w2c(R_BUMI*mult*0.72, R_BUMI*mult*0.72, cx, cy, scale)
            if 0 < lx < W and 0 < ly < H:
                c.create_text(lx, ly-6, text="%.0fkm alt" % ((R_BUMI*(mult-1))/1000),
                              fill="#334466", font=("Courier",7))

        draw_circle_pts(c, cx, cy, scale, R_BUMI, COAST, fill=OCEAN, width=2)
        ea = self.earth_angle
        for (cfx,cfy,rfx,rfy) in [(0.30,0.15,0.18,0.28),(-0.20,0.10,0.14,0.22),
                                    (0.50,-0.20,0.12,0.18),(-0.40,-0.15,0.10,0.16),
                                    (0.00,0.40,0.08,0.12)]:
            rcfx = cfx*math.cos(ea) - cfy*math.sin(ea)
            rcfy = cfx*math.sin(ea) + cfy*math.cos(ea)
            lx = rcfx*R_BUMI; ly = rcfy*R_BUMI
            if lx*lx + ly*ly < (R_BUMI*0.85)**2:
                px, py = w2c(lx, ly, cx, cy, scale)
                rx = rfx*R_BUMI*scale; ry = rfy*R_BUMI*scale
                c.create_oval(px-rx, py-ry, px+rx, py+ry, fill=LAND, outline="")

        lp_x, lp_y = w2c(R_BUMI, 0.0, cx, cy, scale)
        lp_r = max(4, int(R_BUMI * scale * 0.012))
        c.create_oval(lp_x - lp_r, lp_y - lp_r, lp_x + lp_r, lp_y + lp_r,
                      fill=ORANGE, outline=YELLOW, width=1)
        if lp_r > 5:
            c.create_text(lp_x + lp_r + 4, lp_y, text="⬆ Launchpad",
                          fill=ORANGE, font=("Courier", 7), anchor='w')

        draw_circle_pts(c, cx, cy, scale, self.MOON_DIST, "#223344", width=1, dash=(4, 8))
        moon_wx = self._moon_x
        moon_wy = self._moon_y
        moon_sx, moon_sy = w2c(moon_wx, moon_wy, cx, cy, scale)
        moon_r_px = max(4, int(self.MOON_R_VIS * scale))
        if -50 < moon_sx < W+50 and -50 < moon_sy < H+50:
            c.create_oval(moon_sx - moon_r_px, moon_sy - moon_r_px,
                          moon_sx + moon_r_px, moon_sy + moon_r_px,
                          fill="#aaaacc", outline="#ddddee", width=1)
            if moon_r_px >= 5:
                for cr_off in [(-0.3, -0.2, 0.18), (0.25, 0.3, 0.12), (-0.1, 0.35, 0.10)]:
                    crx = moon_sx + cr_off[0] * moon_r_px
                    cry = moon_sy + cr_off[1] * moon_r_px
                    crr = max(1, cr_off[2] * moon_r_px)
                    c.create_oval(crx-crr, cry-crr, crx+crr, cry+crr,
                                  fill="#888899", outline="", width=0)
            c.create_text(moon_sx, moon_sy - moon_r_px - 9,
                          text="Bulan (%.0f km)" % (math.sqrt(self._moon_x**2 + self._moon_y**2) / 1000),
                          fill="#aabbdd", font=("Courier", 7))

        draw_circle_pts(c, cx, cy, scale, self.MARS_DIST, "#3a1a0a", width=1, dash=(4, 12))
        mars_wx = self._mars_x
        mars_wy = self._mars_y
        mars_sx, mars_sy = w2c(mars_wx, mars_wy, cx, cy, scale)
        mars_r_px = max(4, int(self.MARS_R_VIS * scale))
        if -50 < mars_sx < W+50 and -50 < mars_sy < H+50:
            c.create_oval(mars_sx - mars_r_px, mars_sy - mars_r_px,
                          mars_sx + mars_r_px, mars_sy + mars_r_px,
                          fill="#bb4422", outline="#ff7744", width=1)
            if mars_r_px >= 5:
                cap_r = max(2, int(mars_r_px * 0.25))
                c.create_oval(mars_sx - cap_r, mars_sy - mars_r_px,
                              mars_sx + cap_r, mars_sy - mars_r_px + cap_r*2,
                              fill="#ffeeee", outline="", width=0)
            c.create_text(mars_sx, mars_sy - mars_r_px - 9,
                          text="Mars (%.2e km)" % (self.MARS_DIST / 1000),
                          fill="#ff9966", font=("Courier", 7))

        if self.autopilot_active or (hasattr(self, 'autopilot_phase') and self.autopilot_phase == 'done'):
            if hasattr(self, 'autopilot_intercept_angle') and self.autopilot_intercept_angle != 0:
                ia = self.autopilot_intercept_angle
                ir = self.autopilot_r_target
                iw_x = ir * math.cos(ia)
                iw_y = ir * math.sin(ia)
                ix, iy = w2c(iw_x, iw_y, cx, cy, scale)
                cr_size = max(8, int(12 / max(self.zoom_level, 0.1)))
                c.create_line(ix - cr_size, iy, ix + cr_size, iy,
                              fill=YELLOW, width=2)
                c.create_line(ix, iy - cr_size, ix, iy + cr_size,
                              fill=YELLOW, width=2)
                c.create_oval(ix - 5, iy - 5, ix + 5, iy + 5,
                              outline=YELLOW, fill="", width=2)
                lbl = "🎯 Bulan" if self.autopilot_target == 'moon' else "🎯 Mars"
                c.create_text(ix + cr_size + 4, iy,
                              text="%s intercept" % lbl,
                              fill=YELLOW, font=("Courier", 7), anchor='w')

        n_perm = len(self.perm_trail_x)
        if n_perm >= 2:
            if n_perm > 5000:
                step = max(1, n_perm // 5000)
                xs = self.perm_trail_x[::step]
                ys = self.perm_trail_y[::step]
            else:
                xs, ys = self.perm_trail_x, self.perm_trail_y
            draw_path_pts(c, xs, ys, cx, cy, scale, "#44aaff", width=1, smooth=False)

        if self.show_orbit_line and len(self.cur_ox) > 2:
            draw_path_pts(c, self.cur_ox, self.cur_oy, cx, cy, scale,
                          self.current_orbit.color, width=1, dash=(3,6), smooth=False)

        if self.show_ghost and len(self.ghost_x) > 2:
            draw_path_pts(c, self.ghost_x, self.ghost_y, cx, cy, scale,
                          "#33bb33", width=2, smooth=False)
            mid = len(self.ghost_x)//3
            gx, gy = w2c(self.ghost_x[mid], self.ghost_y[mid], cx, cy, scale)
            lbl = "preview → e=%.2f" % self.man_ecc_var.get()
            c.create_text(gx, gy-10, text=lbl, fill="#55cc55", font=("Courier",7))

        for orb in self.orbits[-5:]:
            n = len(orb.trail_x)
            if n < 2: continue
            draw_path_pts(c, orb.trail_x, orb.trail_y, cx, cy, scale,
                          orb.color, width=1, smooth=False)
            mid = n // 2
            lx2, ly2 = w2c(orb.trail_x[mid], orb.trail_y[mid], cx, cy, scale)
            if 10 < lx2 < W-10 and 10 < ly2 < H-10:
                c.create_text(lx2, ly2-10, text=orb.label,
                              fill=orb.color, font=("Courier",7))

        for orb in self.orbits[1:]:
            if orb.trail_x:
                mx, my = w2c(orb.trail_x[0], orb.trail_y[0], cx, cy, scale)
                c.create_text(mx, my, text="★", fill=RED, font=("Courier",11))

        el = orbital_elements(self.sat_x, self.sat_y, self.sat_vx, self.sat_vy)
        r_p, alt_p, r_a, alt_a = periapsis_apoapsis(el)
        r_now = math.sqrt(self.sat_x**2 + self.sat_y**2)
        hz = self.sat_x*self.sat_vy - self.sat_y*self.sat_vx
        ex = (self.sat_vy*hz - G*M*self.sat_x/r_now)/(G*M)
        ey = (-self.sat_vx*hz - G*M*self.sat_y/r_now)/(G*M)
        emag = math.sqrt(ex*ex + ey*ey)
        if emag > 1e-6:
            enx = ex/emag; eny = ey/emag
        else:
            enx = self.sat_x/r_now; eny = self.sat_y/r_now

        px_w = enx * r_p;  py_w = eny * r_p
        ppx, ppy = w2c(px_w, py_w, cx, cy, scale)
        c.create_oval(ppx-5, ppy-5, ppx+5, ppy+5, fill="#ff4444", outline=WHITE, width=1)
        c.create_text(ppx+8, ppy, text="◀ Periapsis\n  %.0f km" % alt_p,
                      fill="#ff8888", font=("Courier",7), anchor='w')

        if r_a is not None:
            ax_w = -enx * r_a; ay_w = -eny * r_a
            apx, apy = w2c(ax_w, ay_w, cx, cy, scale)
            c.create_oval(apx-5, apy-5, apx+5, apy+5, fill="#4488ff", outline=WHITE, width=1)
            c.create_text(apx+8, apy, text="◀ Apoapsis\n  %.0f km" % alt_a,
                          fill="#88aaff", font=("Courier",7), anchor='w')

        spx, spy = w2c(self.sat_x, self.sat_y, cx, cy, scale)
        alen = W * 0.07
        v_mag = math.sqrt(self.sat_vx**2 + self.sat_vy**2)
        if v_mag > 0:
            draw_arrow_c(c, spx, spy,
                         spx + self.sat_vx/v_mag*alen,
                         spy - self.sat_vy/v_mag*alen, LIME, width=2)
        gxn = -self.sat_x/r_now; gyn = -self.sat_y/r_now
        draw_arrow_c(c, spx, spy, spx+gxn*alen*0.7, spy-gyn*alen*0.7, YELLOW, width=2)

        tbx, tby = self._get_thrust_vector_unit()
        thrust_color = RED if self._thrust_active else "#884400"
        thrust_width = 3 if self._thrust_active else 1
        draw_arrow_c(c, spx, spy,
                     spx + tbx * alen * 0.9,
                     spy - tby * alen * 0.9,
                     thrust_color, width=thrust_width)
        ang_label = "%.0f°" % self.thrust_angle_deg.get()
        c.create_text(spx + tbx * alen * 1.05,
                      spy - tby * alen * 1.05,
                      text=ang_label, fill=thrust_color,
                      font=("Courier", 6), anchor='c')

        if self.autopilot_active and self.autopilot_phase in ('waiting', 'coast'):
            vx = self.sat_vx; vy = self.sat_vy
            v_mag = math.sqrt(vx*vx + vy*vy)
            if v_mag > 1:
                ux = vx / v_mag
                uy = vy / v_mag
                arrow_len = alen * 0.7
                ex = spx + ux * arrow_len
                ey = spy - uy * arrow_len
                draw_arrow_c(c, spx, spy, ex, ey, AP_COLOR, width=3)
                c.create_text(spx + ux * (arrow_len + 12),
                              spy - uy * (arrow_len + 12),
                              text="AP Burn", fill=AP_COLOR,
                              font=("Courier", 7), anchor='c')

        if self.autopilot_launch_active:
            if self.launch_phase == 'vertical':
                lbl = "🚀 Launch: Vertical"
                col = LAUNCH_COLOR
            elif self.launch_phase == 'guidance':
                lbl = "🚀 Launch: Guidance"
                col = LAUNCH_COLOR
            else:
                lbl = "🚀 Launch: Done"
                col = LAUNCH_COLOR
            c.create_text(W//2, H-58, text=lbl, fill=col,
                          font=("Courier", 9, "bold"))
            if self.launch_phase in ('vertical', 'guidance'):
                r_now2 = math.sqrt(self.sat_x**2 + self.sat_y**2)
                if self.launch_phase == 'vertical':
                    dx = self.sat_x / r_now2
                    dy = self.sat_y / r_now2
                else:
                    v_circ_tgt = math.sqrt(G * M / (R_BUMI + self.launch_target_alt*1000))
                    tx = -self.sat_y / r_now2
                    ty =  self.sat_x / r_now2
                    vx_target = v_circ_tgt * tx
                    vy_target = v_circ_tgt * ty
                    dvx = vx_target - self.sat_vx
                    dvy = vy_target - self.sat_vy
                    mag = math.sqrt(dvx*dvx + dvy*dvy)
                    if mag > 1:
                        dx = dvx / mag
                        dy = dvy / mag
                    else:
                        dx = tx
                        dy = ty
                arrow_len = alen * 0.7
                ex = spx + dx * arrow_len
                ey = spy - dy * arrow_len
                draw_arrow_c(c, spx, spy, ex, ey, LAUNCH_COLOR, width=3)

        if self.return_to_earth_active and self.return_phase in ('burn3', 'coast'):
            vx = self.sat_vx; vy = self.sat_vy
            v_mag = math.sqrt(vx*vx + vy*vy)
            if v_mag > 1:
                ux = vx / v_mag
                uy = vy / v_mag
                arrow_len = alen * 0.7
                ex = spx + ux * arrow_len
                ey = spy - uy * arrow_len
                draw_arrow_c(c, spx, spy, ex, ey, RETURN_COLOR, width=3)
                c.create_text(spx + ux * (arrow_len + 12),
                              spy - uy * (arrow_len + 12),
                              text="Return Burn", fill=RETURN_COLOR,
                              font=("Courier", 7), anchor='c')

        self._sat_px = spx; self._sat_py = spy
        flash = self.maneuver_flash
        sat_color = RED if (flash > 0 and flash % 2 == 0) else WHITE
        if self.follow_satellite:
            c.create_oval(spx-14, spy-14, spx+14, spy+14,
                          outline=CYAN, fill="", width=1, dash=(4,3))
        c.create_oval(spx-7, spy-7, spx+7, spy+7,
                      fill=sat_color, outline=CYAN, width=2)

        el = orbital_elements(self.sat_x, self.sat_y, self.sat_vx, self.sat_vy)
        if el['e'] < 0.05:
            otype = "🟢 ORBIT SIRKULAR  (e=%.3f)" % el['e'];  ocol = CYAN
        elif el['e'] < 1.0:
            otype = "🟡 ORBIT ELIPS  (e=%.3f)" % el['e'];     ocol = ORANGE
        else:
            otype = "🔴 LINTASAN HIPERBOLA  (e=%.3f)" % el['e']; ocol = RED
        c.create_text(W//2, 14, text=otype, fill=ocol, font=("Courier",10,'bold'))

        legend = [("─ "+orb.label, orb.color) for orb in self.orbits[-5:]]
        legend += [("→ v (kecepatan)", LIME), ("→ g (gravitasi)", YELLOW),
                   ("→ thrust (arah burn)", RED if self._thrust_active else "#884400")]
        if self.autopilot_active:
            legend.append(("→ AP Burn", AP_COLOR))
        if self.return_to_earth_active:
            legend.append(("→ Return Burn", RETURN_COLOR))
        if self.autopilot_launch_active:
            legend.append(("→ Launch Burn", LAUNCH_COLOR))
        if self.show_ghost: legend.append(("-- preview manuver", "#44aa44"))
        n_perm = len(self.perm_trail_x)
        if n_perm > 0:
            legend.append(("·· jejak abadi (%dk titik)" % (n_perm // 1000 or 1)
                           if n_perm >= 500 else "·· jejak abadi", "#44aaff"))

        if self.flyby_returned:
            legend.append(("🛰️ Kembali ke Bumi (Flyby)", "#88dd88"))

        for i, (txt, col) in enumerate(legend):
            c.create_text(8, 30+i*14, text=txt, fill=col,
                          font=("Courier",7), anchor='w')
        hint = "Scroll=zoom  Drag=geser  Dbl-click=reset zoom  [%.1fx]" % self.zoom_level
        if self.follow_satellite: hint += "  [FOKUS ON]"
        if self.lock_zoom: hint += "  [LOCK 🔒]"
        c.create_text(W//2, H-8, text=hint,
                      fill=CYAN if self.follow_satellite else "#445566",
                      font=("Courier",7))

        # ── Indikator Gravitasi Bulan ──────────────────────────────────
        grav_text = "🌕 Grav Bulan: ON  [0]" if self.moon_grav_on else "🌕 Grav Bulan: OFF [0]"
        grav_col  = LIME if self.moon_grav_on else "#445566"
        c.create_text(W - 8, 28, text=grav_text, fill=grav_col,
                      font=("Courier", 7), anchor='e')

        sim_time_sec = self.sim_time
        days = int(sim_time_sec // 86400)
        hours = int((sim_time_sec % 86400) // 3600)
        minutes = int((sim_time_sec % 3600) // 60)
        seconds = int(sim_time_sec % 60)
        time_str = f"{days}d {hours:02d}:{minutes:02d}:{seconds:02d}"
        c.create_text(W-10, H-8, text=f"⏱ {time_str}", fill=TEXT_COL, font=("Courier",8), anchor='se')

        if self.autopilot_active:
            if self.autopilot_phase == 'waiting':
                ap_label = f"⏳ MENUNGGU FASE {self.autopilot_target.upper()}"
                ap_col = "#ffaa44"
            else:
                ap_label = "🌕 AUTOPILOT → BULAN" if self.autopilot_target == 'moon' else "🔴 AUTOPILOT → MARS"
                ap_col = "#aaccff" if self.autopilot_target == 'moon' else "#ff7744"
            mode_text = f" [{self.autopilot_mode}]"
            ap_label += mode_text
            tw = len(ap_label) * 7 + 16
            c.create_rectangle(W//2 - tw//2, H-32, W//2 + tw//2, H-18,
                                fill="#0a0a20", outline=ap_col, width=1)
            c.create_text(W//2, H-25, text=ap_label, fill=ap_col,
                          font=("Courier", 9, "bold"))
            if self.autopilot_target == 'moon':
                tx_w = self._moon_x
                ty_w = self._moon_y
            else:
                tx_w = self._mars_x
                ty_w = self._mars_y
            tpx, tpy = w2c(tx_w, ty_w, cx, cy, scale)
            if -200 < tpx < W+200 and -200 < tpy < H+200:
                c.create_line(spx, spy, tpx, tpy,
                              fill=ap_col, width=1, dash=(4, 6))

        if self.return_to_earth_active:
            if self.return_phase == 'burn3':
                ret_label = "⏳ Burn3: Menuju Bumi"
                ret_col = RETURN_COLOR
            elif self.return_phase == 'coast':
                ret_label = "⏳ Coast: Menuju Bumi"
                ret_col = RETURN_COLOR
            elif self.return_phase == 'burn4':
                ret_label = "⏳ Burn4: Sirkularisasi"
                ret_col = RETURN_COLOR
            elif self.return_phase == 'done':
                ret_label = "✅ Orbit Bumi 4000 km – pendaratan manual"
                ret_col = "#44ff88"
            else:
                ret_label = "🔄 Kembali ke Bumi"
                ret_col = RETURN_COLOR
            tw = len(ret_label) * 7 + 16
            c.create_rectangle(W//2 - tw//2, H-45, W//2 + tw//2, H-31,
                                fill="#0a0a20", outline=ret_col, width=1)
            c.create_text(W//2, H-38, text=ret_label, fill=ret_col,
                          font=("Courier", 9, "bold"))

        if self.flyby_returned and not self.return_to_earth_active:
            c.create_text(W//2, H-45, text="🛰️ Satelit telah kembali ke Bumi setelah flyby!",
                          fill="#88dd88", font=("Courier", 10, "bold"))

    def _draw_energy(self):
        if not self._energy_dirty:
            return
        self._energy_dirty = False
        c = self.c_energy; c.delete("all")
        W = c.winfo_width()
        H = c.winfo_height()
        if W < 10: W = 300
        if H < 10: H = 200
        pad = min(52, W // 6, H // 5)
        c.create_text(W//2, 10, text="Energi Mekanik Spesifik (J/kg)",
                      fill=WHITE, font=("Courier",9,'bold'))
        if len(self.energy_history) < 2:
            c.create_text(W//2, H//2, text="Belum ada data – jalankan simulasi",
                          fill=GRAY, font=("Courier",9))
            return
        step  = max(1, len(self.energy_history) // 600)
        pts   = self.energy_history[::step]
        tvals = [p[0] / 60 for p in pts]
        evals = [p[1]       for p in pts]
        cols  = [p[2]       for p in pts]
        evals_np = np.array(evals, dtype=np.float64)
        smoothed = evals_np.copy()
        win = max(1, len(evals) // 120)
        i = 0
        while i < len(evals):
            j = i + 1
            while j < len(evals) and cols[j] == cols[i]:
                j += 1
            seg = evals_np[i:j]
            if len(seg) > 1:
                kernel = np.ones(min(win*2+1, len(seg))) / min(win*2+1, len(seg))
                smoothed[i:j] = np.convolve(seg, kernel, mode='same')
            i = j
        evals = smoothed.tolist()
        tmin, tmax = min(tvals), max(tvals)
        if tmax == tmin: tmax += 1

        evals_sorted = sorted(evals)
        n_e = len(evals_sorted)
        p2  = evals_sorted[max(0, int(n_e * 0.02))]
        p98 = evals_sorted[min(n_e - 1, int(n_e * 0.98))]
        emin = p2
        emax = p98

        span = emax - emin
        min_span = max(abs(emin) * 5e-4, 1e3) if emin != 0 else 1e4
        if span < min_span:
            mid_e = (emax + emin) / 2
            emin  = mid_e - min_span / 2
            emax  = mid_e + min_span / 2
            span  = min_span
        else:
            pad_e = span * 0.12
            emin -= pad_e; emax += pad_e
            span  = emax - emin

        title_h = 22
        bottom_h = 18
        plot_h = H - pad - title_h - bottom_h
        plot_w = W - 2 * pad
        if plot_h < 10 or plot_w < 10:
            return
        y_top = title_h
        y_bot = title_h + plot_h

        def tp(t, e):
            px = pad + (t - tmin) / (tmax - tmin) * plot_w
            py = y_bot - (e - emin) / span * plot_h
            return px, max(y_top - 4, min(y_bot + 4, py))

        c.create_line(pad, y_top, pad, y_bot, fill=GRAY)
        c.create_line(pad, y_bot, W - pad, y_bot, fill=GRAY)
        c.create_text(W // 2, H - 5, text="Waktu (menit)", fill=TEXT_COL, font=("Courier",7))

        if emin < 0 < emax:
            _, py_zero = tp(tmin, 0.0)
            c.create_line(pad, py_zero, W - pad, py_zero,
                          fill="#556677", width=1, dash=(4, 4))
            c.create_text(pad - 4, py_zero, text="0",
                          fill="#556677", font=("Courier",6), anchor='e')

        for i in range(5):
            t = tmin + i * (tmax - tmin) / 4
            e = emin + i * span / 4
            px, _ = tp(t, emin)
            _, py  = tp(tmin, e)
            c.create_text(px, y_bot + 8, text="%.0f" % t,
                          fill=TEXT_COL, font=("Courier",6))
            c.create_text(pad - 4, py, text="%.2e" % e,
                          fill=TEXT_COL, font=("Courier",6), anchor='e')

        prev = None
        for col, t, e in zip(cols, tvals, evals):
            px, py = tp(t, e)
            if prev is not None:
                if prev[2] == col:
                    c.create_line(prev[0], prev[1], px, py, fill=col, width=1)
            prev = (px, py, col)

        for t_man_s in self.maneuver_times:
            t_man = t_man_s / 60.0
            if tmin <= t_man <= tmax:
                px, _ = tp(t_man, emin)
                if pad < px < W - pad:
                    c.create_line(px, y_top, px, y_bot, fill=RED, dash=(3, 3))
                    c.create_text(px + 2, y_top + 4, text="Δv",
                                  fill=RED, font=("Courier",7))

    def _draw_info(self):
        c = self.c_info; c.delete("all")
        W = c.winfo_width()
        H = c.winfo_height()
        if W < 2: W = 300
        if H < 2: H = 300
        el = orbital_elements(self.sat_x, self.sat_y, self.sat_vx, self.sat_vy)
        r_p, alt_p, r_a, alt_a = periapsis_apoapsis(el)
        try:
            dvx, dvy = self._compute_maneuver_dv()
            dv_mag   = math.sqrt(dvx**2+dvy**2)
            el_new   = orbital_elements(self.sat_x, self.sat_y,
                                        self.sat_vx+dvx, self.sat_vy+dvy)
            r_p2, alt_p2, r_a2, alt_a2 = periapsis_apoapsis(el_new)
            man_ok   = True
        except Exception:
            dv_mag = 0; el_new = el
            r_p2, alt_p2, r_a2, alt_a2 = r_p, alt_p, r_a, alt_a
            man_ok = False
        c.create_rectangle(2, 2, W-2, H-2, fill="#0a0a18", outline=GRAY)
        def classify(e):
            if e < 0.05: return "Sirkular", CYAN
            if e < 1.0:  return "Elips (e=%.3f)" % e, ORANGE
            return "Hiperbola", RED
        cur_type, cur_col = classify(el['e'])
        new_type, new_col = classify(el_new['e']) if man_ok else (cur_type, cur_col)
        divx = W // 2
        c.create_rectangle(4, 3, divx-2, 17, fill="#0d1a0d", outline="")
        c.create_text(8, 10, text="ORBIT SAAT INI", fill=cur_col,
                      font=("Courier",8,'bold'), anchor='w')
        c.create_rectangle(divx+2, 3, W-4, 17, fill="#1a0d00", outline="")
        c.create_text(divx+6, 10, text="SETELAH MANUVER", fill=ORANGE,
                      font=("Courier",8,'bold'), anchor='w')
        c.create_line(divx, 3, divx, H-50, fill=GRAY, width=1, dash=(2,4))
        apo_str  = "%.0f km" % alt_a  if alt_a  is not None else "—"
        apo_str2 = "%.0f km" % alt_a2 if alt_a2 is not None else "—"
        lrows = [
            ("Tipe",         cur_type,                                         cur_col),
            ("Altitude",     "%.1f km"   % el['alt'],                          TEXT_COL),
            ("Kecepatan",    "%.3f km/s" % (el['v']/1e3),                      TEXT_COL),
            ("Eksentrisitas","%.4f"      % el['e'],                             cur_col),
            ("Periapsis ▼",  "%.0f km"   % alt_p,                              "#ff8888"),
            ("Apoapsis ▲",   apo_str,                                           "#88aaff"),
            ("Semi-major",   "%.1f km"   % (el['a']/1e3 if abs(el['a'])<1e10 else 0), TEXT_COL),
            ("Periode",      "%.2f mnt"  % (el['T']/60),                       TEXT_COL),
            ("Energi",       "%.3e J/kg" % el['E'],                            TEXT_COL),
            ("─────────────", "─────────────",                                  GRAY),
            ("M Satelit",    "%.0f kg"   % M_SATELIT,                          "#aaaaff"),
            ("M Bulan",      "7.342×10²² kg",                                  "#aabbdd"),
            ("Grav Bulan",   "ON 🌕" if self.moon_grav_on else "OFF",
                             LIME if self.moon_grav_on else "#556677"),
        ]
        rrows = [
            ("Tipe baru",     new_type,                                         new_col),
            ("Alt. periapsis","%.0f km"  % self.man_alt_var.get(),              ORANGE),
            ("Target e",      "%.2f"     % self.man_ecc_var.get(),              ORANGE),
            ("Δv diperlukan", "%.2f m/s" % dv_mag,                              YELLOW),
            ("Periapsis ▼",   "%.0f km"  % alt_p2,                             "#ff8888"),
            ("Apoapsis ▲",    apo_str2,                                          "#88aaff"),
            ("e baru",        "%.4f"     % el_new['e'],                         new_col),
            ("Periode baru",  "%.2f mnt" % (el_new['T']/60),                   TEXT_COL),
            ("Thrust Δv/s",   "%.0f m/s" % self.thrust_dv_rate.get(),          "#ff9900"),
            ("Thrust arah",   "%.0f°" % self.thrust_angle_deg.get() + " " + self._angle_label_var.get().split("[")[-1].rstrip("]"), "#ff9900"),
            ("Total Δv",      "%.1f m/s" % self.total_dv_applied,              RED if self._thrust_active else YELLOW),
        ]
        hist_h  = 50
        n_rows  = max(len(lrows), len(rrows))
        avail   = H - 20 - hist_h
        dy      = max(12, min(15, avail // n_rows))
        y0      = 22
        for i, (lbl, val, col) in enumerate(lrows):
            y = y0 + i*dy
            c.create_text(8,       y, text=lbl+":", fill="#556677", font=("Courier",7), anchor='w')
            c.create_text(divx-4,  y, text=val,     fill=col,       font=("Courier",8), anchor='e')
        for i, (lbl, val, col) in enumerate(rrows):
            y = y0 + i*dy
            c.create_text(divx+4,  y, text=lbl+":", fill="#556677", font=("Courier",7), anchor='w')
            c.create_text(W-4,     y, text=val,     fill=col,       font=("Courier",8), anchor='e')
        sep_y = H - hist_h
        c.create_line(4, sep_y, W-4, sep_y, fill=GRAY, dash=(2,6))
        c.create_text(8, sep_y+8, text="RIWAYAT:", fill=WHITE,
                      font=("Courier",7,'bold'), anchor='w')
        pill_h  = 14
        pill_y  = sep_y + 20
        x_pill  = 8
        for orb in self.orbits:
            mark = "★" if orb is self.current_orbit else "·"
            if len(orb.trail_x) > 4:
                mid = len(orb.trail_x) // 2
                try:
                    dx2 = orb.trail_x[mid+1] - orb.trail_x[mid-1]
                    dy2 = orb.trail_y[mid+1] - orb.trail_y[mid-1]
                    avx = dx2 / (2 * self.sim_dt)
                    avy = dy2 / (2 * self.sim_dt)
                    el_o = orbital_elements(orb.trail_x[mid], orb.trail_y[mid], avx, avy)
                    ot, _ = classify(el_o['e'])
                    txt = "%s%s e=%.2f" % (mark, orb.label, el_o['e'])
                except Exception:
                    txt = "%s%s" % (mark, orb.label)
            else:
                txt = "%s%s" % (mark, orb.label)
            pill_w = min(W - 12, len(txt) * 5 + 10)
            if x_pill + pill_w > W - 4:
                x_pill  = 8
                pill_y += pill_h + 3
            if pill_y + pill_h > H - 2:
                break
            c.create_rectangle(x_pill, pill_y,
                                x_pill + pill_w, pill_y + pill_h,
                                fill="#0d0d20", outline=orb.color)
            c.create_text(x_pill + 4, pill_y + pill_h // 2,
                          text=txt, fill=orb.color,
                          font=("Courier",7), anchor='w')
            x_pill += pill_w + 4

    # ---------- THRUSTER ----------
    def _get_thrust_vector_unit(self):
        r_now = math.sqrt(self.sat_x**2 + self.sat_y**2)
        v_now = math.sqrt(self.sat_vx**2 + self.sat_vy**2)
        if v_now > 0.1:
            px = self.sat_vx / v_now
            py = self.sat_vy / v_now
        else:
            px = -self.sat_y / r_now
            py =  self.sat_x / r_now
        nx = -py
        ny =  px
        ang = math.radians(self.thrust_angle_deg.get())
        bx = math.cos(ang) * px + math.sin(ang) * nx
        by = math.cos(ang) * py + math.sin(ang) * ny
        return bx, by

    def _on_burn_press(self, event=None):
        if self._thrust_active:
            return
        self._thrust_active = True
        self._burn_btn.config(bg="#550000", relief=tk.SUNKEN)
        self._new_orbit_segment()
        self._locked_thrust_bx, self._locked_thrust_by = self._get_thrust_vector_unit()
        self._do_thrust_tick()

    def _on_burn_release(self, event=None):
        self._thrust_active = False
        self._locked_thrust_bx = None
        self._locked_thrust_by = None
        self._burn_btn.config(bg="#220000", relief=tk.RAISED)
        if self._thrust_after is not None:
            self.root.after_cancel(self._thrust_after)
            self._thrust_after = None
        self._update_ghost()
        if not self.running:
            self._draw_all()

    def _do_thrust_tick(self):
        if not self._thrust_active:
            return
        tick_ms   = 50
        tick_s    = tick_ms / 1000.0
        dv_per_tick = self.thrust_dv_rate.get() * tick_s
        bx = self._locked_thrust_bx
        by = self._locked_thrust_by
        self.sat_vx += dv_per_tick * bx
        self.sat_vy += dv_per_tick * by
        self.total_dv_applied += dv_per_tick
        self.maneuver_flash = 6
        self.total_dv_var.set("Total Δv: %.1f m/s" % self.total_dv_applied)
        self.status_var.set(
            "🔥 BURNING! Δv/s=%.0f m/s | Arah=%.0f° [TERKUNCI] | Total Δv=%.1f m/s" % (
                self.thrust_dv_rate.get(),
                self.thrust_angle_deg.get(),
                self.total_dv_applied))
        self._update_ghost()
        if not self.running:
            self._draw_all()
        self._thrust_after = self.root.after(tick_ms, self._do_thrust_tick)

    # ================================================================
    # AUTOPILOT HOHMANN + PHASING
    # ================================================================
    def _autopilot_to_moon(self):
        self._start_autopilot('moon', self.MOON_DIST, self.MOON_PERIOD)

    def _autopilot_to_mars(self):
        self._start_autopilot('mars', self.MARS_DIST, self.MARS_PERIOD)

    def _start_autopilot(self, target_name, r_target, target_period):
        r_now = math.sqrt(self.sat_x**2 + self.sat_y**2)
        if r_now < R_BUMI * 1.01:
            self.status_var.set("⚠ Satelit belum di orbit! Naikan dulu ke orbit sebelum autopilot.")
            return

        a_transfer_est = (r_now + r_target) / 2.0
        T_transfer_est = math.pi * math.sqrt(a_transfer_est**3 / (G * M))

        self.autopilot_active        = True
        self.autopilot_target        = target_name
        self.autopilot_phase         = 'waiting'
        self.autopilot_r_target      = r_target
        self.autopilot_a_transfer    = a_transfer_est
        self.autopilot_transfer_time = T_transfer_est
        self.autopilot_delta_theta_req = 0.0
        self.autopilot_burn1_applied = False
        self.autopilot_burn2_done    = False
        self.autopilot_wait_start_time = self.sim_time
        self.autopilot_intercept_angle = 0.0
        self._ap_prev_vr    = 0.0
        self._ap_frame_cnt  = 0
        self._ap_phase_cache_time = -1e9
        self._ap_phase_cache_req  = 0.0
        self._ap_prev_err   = None
        self._ap_err_history = []
        self._ap_r_target_dynamic = r_target
        self.flyby_returned = False

        label = "Bulan" if target_name == 'moon' else "Mars"
        self.status_var.set(
            f"🚀 AUTOPILOT {label} aktif – menghitung fase optimal ... "
            f"T_transfer≈{T_transfer_est/3600:.1f} jam. Naikkan dt untuk mempercepat.")

        if self.mode_btn:
            self.mode_btn.config(state=tk.NORMAL, text=f"🔄 Mode: {self.autopilot_mode}")

        if not self.running:
            self.running = True
            self._sim_step()

    def _predict_intercept_error(self):
        r_now = math.sqrt(self.sat_x**2 + self.sat_y**2)
        r_target_guess = self.autopilot_r_target
        if r_now <= R_BUMI or r_now >= r_target_guess * 2.0:
            return float('inf')

        t_arrive = self.sim_time
        ix, iy, ir = self._astropy_body_xy(self.autopilot_target, self.sim_time)

        for _ in range(2):
            r_target_dyn = ir
            a_tr = (r_now + r_target_dyn) / 2.0
            if a_tr <= 0:
                return float('inf')
            T_tr = math.pi * math.sqrt(a_tr**3 / (G * M))
            t_arrive = self.sim_time + T_tr
            ix, iy, ir = self._astropy_body_xy(self.autopilot_target, t_arrive)

        self._ap_r_target_dynamic = ir

        a_tr = (r_now + ir) / 2.0
        if a_tr <= 0:
            return float('inf')
        T_tr = math.pi * math.sqrt(a_tr**3 / (G * M))

        theta_s = math.atan2(self.sat_y, self.sat_x)
        theta_arrive = theta_s + math.pi
        theta_t_arrive = math.atan2(iy, ix)
        err = (theta_arrive - theta_t_arrive + math.pi) % (2*math.pi) - math.pi
        return err

    def _autopilot_check(self):
        if not self.autopilot_active:
            return

        if self.autopilot_phase == 'waiting':
            r_now = math.sqrt(self.sat_x**2 + self.sat_y**2)
            err = self._predict_intercept_error()
            if math.isinf(err):
                self._ap_prev_err = None
                self._ap_frame_cnt += 1
                if self._ap_frame_cnt % 10 == 0:
                    self.status_var.set("⏳ Menunggu orbit valid untuk autopilot ...")
                return

            if not hasattr(self, '_ap_err_history'):
                self._ap_err_history = []
            self._ap_err_history.append(err)
            if len(self._ap_err_history) > 5:
                self._ap_err_history.pop(0)

            self._ap_frame_cnt += 1
            if self._ap_frame_cnt % 10 == 0:
                self.status_var.set(
                    f"⏳ Menunggu fase optimal: err={math.degrees(err):.1f}° "
                    f"(target ~0°)  Δt={self.sim_time - self.autopilot_wait_start_time:.0f} s")

            r_target = getattr(self, '_ap_r_target_dynamic', self.autopilot_r_target)
            a_tr = (r_now + r_target) / 2.0
            T_tr = math.pi * math.sqrt(a_tr**3 / (G * M))
            self.autopilot_a_transfer    = a_tr
            self.autopilot_transfer_time = T_tr
            self.autopilot_delta_theta_req = err

            prev_err = self._ap_prev_err
            self._ap_prev_err = err

            threshold = 0.01
            burn_now = False

            if abs(err) < threshold:
                if all(abs(e) < math.pi/2 for e in self._ap_err_history):
                    burn_now = True
            elif (prev_err is not None
                  and prev_err * err < 0
                  and abs(prev_err) < threshold * 3
                  and abs(err)      < threshold * 3):
                burn_now = True

            if burn_now and not self.autopilot_burn1_applied:
                r_now2 = math.sqrt(self.sat_x**2 + self.sat_y**2)
                v_now  = math.sqrt(self.sat_vx**2 + self.sat_vy**2)

                r_target_val = getattr(self, '_ap_r_target_dynamic', self.autopilot_r_target)
                for _ in range(2):
                    a_trans        = (r_now2 + r_target_val) / 2.0
                    T_trans_actual = math.pi * math.sqrt(a_trans**3 / (G * M))
                    t_arrive_iter  = self.sim_time + T_trans_actual
                    _, _, r_target_val = self._astropy_body_xy(self.autopilot_target, t_arrive_iter)

                a_trans        = (r_now2 + r_target_val) / 2.0
                T_trans_actual = math.pi * math.sqrt(a_trans**3 / (G * M))
                self.autopilot_a_transfer    = a_trans
                self.autopilot_transfer_time = T_trans_actual
                self.autopilot_r_target       = r_target_val

                v_transfer = math.sqrt(G * M * (2.0 / r_now2 - 1.0 / a_trans))
                dv1 = v_transfer - v_now

                if v_now > 1.0:
                    ux = self.sat_vx / v_now
                    uy = self.sat_vy / v_now
                else:
                    ux = -self.sat_y / r_now2
                    uy =  self.sat_x / r_now2

                self.sat_vx += dv1 * ux
                self.sat_vy += dv1 * uy
                self.total_dv_applied += abs(dv1)
                self.total_dv_var.set("Total Δv: %.1f m/s" % self.total_dv_applied)
                self._new_orbit_segment()
                self._update_ghost()

                self.autopilot_phase         = 'coast'
                self.autopilot_burn1_applied = True
                self._ap_prev_err            = None
                self._ap_err_history         = []

                t_intercept = self.sim_time + T_trans_actual
                if self.autopilot_target == 'moon':
                    ix, iy, _ = self._astropy_body_xy('moon', t_intercept)
                else:
                    ix, iy, _ = self._astropy_body_xy('mars', t_intercept)
                self.autopilot_intercept_angle = math.atan2(iy, ix) % (2*math.pi)

                label = "Bulan" if self.autopilot_target == 'moon' else "Mars"
                self.status_var.set(
                    f"🚀 Burn1 selesai! Δv={abs(dv1):.1f} m/s. Menuju {label} "
                    f"— tiba dalam {T_trans_actual/3600:.1f} jam.")
                self.maneuver_times.append(self.sim_time)
                self.maneuver_flash = 12
            return

        if self.autopilot_phase == 'coast':
            r_now   = math.sqrt(self.sat_x**2 + self.sat_y**2)
            r_target = self.autopilot_r_target

            vr = (self.sat_x * self.sat_vx + self.sat_y * self.sat_vy) / r_now
            if self._ap_prev_vr == 0.0:
                self._ap_prev_vr = vr
                return

            prev_vr = self._ap_prev_vr
            self._ap_prev_vr = vr

            if r_now >= r_target * 0.90 and prev_vr > 0 and vr <= 0:
                self._autopilot_burn2()
            elif r_now >= r_target * 1.05:
                self._autopilot_burn2()
            return

    def _autopilot_burn2(self):
        label = "Bulan" if self.autopilot_target == 'moon' else "Mars"

        if self.autopilot_mode == "Flyby":
            self.status_var.set(
                f"🛰️ FLYBY {label} berhasil! Satelit melewati target dan melanjutkan perjalanan menuju Bumi.")
            self.autopilot_active = False
            self.autopilot_phase = 'done'
            self.autopilot_burn2_done = True
            self._ap_prev_vr = 0.0
            self._ap_frame_cnt = 0
            self._ap_prev_err = None
            self._ap_phase_cache_time = -1e9
            if self.mode_btn:
                self.mode_btn.config(state=tk.DISABLED)
            return

        r_now = math.sqrt(self.sat_x**2 + self.sat_y**2)
        v_now = math.sqrt(self.sat_vx**2 + self.sat_vy**2)
        v_circ = math.sqrt(G * M / r_now)
        if v_now > 1.0:
            ux = self.sat_vx / v_now
            uy = self.sat_vy / v_now
        else:
            ux = -self.sat_y / r_now
            uy =  self.sat_x / r_now
        dv2 = v_circ - v_now
        self.sat_vx += dv2 * ux
        self.sat_vy += dv2 * uy
        self.total_dv_applied += abs(dv2)
        self.total_dv_var.set("Total Δv: %.1f m/s" % self.total_dv_applied)
        self._new_orbit_segment()
        self._update_ghost()

        alt_km = (r_now - R_BUMI) / 1000
        self.status_var.set(
            f"✅ RENDEZVOUS {label} selesai! Burn2 Δv={abs(dv2):.1f} m/s → "
            f"Orbit sirkular alt={alt_km:.0f} km (satelit mengorbit bersama {label})")
        self.autopilot_active = False
        self.autopilot_phase = 'done'
        self.autopilot_burn2_done = True
        self._ap_prev_vr = 0.0
        self._ap_frame_cnt = 0
        if self.mode_btn:
            self.mode_btn.config(state=tk.DISABLED)

    def _autopilot_cancel(self):
        if self.autopilot_active:
            self.autopilot_active = False
            self.autopilot_phase = None
            self._ap_prev_vr = 0.0
            self._ap_frame_cnt = 0
            self._ap_prev_err = None
            self._ap_phase_cache_time = -1e9
            self.status_var.set("✖ Autopilot dibatalkan.")
            if self.mode_btn:
                self.mode_btn.config(state=tk.NORMAL)
        else:
            self.status_var.set("Tidak ada autopilot aktif.")

    def _toggle_autopilot_mode(self):
        if self.autopilot_active and self.autopilot_phase in ('waiting', 'coast'):
            if self.autopilot_mode == "Rendezvous":
                self.autopilot_mode = "Flyby"
                self.status_var.set("🔄 Mode diubah ke FLYBY (satelit akan melewati target dan kembali ke Bumi)")
            else:
                self.autopilot_mode = "Rendezvous"
                self.status_var.set("🔄 Mode diubah ke RENDEZVOUS (satelit akan mengorbit target)")
            if self.mode_btn:
                self.mode_btn.config(text=f"🔄 Mode: {self.autopilot_mode}")
        else:
            self.status_var.set("⚠ Ubah mode hanya saat autopilot aktif dan belum mencapai intercept.")

    # ================================================================
    # KEMBALI KE BUMI
    # ================================================================
    def _start_return_to_earth(self):
        if self.return_to_earth_active:
            self.status_var.set("⚠ Proses kembali ke Bumi sudah aktif.")
            return

        # Batalkan autopilot launch yang sedang berjalan
        if self.autopilot_launch_active:
            self.autopilot_launch_active = False
            self.launch_phase = 'done'
            if hasattr(self, '_launch_prev_vr'):
                del self._launch_prev_vr
            if self.launch_btn:
                self.launch_btn.config(state=tk.DISABLED)
            if self.launch_cancel_btn:
                self.launch_cancel_btn.config(state=tk.DISABLED)
            self._was_above_surface = True

        # Batalkan autopilot Hohmann/intercept yang sedang berjalan
        if self.autopilot_active:
            self.autopilot_active = False
            self.autopilot_phase = None
            self._ap_prev_vr = 0.0
            self._ap_frame_cnt = 0
            self._ap_prev_err = None
            self._ap_phase_cache_time = -1e9
            if self.mode_btn:
                self.mode_btn.config(state=tk.NORMAL)

        r_now = math.sqrt(self.sat_x**2 + self.sat_y**2)
        v_now = math.sqrt(self.sat_vx**2 + self.sat_vy**2)

        if r_now < R_BUMI * 1.005:
            self.status_var.set("⚠ Satelit masih di permukaan — jalankan simulasi dulu.")
            return

        # Strategi return:
        # Jika satelit sudah di luar r_burn4 (orbit tinggi/hiperbola):
        #   → burn langsung sekarang di posisi saat ini (r_now = apoapsis baru)
        # Jika satelit masih di dalam r_burn4 (orbit rendah/sedang):
        #   → tunggu apoapsis orbit saat ini, baru burn di sana
        # Burn3 selalu dilakukan di apoapsis agar periapsis tepat = 300 km.

        r_peri_target = R_BUMI + ALT_TARGET_RETURN * 1000  # 4000 km
        r_burn4 = R_BUMI + ALT_TARGET_RETURN * 1000

        self.return_to_earth_active = True
        self.return_burn3_done = False
        self._return_correction_done = False
        if hasattr(self, '_return_prev_vr'):
            del self._return_prev_vr

        el_now = orbital_elements(self.sat_x, self.sat_y, self.sat_vx, self.sat_vy)
        _, alt_peri_now, r_apo_now, alt_apo_now = periapsis_apoapsis(el_now)

        if r_now > r_burn4:
            # Sudah di luar 4000 km — burn retrograde langsung dari posisi saat ini
            # r_now menjadi apoapsis, periapsis diarahkan ke 4000 km
            self.return_phase = 'burn3'
            self._do_burn3_at_apoapsis(r_now, r_peri_target)
        elif r_now >= r_burn4 * 0.95:
            # Sudah sangat dekat 4000 km — langsung sirkularisasi sekarang
            self.return_phase = 'coast'
            self.return_burn3_done = True
            self._return_burn4()
        else:
            # Di orbit rendah — tunggu apoapsis orbit saat ini
            # Jika apoapsis < 4000 km, burn3 akan set periapsis serendah apoapsis bisa
            # (burn4 akan menangani sirkularisasi akhir)
            self.return_phase = 'wait_apoapsis'
            self._return_prev_vr_wait = (self.sat_x*self.sat_vx + self.sat_y*self.sat_vy) / r_now
            alt_apo_str = f"{alt_apo_now:.0f} km" if alt_apo_now is not None else "∞"
            self.status_var.set(
                f"⏳ Menunggu apoapsis ({alt_apo_str}) sebelum Burn3...")

        if not self.running:
            self.running = True
            self._sim_step()

    def _do_burn3_at_apoapsis(self, r_apo, r_peri_target):
        """Mulai burn3 bertahap (gradual) di apoapsis r_apo untuk set periapsis = r_peri_target."""
        rx = self.sat_x / r_apo;  ry = self.sat_y / r_apo
        tx = -ry;                  ty =  rx
        v_rad = self.sat_vx * rx + self.sat_vy * ry
        v_tan = self.sat_vx * tx + self.sat_vy * ty

        v_tan_needed = math.sqrt(G * M * 2.0 * r_peri_target /
                                 (r_apo * (r_apo + r_peri_target)))
        # Hitung total Δv yang dibutuhkan (tidak langsung diaplikasikan)
        dvx_total = (v_tan_needed - v_tan) * tx + (0.0 - v_rad) * rx
        dvy_total = (v_tan_needed - v_tan) * ty + (0.0 - v_rad) * ry
        dv_mag = math.sqrt(dvx_total**2 + dvy_total**2)

        # Simpan sisa Δv untuk diaplikasikan bertahap di _return_check
        self._burn3_dvx_rem = dvx_total
        self._burn3_dvy_rem = dvy_total
        # Thrust rate: selesaikan burn dalam ~60 detik (realistis untuk manuver besar)
        self._burn3_dv_rate = max(10.0, dv_mag / 60.0)

        # Masuk fase burning3 (bukan langsung coast)
        self.return_phase = 'burning3'
        self.maneuver_times.append(self.sim_time)
        self.maneuver_flash = 12
        self._new_orbit_segment()
        self.status_var.set(
            f"🔥 Burn3 dimulai! Total Δv={dv_mag:.0f} m/s @ {self._burn3_dv_rate:.1f} m/s²")

    def _do_radial_correction(self, target_alt_km=200):
        """Sudah tidak dipakai — periapsis diset langsung oleh burn3 baru."""
        pass

    def _return_check(self):
        if not self.return_to_earth_active:
            return

        if self.return_phase == 'wait_apoapsis':
            # Tunggu sampai v_rad berubah dari positif ke negatif (= apoapsis tercapai)
            r_now = math.sqrt(self.sat_x**2 + self.sat_y**2)
            vr = (self.sat_x * self.sat_vx + self.sat_y * self.sat_vy) / r_now
            prev_vr = getattr(self, '_return_prev_vr_wait', vr)
            self._return_prev_vr_wait = vr
            # Apoapsis: vr berubah dari > 0 ke <= 0
            if prev_vr > 0.5 and vr <= 0.5:
                r_burn4 = R_BUMI + ALT_TARGET_RETURN * 1000
                # Jika apoapsis lebih kecil dari target, gunakan periapsis = apoapsis (sirkularisasi)
                r_peri_target = min(r_now * 0.98, r_burn4)
                r_peri_target = max(r_peri_target, R_BUMI + 200e3)
                self.return_phase = 'burn3'
                self._do_burn3_at_apoapsis(r_now, r_peri_target)
            return

        if self.return_phase == 'burning3':
            # Aplikasikan Δv secara bertahap setiap sub-step
            dt_sub = self._current_dt_sub if self._current_dt_sub > 0 else 10.0
            dv_rem = math.sqrt(self._burn3_dvx_rem**2 + self._burn3_dvy_rem**2)
            if dv_rem < 0.1:
                # Burn selesai — pindah ke coast
                self.return_phase = 'coast'
                self.return_burn3_done = True
                el = orbital_elements(self.sat_x, self.sat_y, self.sat_vx, self.sat_vy)
                r_p, alt_p, _, _ = periapsis_apoapsis(el)
                self._update_ghost()
                self.status_var.set(
                    f"✅ Burn3 selesai! Pe={alt_p:.0f} km. Coast menuju Bumi.")
            else:
                # Terapkan sebagian Δv proporsional dengan dt_sub
                frac = min(1.0, self._burn3_dv_rate * dt_sub / dv_rem)
                dvx_now = self._burn3_dvx_rem * frac
                dvy_now = self._burn3_dvy_rem * frac
                self.sat_vx += dvx_now
                self.sat_vy += dvy_now
                dv_now = math.sqrt(dvx_now**2 + dvy_now**2)
                self.total_dv_applied += dv_now
                self.total_dv_var.set("Total Δv: %.1f m/s" % self.total_dv_applied)
                self._burn3_dvx_rem -= dvx_now
                self._burn3_dvy_rem -= dvy_now
                dv_rem_new = math.sqrt(self._burn3_dvx_rem**2 + self._burn3_dvy_rem**2)
                self.status_var.set(
                    f"🔥 Burning... sisa Δv={dv_rem_new:.0f} m/s")
            return

        if self.return_phase == 'coast':
            r_now = math.sqrt(self.sat_x**2 + self.sat_y**2)
            alt_now = (r_now - R_BUMI) / 1000.0
            r_burn4 = R_BUMI + ALT_TARGET_RETURN * 1000   # 4000 km

            vr = (self.sat_x * self.sat_vx + self.sat_y * self.sat_vy) / r_now
            if not hasattr(self, '_return_prev_vr'):
                self._return_prev_vr = vr
                return

            prev_vr = self._return_prev_vr
            self._return_prev_vr = vr

            # Cegah crash: jika periapsis ternyata di bawah permukaan saat sudah
            # dekat Bumi, lakukan koreksi prograde kecil.
            if r_now < R_BUMI * 3.0 and not self._return_correction_done:
                el = orbital_elements(self.sat_x, self.sat_y, self.sat_vx, self.sat_vy)
                r_p, alt_p, _, _ = periapsis_apoapsis(el)
                if alt_p < 150:
                    # Koreksi darurat: naikkan periapsis ke 200 km
                    # Gunakan metode snap v_tan + nol-kan v_rad
                    v_now2 = math.sqrt(self.sat_vx**2 + self.sat_vy**2)
                    r_peri_safe = R_BUMI + 200e3
                    if r_now > r_peri_safe and v_now2 > 1.0:
                        rx2 = self.sat_x/r_now; ry2 = self.sat_y/r_now
                        tx2 = -ry2; ty2 = rx2
                        v_rad2 = self.sat_vx*rx2 + self.sat_vy*ry2
                        v_tan2 = self.sat_vx*tx2 + self.sat_vy*ty2
                        v_t_safe = math.sqrt(G*M*2.0*r_peri_safe /
                                             (r_now*(r_now+r_peri_safe)))
                        dvx2 = (v_t_safe - v_tan2)*tx2 + (0.0 - v_rad2)*rx2
                        dvy2 = (v_t_safe - v_tan2)*ty2 + (0.0 - v_rad2)*ry2
                        dv_m2 = math.sqrt(dvx2**2 + dvy2**2)
                        self.sat_vx += dvx2; self.sat_vy += dvy2
                        self.total_dv_applied += dv_m2
                        self.total_dv_var.set("Total Δv: %.1f m/s" % self.total_dv_applied)
                        self._new_orbit_segment()
                    self._return_correction_done = True
                    self.status_var.set(f"🚨 Koreksi darurat periapsis!")
                    return

            # Trigger burn4: sirkularisasi di periapsis saat mendekati r_burn4
            # vr berubah dari negatif ke >= -0.5 → periapsis tercapai
            at_periapsis = (prev_vr < -0.5 and vr >= -0.5)
            near_target  = (r_now <= r_burn4 * 1.05)
            if near_target and at_periapsis:
                self._return_burn4()

    def _return_burn4(self):
        r_now = math.sqrt(self.sat_x**2 + self.sat_y**2)
        v_now = math.sqrt(self.sat_vx**2 + self.sat_vy**2)
        v_circ = math.sqrt(G * M / r_now)

        # Hitung komponen radial dan tangensial
        rx = self.sat_x / r_now;  ry = self.sat_y / r_now
        # Tangensial: tegak lurus radial, searah orbit (CCW)
        tx = -ry;  ty = rx
        v_tan = self.sat_vx * tx + self.sat_vy * ty
        v_rad = self.sat_vx * rx + self.sat_vy * ry

        # Pastikan arah tangensial sesuai orbit (v_tan bisa negatif jika retrograde)
        sign = 1.0 if v_tan >= 0 else -1.0
        tx *= sign;  ty *= sign
        v_tan_abs = abs(v_tan)

        # Snap: set v_tan = v_circ, nol-kan v_rad
        dvx = (v_circ - v_tan_abs) * tx + (0.0 - v_rad) * rx
        dvy = (v_circ - v_tan_abs) * ty + (0.0 - v_rad) * ry
        dv4 = math.sqrt(dvx**2 + dvy**2)

        self.sat_vx += dvx
        self.sat_vy += dvy
        self.total_dv_applied += dv4
        self.total_dv_var.set("Total Δv: %.1f m/s" % self.total_dv_applied)
        self._new_orbit_segment()
        self._update_ghost()

        alt_km = (r_now - R_BUMI) / 1000
        self.status_var.set(
            f"✅ Orbit Bumi 4000 km tercapai! Burn4 Δv={abs(dv4):.1f} m/s → "
            f"Orbit sirkular alt={alt_km:.0f} km. Gunakan thruster manual untuk pendaratan.")
        self.return_to_earth_active = False
        self.return_phase = 'done'
        if hasattr(self, '_return_prev_vr'):
            del self._return_prev_vr
        self.maneuver_times.append(self.sim_time)
        self.maneuver_flash = 12

    # ---------- TOGGLE ----------
    def _toggle_ghost(self):
        self.show_ghost = not self.show_ghost
        self._update_ghost()
        if not self.running: self._draw_all()
        self.status_var.set("Ghost preview: " + ("ON" if self.show_ghost else "OFF"))

    def _toggle_orbit_line(self):
        self.show_orbit_line = not self.show_orbit_line
        if not self.running: self._draw_all()
        self.status_var.set("Garis orbit: " + ("ON" if self.show_orbit_line else "OFF"))

    # ---------- RESET & HISTORY ----------
    def _reset(self):
        self.running = False
        if self.after_id:
            self.root.after_cancel(self.after_id); self.after_id = None
        self._thrust_active = False
        if self._thrust_after is not None:
            self.root.after_cancel(self._thrust_after)
            self._thrust_after = None
        self.autopilot_launch_active = False
        self.launch_phase = None
        self.launch_alt_peak = 0.0
        self.launch_time_since_peak = 0.0
        self._current_dt_sub = 0.0
        self._apply_initial_orbit()
        self.orbits = []; self.orbit_idx = 0
        self._new_orbit_segment()
        self.energy_history = []; self._energy_rv_history = []; self.sim_time = 0.0
        self.maneuver_times = []
        self.perm_trail_x = []
        self.perm_trail_y = []
        self.earth_angle = 0.0
        self.sim_epoch = Time.now()
        self._astropy_cache_time = -1e9
        self.sim_time = 0.0
        moon_x, moon_y, moon_r = self._astropy_body_xy('moon', 0.0)
        mars_x, mars_y, mars_r = self._astropy_body_xy('mars', 0.0)
        self._moon_x = moon_x; self._moon_y = moon_y
        self._mars_x = mars_x; self._mars_y = mars_y
        self.MOON_DIST  = moon_r
        self.MARS_DIST  = mars_r
        self.moon_angle = math.atan2(moon_y, moon_x)
        self.mars_angle = math.atan2(mars_y, mars_x)
        self.pending_maneuver = None; self.maneuver_flash = 0
        self.total_dv_applied = 0.0
        self.total_dv_var.set("Total Δv: 0 m/s")
        self.autopilot_active = False
        self.autopilot_phase  = None
        self._ap_prev_vr = 0.0
        self._ap_frame_cnt = 0
        self._ap_prev_err = None
        self._ap_phase_cache_time = -1e9
        self._ap_phase_cache_req  = 0.0
        self._ap_err_history = []
        self.autopilot_mode = "Rendezvous"
        self.flyby_returned = False
        self.return_to_earth_active = False
        self.return_phase = None
        self._burn3_dvx_rem = 0.0
        self._burn3_dvy_rem = 0.0
        self._burn3_dv_rate = 50.0
        if self.mode_btn:
            self.mode_btn.config(text="🔄 Mode: Rendezvous", state=tk.NORMAL)
        self.lock_zoom = False
        # gravitasi bulan TIDAK direset — state toggle dipertahankan pengguna
        global _moon_grav_on
        _moon_grav_on = self.moon_grav_on
        self._reset_zoom()
        self._orbit_cache_valid = False
        self._cache_frame = 0
        self._update_ghost()
        self._draw_all()
        self._update_launch_button_state()
        alt = self.init_alt_var.get()
        if alt <= 0:
            self.status_var.set("🚀 Siap. Alt=0 → Jalankan → satelit naik → BURN prograde untuk orbit!")
        else:
            self.status_var.set("Reset. Atur orbit lalu tekan Jalankan.")

    def _clear_history(self):
        cur = self.current_orbit
        self.orbits = [cur]
        self.orbit_idx = 1
        self._draw_all()
        n = len(self.perm_trail_x)
        self.status_var.set("Riwayat orbit dihapus. Jejak abadi tetap tersimpan (%d titik)." % n)

    # ======================================================================
    # JENDELA KEDUA: Grafik Energi Live — KE, PE, TE dari simulasi berjalan
    # ======================================================================
    def _open_energy_example_window(self):
        import matplotlib
        matplotlib.use("TkAgg")
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

        if hasattr(self, '_energy_win') and self._energy_win is not None:
            try:
                self._energy_win.lift()
                return
            except Exception:
                pass

        win = tk.Toplevel(self.root)
        win.title("Grafik Dinamika Energi – Live Simulasi")
        win.configure(bg="#0d0d1a")
        win.geometry("960x580")
        self._energy_win = win

        BG_MPL = "#0d0d1a"
        AX_BG  = "#0a0a16"
        GRID_C = "#223355"
        TEXT_C = "#ccddff"

        fig = Figure(figsize=(9.2, 4.6), dpi=100, facecolor=BG_MPL)
        ax  = fig.add_subplot(111, facecolor=AX_BG)

        line_ke, = ax.plot([], [], color="#44ff44", linewidth=2,
                           label="Energi Kinetik  KE = ½v²")
        line_pe, = ax.plot([], [], color="#ff4444", linewidth=2,
                           label="Energi Potensial  PE = −GM/r")
        line_te, = ax.plot([], [], color="#ffffff", linewidth=1.5,
                           linestyle="--", label="Energi Total  E = KE + PE")

        _maneuver_lines = []

        ax.axhline(y=0, color="#445566", linewidth=0.8, linestyle=":")
        ax.set_title("Gambar 4.4  Dinamika Energi Spesifik Terhadap Waktu (Live)",
                     color=TEXT_C, fontsize=11, pad=8)
        ax.set_xlabel("Waktu Simulasi (menit)", color=TEXT_C, fontsize=10)
        ax.set_ylabel("Energi Spesifik (J/kg)", color=TEXT_C, fontsize=10)
        ax.tick_params(colors=TEXT_C, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#334466")
        ax.grid(True, color=GRID_C, linestyle="--", linewidth=0.5, alpha=0.7)
        legend = ax.legend(facecolor="#0d1a2a", edgecolor="#334466",
                           labelcolor=TEXT_C, fontsize=9, loc="upper left")
        fig.tight_layout(pad=1.4)

        canvas_mpl = FigureCanvasTkAgg(fig, master=win)
        canvas_mpl.draw()
        canvas_mpl.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=6, pady=(6, 0))

        tb_frame = tk.Frame(win, bg=BG_MPL)
        tb_frame.pack(fill=tk.X, padx=6)
        toolbar = NavigationToolbar2Tk(canvas_mpl, tb_frame)
        toolbar.config(bg=BG_MPL)
        toolbar.update()

        info_var = tk.StringVar(value="Jalankan simulasi untuk melihat grafik bergerak…")
        tk.Label(win, textvariable=info_var, bg=BG_MPL, fg="#88aaff",
                 font=("Courier", 9)).pack(pady=(3, 6))

        _after_id = [None]
        _last_len  = [0]

        def _refresh():
            hist = self.energy_history
            n = len(hist)

            if n < 2:
                info_var.set("Belum ada data – jalankan simulasi…")
                _after_id[0] = win.after(300, _refresh)
                return

            step = max(1, n // 1200)
            pts  = hist[::step]

            t_arr = np.array([p[0] / 60.0 for p in pts])
            E_arr = np.array([p[1]         for p in pts])

            if hasattr(self, '_energy_rv_history') and len(self._energy_rv_history) >= n:
                rv_pts = self._energy_rv_history[::step]
                r_arr  = np.array([rv[0] for rv in rv_pts])
                v_arr  = np.array([rv[1] for rv in rv_pts])
                KE_arr = 0.5 * v_arr**2
                PE_arr = -G * M / r_arr
                TE_arr = KE_arr + PE_arr
            else:
                KE_arr = E_arr * 0
                PE_arr = E_arr * 0
                TE_arr = E_arr

            line_ke.set_data(t_arr, KE_arr)
            line_pe.set_data(t_arr, PE_arr)
            line_te.set_data(t_arr, TE_arr)

            for ln in _maneuver_lines:
                try: ln.remove()
                except Exception: pass
            _maneuver_lines.clear()

            t_min = float(t_arr[0]);  t_max = float(t_arr[-1])

            for t_man_s in self.maneuver_times:
                t_man = t_man_s / 60.0
                if t_min <= t_man <= t_max:
                    vl = ax.axvline(x=t_man, color="#ff3333",
                                    linewidth=1, linestyle=":", alpha=0.8)
                    _maneuver_lines.append(vl)

            all_e = np.concatenate([KE_arr, PE_arr, TE_arr])
            valid  = all_e[np.isfinite(all_e)]
            if len(valid) > 4:
                y_lo = float(np.percentile(valid, 2))
                y_hi = float(np.percentile(valid, 98))
                pad_y = (y_hi - y_lo) * 0.12 if y_hi != y_lo else abs(y_hi) * 0.1 + 1e3
                ax.set_xlim(t_min, max(t_max, t_min + 0.1))
                ax.set_ylim(y_lo - pad_y, y_hi + pad_y)

            if len(E_arr) > 0:
                el = orbital_elements(self.sat_x, self.sat_y, self.sat_vx, self.sat_vy)
                r_now = math.sqrt(self.sat_x**2 + self.sat_y**2)
                v_now = math.sqrt(self.sat_vx**2 + self.sat_vy**2)
                ke_now = 0.5 * v_now**2
                pe_now = -G * M / r_now
                info_var.set(
                    f"t={self.sim_time/60:.1f} mnt  │  "
                    f"KE={ke_now:.3e} J/kg  │  "
                    f"PE={pe_now:.3e} J/kg  │  "
                    f"E={ke_now+pe_now:.3e} J/kg  │  "
                    f"alt={el['alt']:.0f} km  │  {n} titik"
                )

            canvas_mpl.draw_idle()
            _last_len[0] = n
            _after_id[0] = win.after(250, _refresh)

        def _on_close():
            import matplotlib.pyplot as plt
            if _after_id[0] is not None:
                try: win.after_cancel(_after_id[0])
                except Exception: pass
            plt.close(fig)
            self._energy_win = None
            win.destroy()
        win.protocol("WM_DELETE_WINDOW", _on_close)

        _after_id[0] = win.after(250, _refresh)

if __name__ == "__main__":
    root = tk.Tk()
    app  = OrbitApp(root)
    root.mainloop()