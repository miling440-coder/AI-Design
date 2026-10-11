# 第二期《内卷与躺平 · 西西弗》v2：点阵线描 + 粒子变形
import sys, subprocess, wave, json
from engine import *

DUR = 160.0
HX = 720; HH = 170; CH = 150
rng0 = np.random.default_rng(7)

# ------------------------------------------------------------ geometry helpers
def lines(*segs): return [np.array(s, np.float32) for s in segs]
def scale_pts(p, cx, cy, s, ox=0, oy=0):
    q = p.copy(); q[:, 0] = (q[:, 0] - ox) * s + cx; q[:, 1] = (q[:, 1] - oy) * s + cy; return q

def run_ph(t, f=2.4, k=0): return 2 * math.pi * f * t + k * 0.9

# Sisyphus slope
SB = np.array([180.0, GY]); SP = np.array([1080.0, 300.0]); SD = SP - SB; SL = float(np.hypot(*SD)); Sd = SD / SL
SN = np.array([-Sd[1], Sd[0]]) * -1  # up-left normal
if SN[1] > 0: SN = -SN
SR = 58.0
def slope_pt(s): return SB + Sd * SL * s if s >= 0 else np.array([SB[0] + Sd[0] * SL * s, GY])
def stone_c(s): return slope_pt(s) + SN * SR
def mountain_polys():
    m = [np.array([[60, GY], [180, GY], SP, [1250, 520], [1420, GY]], np.float32)]
    m.append(np.array([[600, 600], [680, 585]], np.float32)); m.append(np.array([[900, 430], [960, 410]], np.float32))
    m.append(np.array([[1180, 520], [1250, 560]], np.float32))
    return m
def boulder_polys(c, ang, R=SR):
    ps = [circle(c[0], c[1], R)]
    for a0, r0, r1 in ((0.3, 0.2, 0.75), (2.4, 0.1, 0.6), (4.2, 0.35, 0.8)):
        a = ang + a0; ps.append(np.array([[c[0] + R * r0 * math.cos(a), c[1] + R * r0 * math.sin(a)],
                                           [c[0] + R * r1 * math.cos(a + 0.4), c[1] + R * r1 * math.sin(a + 0.4)]], np.float32))
    return ps

def wheel_polys(cx, cy, R, ang, spokes=10):
    ps = [circle(cx, cy, R), circle(cx, cy, R - 16), circle(cx, cy, 12)]
    for i in range(spokes):
        a = ang + i * 2 * math.pi / spokes
        ps.append(np.array([[cx + 12 * math.cos(a), cy + 12 * math.sin(a)], [cx + (R - 16) * math.cos(a), cy + (R - 16) * math.sin(a)]], np.float32))
    base = cy + R + 30
    ps.append(np.array([[cx - R * 0.55, base], [cx, cy], [cx + R * 0.55, base]], np.float32))
    return ps

def heart_poly(cx, cy, s):
    a = np.linspace(0, 2 * math.pi, 160)
    x = 16 * np.sin(a) ** 3; y = -(13 * np.cos(a) - 5 * np.cos(2 * a) - 2 * np.cos(3 * a) - np.cos(4 * a))
    return np.stack([cx + x * s, cy + y * s], 1)

def clock_polys(cx, cy, R, mh, hh):
    ps = [circle(cx, cy, R)]
    for i in range(12):
        a = i * math.pi / 6; r0 = R * 0.84
        ps.append(np.array([[cx + r0 * math.cos(a), cy + r0 * math.sin(a)], [cx + R * 0.94 * math.cos(a), cy + R * 0.94 * math.sin(a)]], np.float32))
    ps.append(np.array([[cx, cy], [cx + R * 0.78 * math.sin(mh), cy - R * 0.78 * math.cos(mh)]], np.float32))
    ps.append(np.array([[cx, cy], [cx + R * 0.5 * math.sin(hh), cy - R * 0.5 * math.cos(hh)]], np.float32))
    return ps

def kite_polys(cx, cy, s=1.0):
    d = np.array([[cx, cy - 46 * s], [cx + 32 * s, cy], [cx, cy + 56 * s], [cx - 32 * s, cy], [cx, cy - 46 * s]], np.float32)
    tail = np.array([[cx + (8 if i % 2 else -8) * s, cy + 56 * s + i * 14 * s] for i in range(7)], np.float32)
    return [d, np.array([[cx, cy - 46 * s], [cx, cy + 56 * s]], np.float32), np.array([[cx - 32 * s, cy], [cx + 32 * s, cy]], np.float32), tail]

def snail_polys(cx, cy):
    a = np.linspace(0, 5.5 * math.pi, 140); r = 4 + a * 2.0
    sp = np.stack([cx + r * np.cos(a), cy - 30 - r * np.sin(a) * 0.95], 1)
    body = np.array([[cx - 42, cy], [cx + 50, cy], [cx + 62, cy - 12], [cx + 58, cy - 26]], np.float32)
    ant = np.array([[cx + 56, cy - 24], [cx + 66, cy - 46]], np.float32)
    return [sp, body, ant]

def book_closed(cx, cy):
    return lines([[cx - 115, cy - 140], [cx + 115, cy - 140], [cx + 115, cy + 140], [cx - 115, cy + 140], [cx - 115, cy - 140]],
                 [[cx - 96, cy - 140], [cx - 96, cy + 140]], [[cx - 60, cy - 90], [cx + 90, cy - 90], [cx + 90, cy - 40], [cx - 60, cy - 40], [cx - 60, cy - 90]])
def book_open(cx, cy):
    ps = lines([[cx, cy - 120], [cx - 250, cy - 140], [cx - 250, cy + 120], [cx, cy + 140], [cx, cy - 120]],
               [[cx, cy - 120], [cx + 250, cy - 140], [cx + 250, cy + 120], [cx, cy + 140]])
    for i in range(7):
        y = cy - 90 + i * 30
        ps += lines([[cx - 220, y - 8], [cx - 40, y + 4]], [[cx + 40, y + 4], [cx + 220, y - 8]])
    return ps

def desk_polys(x):
    T = 640
    return lines([[x - 34, GY], [x - 34, GY - 46], [x + 26, GY - 46]], [[x - 34, GY - 46], [x - 40, GY - 150]],
                 [[x + 40, T], [x + 380, T]], [[x + 60, T], [x + 60, GY]], [[x + 360, T], [x + 360, GY]],
                 [[x + 170, T - 120], [x + 340, T - 120], [x + 340, T - 18], [x + 170, T - 18], [x + 170, T - 120]],
                 [[x + 255, T - 18], [x + 255, T]], [[x + 215, T], [x + 295, T]])

def ropes(t, a, b, amp=7.0):
    xs = np.linspace(0, 1, 70); out = []
    for (p0, p1, ph) in ((a[0], a[1], 0), (b[0], b[1], 1.7)):
        p0 = np.array(p0); p1 = np.array(p1)
        base = p0[None] + (p1 - p0)[None] * xs[:, None]
        off = amp * np.sin(xs * 22 + t * 34 + ph) * np.sin(xs * math.pi)
        base[:, 1] += off; out.append(base.astype(np.float32))
    return out

def stairs_polys(o, x0=150, w=130, h=58, xmax=1300):
    ps = []
    for i in range(-2, 12):
        x = x0 + (i - o) * w; y = GY - (i - o) * h
        ps.append(np.array([[x, y], [x + w, y], [x + w, y - h]], np.float32))
    pts = sample_polys(ps, 4000, 6)[:, :2]
    m = (pts[:, 0] > x0) & (pts[:, 0] < xmax) & (pts[:, 1] <= GY + 1)
    return pts[m]

def esc_offset(t):
    t0, t1 = 33.7, 36.6
    if t < t1: return 0.9 * (t - t0)
    return 0.9 * (t1 - t0) + 2.3 * (t - t1)

def merge(n, *arrs):
    """把多组点（各自带权重的 (m,3) 或 (m,2)）合成 n 个粒子"""
    pts = []
    for a in arrs:
        a = np.asarray(a, np.float32)
        if a.shape[1] == 3:
            k = max(1, int(round(a[:, 2].mean() * len(a)))); a = a[np.linspace(0, len(a) - 1, k).astype(int), :2]
        pts.append(a[:, :2])
    return spread(np.concatenate(pts), n, 0.7)

def polys_n(polys, n, sp=6.0): return sample_polys(polys, n, sp)
def dust_ground(n, seed=3, x0=100, x1=1340):
    r = np.random.default_rng(seed); return spread(np.stack([r.uniform(x0, x1, n), GY - np.abs(r.normal(0, 3, n))], 1).astype(np.float32), n)
def point(n, x, y, s=2.5):
    r = np.random.default_rng(n); return spread(np.stack([x + r.normal(0, s, n), y + r.normal(0, s, n)], 1).astype(np.float32), n)
def burst(prev, r=420, seed=1, cx=None, cy=None):
    g = np.random.default_rng(seed); p = prev.copy()
    c = np.array([cx if cx is not None else p[:, 0].mean(), cy if cy is not None else p[:, 1].mean()])
    v = p[:, :2] - c; v = v / (np.hypot(v[:, 0], v[:, 1])[:, None] + 1e-3)
    p[:, :2] += v * g.uniform(0.3, 1.0, (len(p), 1)) * r + g.normal(0, 40, (len(p), 2)); return p

def hidden(actor):
    return S(lambda n, t: actor.dust.copy())

# ------------------------------------------------------------ actors
HERO = Actor("hero", 520, 1); OBJ = Actor("obj", 1700, 2); OBJ2 = Actor("obj2", 700, 3)
CROWD = Actor("crowd", 3600, 4); TXT = Actor("txt", 2600, 5); STAR = Actor("star", 650, 6)
ACTORS = [STAR, CROWD, OBJ, OBJ2, TXT, HERO]

def crowd_people(n, specs):
    """specs: list of (pose, x, y, h, ph, rot, flip)"""
    per = n // len(specs); out = []
    for i, (pz, x, y, h, ph, rot, fl) in enumerate(specs):
        k = per if i < len(specs) - 1 else n - per * (len(specs) - 1)
        out.append(person(pz, x, y, h, k, ph=ph, rot=rot, flip=fl))
    return np.concatenate(out)

# ---- STAR: faint dust always
sr = np.random.default_rng(11)
STARP = spread(np.stack([sr.uniform(0, W, 650), sr.uniform(20, GY - 60, 650)], 1).astype(np.float32), 650)
STAR.key(0, S(lambda n, t: STARP), 3.0, a=0.16)
STAR.key(81.4, S(lambda n, t: STARP), 2.0, a=0.75)
STAR.key(85.8, S(lambda n, t: STARP), 1.5, a=0.16)
STAR.key(157.0, S(lambda n, t: STARP), 2.5, a=0.0)

# ---- ACT 1 ------------------------------------------------------------
LX = [150, 290, 430, 570, 870, 1010, 1150, 1290]
HERO.key(0.0, A(lambda n, t: person("run", HX, GY, HH, n, ph=run_ph(t))), 1.3, swirl=90)
CROWD.key(0.0, A(lambda n, t: crowd_people(n, [("run", x, GY, CH, run_ph(t, 2.4, i), 0, False) for i, x in enumerate(LX)])), 1.6, a=0.5, swirl=90)
HERO.key(1.9, S(lambda n, t: person("stand", HX, GY, HH, n, rot=math.pi / 2)), 0.55, stag=0.25, swirl=8)
CROWD.key(1.9, S(lambda n, t: crowd_people(n, [("stand", x, GY, CH, 0, math.pi / 2, False) for x in LX])), 0.7, a=0.5, stag=0.5, swirl=8)
HERO.key(3.9, S(lambda n, t: person("stand", HX, GY, HH, n)), 0.8, swirl=30)
L4 = [170, 290, 410, 530]; R4 = [930, 1050, 1170, 1290]
CROWD.key(3.9, A(lambda n, t: crowd_people(n, [("run", x, GY, CH, run_ph(t, 2.6, i), 0, False) for i, x in enumerate(L4)] +
                                             [("stand", x, GY, CH, 0, math.pi / 2, False) for x in R4])), 0.9, a=0.55, swirl=30)
TXT.key(3.9, S(lambda n, t: merge(n, text_shape("卷", 330, 250, 190, 1300, "L", 5), text_shape("躺", 1110, 250, 190, 1300, "L", 5))), 1.0, a=0.32, swirl=120)
HERO.key(7.9, S(lambda n, t: person("tpose", HX, GY, HH, n)), 0.6, stag=0.3, swirl=10)
OBJ.key(7.9, A(lambda n, t: polys_n(ropes(t, ((HX - 64, 590), (520, 610)), ((HX + 64, 590), (980, 702))), n, 6)), 0.9, a=0.9, swirl=40)
# spiral (11.9 - 16.5)
SPW = ["绩效", "KPI", "排名", "考核", "学历", "房贷", "35岁", "年终", "考编", "相亲", "比较", "内推"]
SPP = [text_pts_local(w, "L", 46, 3) for w in SPW]
def spiral_u(t): return cl((t - 11.9) / 4.5)
def spiral_words(n, t):
    u = spiral_u(t); r0 = 470 * (1 - eio(u)) ** 0.9 + 4; om = 1.0 * u + 7.0 * u ** 2
    out = []
    for k, p in enumerate(SPP):
        a = 2 * math.pi * k / 12 + om; r = r0 * (0.72 + 0.28 * ((k * 5) % 3) / 2)
        sc = 0.25 + 0.75 * (r / 470)
        out.append(p * sc + [HX + r * math.cos(a), 500 + r * math.sin(a) * 0.82])
    o = spread(np.concatenate(out), n, 0.5); o[:, 2] *= (1 - 0.9 * u); return o
OBJ.key(11.9, A(spiral_words), 1.0, swirl=80)
BASEP = person("run", 0, 0, CH, 300, ph=0.0)
def spiral_crowd(n, t):
    u = spiral_u(t); r0 = 600 * (1 - eio(u)) + 3; om = 0.6 + 1.0 * u + 7.0 * u ** 2; out = []
    for k in range(8):
        a = 2 * math.pi * k / 8 + om + 0.3; sc = 0.15 + 0.85 * (r0 / 600)
        p = BASEP[:, :2] * sc + [HX + r0 * math.cos(a), 500 + r0 * math.sin(a) * 0.82]; out.append(p)
    o = spread(np.concatenate(out), n, 0.5); o[:, 2] *= (1 - 0.9 * u); return o
CROWD.key(11.9, A(spiral_crowd), 1.2, a=0.45, swirl=60)
HERO.key(12.6, A(lambda n, t: (lambda u: np.column_stack([scale_pts(person("run", 0, 0, HH, n, ph=run_ph(t))[:, :2], HX, 500 + 85 * (1 - u), 1 - 0.95 * u), np.full(n, 0.35 * (1 - 0.93 * u))]))(eio(cl((t - 12.6) / 3.8)))), 1.0, swirl=30)
TXT.key(11.9, hidden(TXT), 1.2, a=0.0)
# title 16.5
for ac in (HERO, OBJ, CROWD):
    ac.key(16.45, S(lambda n, t, prev: burst(prev, 650, 3, HX, 500)), 1.4, a=0.0, stag=0.2, swirl=40)
TXT.key(16.5, S(lambda n, t: text_shape("内卷与躺平", HX, 470, 168, n, "L", 4)), 1.1, a=1.0, stag=0.35, swirl=160)
TXT.key(20.5, S(lambda n, t, prev: burst(prev, 380, 4)), 0.9, a=0.0, stag=0.3)
# wheel 20.7
WC = (HX, GY - 30 - 200); WR = 200
def wheel_ang(t): return -1.6 * (t - 20.7) - (0.0 if t < 23 else 1.2 * (t - 23) ** 1.5)
OBJ.key(20.7, A(lambda n, t: polys_n(wheel_polys(WC[0], WC[1], WR, wheel_ang(t)), n, 6)), 1.2, swirl=120)
HERO.key(20.9, A(lambda n, t: person("run", HX, WC[1] + WR - 16, HH * 0.82, n, ph=run_ph(t, 2.2 if t < 23 else 3.4))), 1.0, swirl=60)
TXT.key(21.0, S(lambda n, t: text_shape("跑", 250, 220, 200, 1200, "L", 5)), 1.0, a=0.3, swirl=100)
WGRID = [(HX + i * 520, GY + j * 650) for j in (-1, 0, 1) for i in range(-3, 4) if not (i == 0 and j == 0)]
def many_wheels(n, t):
    per = n // len(WGRID); out = []
    for k, (x, base) in enumerate(WGRID):
        cy = base - 30 - WR
        w = sample_polys(wheel_polys(x, cy, WR, wheel_ang(t) * (1 + 0.1 * (k % 3))), int(per * 0.62), 9)
        r = person("run", x, cy + WR - 16, HH * 0.82, per - int(per * 0.62), ph=run_ph(t, 3.0, k), spacing=7)
        out += [w, r]
    a = np.concatenate(out); return spread(a[:, :2], n, 0.6) if len(a) != n else a
CROWD.key(24.6, A(many_wheels), 1.4, a=0.55, swirl=40)
TXT.key(24.9, hidden(TXT), 1.0, a=0.0)
# road 27.2
CROWD.key(27.2, S(lambda n, t, prev: burst(prev, 300, 5)), 1.0, a=0.0)
def road(n, t):
    off = (t * 620) % 110; ps = []
    for i in range(16):
        x = -60 + i * 110 - off; ps.append(np.array([[x, GY + 26], [x + 46, GY + 26]], np.float32))
    for i in range(8):
        x = -80 + i * 230 - (t * 380) % 230; ps.append(np.array([[x, GY + 60], [x + 30, GY + 60]], np.float32))
    return polys_n(ps, n, 6)
OBJ.key(27.2, A(road), 1.0, a=0.8, swirl=100)
HERO.key(27.2, A(lambda n, t: person("run", HX, GY, HH, n, ph=run_ph(t, 2.6))), 0.8, swirl=40)
# clock 30.3
def clk(n, t):
    u = t - 30.3; mh = 2 * math.pi * (u * 2.2 + 0.3 * u * u); return polys_n(clock_polys(HX, 330, 120, mh, mh / 12), n, 6)
OBJ.key(30.3, A(clk), 1.0, swirl=120)
# stairs 33.7
def stair_hero_y(t):
    o = esc_offset(t) % 1.0; x0, w, h = 150, 130, 58
    i = math.floor((HX - x0) / w + o); return GY - (i - o) * h
OBJ.key(33.7, A(lambda n, t: spread(stairs_polys(esc_offset(t) % 1.0), n, 0.7)), 1.2, swirl=100)
HERO.key(33.7, A(lambda n, t: person("run", HX + 40, stair_hero_y(t), HH, n, ph=run_ph(t, 2.4 if t < 36.6 else 4.2))), 1.0, swirl=40)
def streaks(n, t):
    g = np.random.default_rng(9); ys = g.uniform(150, 700, 26); xs = g.uniform(0, W, 26); sp = g.uniform(900, 1700, 26); ln = g.uniform(40, 140, 26)
    ps = []
    for y, x, s, l in zip(ys, xs, sp, ln):
        xx = (x - s * t) % (W + 300) - 150; ps.append(np.array([[xx, y], [xx + l, y]], np.float32))
    return polys_n(ps, n, 7)
OBJ2.key(36.6, A(streaks), 0.5, a=0.55, swirl=20)
OBJ2.key(41.3, hidden(OBJ2), 0.8, a=0.0)
# lie flat 41.3
OBJ.key(41.3, S(lambda n, t: dust_ground(n)), 1.1, a=0.35, swirl=60)
HERO.key(41.3, S(lambda n, t: person("stand", HX, GY, HH, n)), 0.7, swirl=20)
TXT.key(41.5, S(lambda n, t: text_shape("躺平", HX, 300, 230, n, "L", 4)), 1.0, a=0.95, swirl=140)
HERO.key(43.4, S(lambda n, t: person("stand", HX, GY, HH, n, rot=math.pi / 2)), 0.4, stag=0.15, swirl=5)
TXT.key(43.4, S(lambda n, t: text_shape("躺平", HX + 330, GY - 70, 130, n, "L", 4, rot=math.pi / 2)), 0.6, a=0.5, stag=0.2, swirl=10)
TXT.key(45.3, S(lambda n, t, prev: burst(prev, 260, 6)), 1.0, a=0.0)
def heart(n, t):
    s = 6.0 * (1 + 0.14 * abs(math.sin(2 * math.pi * 1.9 * t)) ** 6); return polys_n([heart_poly(HX - 10, 540, s)], n, 5)
OBJ.key(45.3, A(heart), 1.0, swirl=100)
def scribble(n, t):
    g = np.random.default_rng(int(t * 12)); ps = []
    for i in range(7):
        a = i / 7 * 2 * math.pi + 0.3; cx, cy = HX - 10 + 165 * math.cos(a), 540 + 120 * math.sin(a)
        ps.append(np.stack([cx + np.arange(6) * 9 - 22, cy + g.uniform(-12, 12, 6)], 1).astype(np.float32))
    return polys_n(ps, n, 5)
OBJ2.key(46.0, A(scribble), 0.6, a=0.8, swirl=40)
OBJ2.key(49.6, hidden(OBJ2), 0.8, a=0.0)

# ---- ACT 2 ------------------------------------------------------------
OBJ.key(49.6, S(lambda n, t, prev: burst(prev, 300, 7)), 1.0, a=0.0)
HERO.key(49.8, S(lambda n, t: person("stand", 960, GY, HH, n, rot=math.pi / 2)), 1.2, swirl=60)
CROWD.key(49.8, A(lambda n, t: person("run", 480, GY, HH, n, ph=run_ph(t, 2.6), spacing=5)), 1.3, a=0.95, swirl=80)
HERO.key(52.0, S(lambda n, t: text_shape("？", 960, 540, 240, n, "L", 4)), 1.1, swirl=120)
CROWD.key(52.0, S(lambda n, t: text_shape("？", 480, 540, 240, n, "L", 4)), 1.1, a=0.95, swirl=120)
OBJ.key(53.8, S(lambda n, t: text_shape("？", HX, 470, 460, n, "L", 4)), 1.2, swirl=150)
HERO.key(53.8, S(lambda n, t: point(n, HX, 470, 60)), 1.0, a=0.0, swirl=80)
CROWD.key(53.8, S(lambda n, t: point(n, HX, 470, 60)), 1.0, a=0.0, swirl=80)
OBJ.key(56.0, S(lambda n, t, prev: burst(prev, 520, 8)), 0.8, a=0.15, stag=0.15)

# Sisyphus 57.8 - 77
SK = [(57.8, 0.03), (68.6, 0.93), (69.3, 0.93), (70.5, 0.02), (71.8, 0.02), (76.0, 0.55), (77.2, 0.55)]
OFF = (SR + 64) / SL
HK = [(57.8, 0.03 - OFF), (68.6, 0.93 - OFF), (70.6, 0.93 - OFF), (72.4, 0.02 - OFF), (72.6, 0.02 - OFF), (76.0, 0.55 - OFF), (77.2, 0.55 - OFF)]
def keyval(keys, t, roll=False):
    if t <= keys[0][0]: return keys[0][1], 0
    for (a, va), (b, vb) in zip(keys, keys[1:]):
        if a <= t <= b:
            u = (t - a) / (b - a)
            if vb < va and roll: e = u ** 2.2
            else: e = float(ease(u)) if vb < va else u
            return va + (vb - va) * e, (1 if vb > va else (-1 if vb < va else 0))
    return keys[-1][1], 0
def sis_stone(n, t, keys=SK, col_t=None):
    s, _ = keyval(keys, t, True); c = stone_c(s)
    return polys_n(boulder_polys(c, s * SL / SR), n, 5)
def sis_hero(n, t, keys=HK, skeys=SK):
    s, d = keyval(keys, t); p = slope_pt(max(s, -0.2)) if s >= 0 else SB + np.array([s * SL * Sd[0], 0])
    ss, sd = keyval(skeys, t)
    if d > 0 and sd > 0: return person("push", p[0], p[1], HH * 0.9, n, ph=s * 90)
    if d < 0: return person("walk", p[0], p[1], HH * 0.9, n, ph=s * 120, flip=True)
    return person("sad" if t > 69.5 else "stand", p[0], p[1], HH * 0.9, n)
OBJ.key(57.8, S(lambda n, t: polys_n(mountain_polys(), n, 6)), 1.6, a=1.0, swirl=120)
OBJ2.key(57.8, A(sis_stone), 1.4, swirl=100)
HERO.key(58.2, A(sis_hero), 1.3, swirl=60)
TXT.key(57.9, S(lambda n, t: text_shape("西西弗", 300, 210, 150, 1800, "L", 4)), 1.2, a=0.42, swirl=120)
TXT.key(66.4, S(lambda n, t, prev: burst(prev, 220, 9)), 1.0, a=0.0)
# 77.2 futile loop: stone loops, ghost echoes on slope
def loop_s(t):
    u = ((t - 77.2) % 1.35) / 1.35
    return 0.06 + 0.84 * (u / 0.7 if u < 0.7 else (1 - ((u - 0.7) / 0.3) ** 2))
OBJ2.key(77.2, A(lambda n, t: polys_n(boulder_polys(stone_c(loop_s(t)), loop_s(t) * SL / SR), n, 5)), 0.6, swirl=30)
HERO.key(77.2, S(lambda n, t: person("sad", *slope_pt(-0.05), HH * 0.9, n)), 0.9, swirl=40)
def ghosts(n, t):
    out = []
    for k, s in enumerate((0.18, 0.34, 0.5, 0.66, 0.82)):
        c = stone_c(s); out.append(sample_polys([circle(c[0], c[1], SR)], 260, 7))
        p = slope_pt(s - OFF); out.append(person("push", p[0], p[1], HH * 0.9, 300, ph=k * 1.3, spacing=7))
    a = np.concatenate(out); return spread(a[:, :2], n, 0.6)
CROWD.key(77.6, S(ghosts), 1.2, a=0.22, swirl=60)
CROWD.key(81.2, S(lambda n, t, prev: burst(prev, 200, 10)), 1.0, a=0.0)
OBJ2.key(81.4, S(lambda n, t: polys_n(boulder_polys(stone_c(0.04), 0.3), n, 5)), 0.9, swirl=40)
HERO.key(81.4, S(lambda n, t: person("lookup", *slope_pt(0.22), HH * 0.9, n)), 1.0, swirl=40)
TXT.key(81.6, S(lambda n, t: text_shape("荒诞", 1100, 230, 170, 1600, "L", 4)), 1.2, a=0.38, swirl=120)
TXT.key(85.6, S(lambda n, t, prev: burst(prev, 220, 11)), 1.0, a=0.0)
# desk 85.8
DX = 470
OBJ.key(85.8, S(lambda n, t: polys_n(desk_polys(DX), n, 6)), 1.5, swirl=120)
OBJ2.key(85.8, hidden(OBJ2), 1.0, a=0.0)
HERO.key(85.8, S(lambda n, t: person("sit", DX, GY, HH, n)), 1.3, swirl=60)
def kpi_stone(n, t):
    u = cl((t - 89.9) / 0.8); y = -150 + (500 + 150) * (u ** 2 if u < 1 else 1)
    if t > 90.7: y = 500 - 18 * abs(math.sin((t - 90.7) * 9)) * math.exp(-(t - 90.7) * 5)
    c = (DX + 20, y)
    a = sample_polys([circle(c[0], c[1], 76)], n // 2, 5)
    b = text_shape("KPI", c[0], c[1] + 4, 52, n - n // 2, "L", 3)
    return np.concatenate([a, b])
OBJ2.key(89.8, A(kpi_stone), 0.4, stag=0.1, swirl=10)
HERO.key(90.7, S(lambda n, t: person("sit", DX, GY, HH, n) if False else person("sad", DX, GY, HH * 0.88, n)), 0.35, stag=0.2, swirl=5)
for ac in (HERO, OBJ, OBJ2, STAR):
    ac.key(92.4, S(lambda n, t, prev: burst(prev, 500, 12, HX, 520)), 0.7, a=0.6 if ac is not STAR else 0.16, stag=0.2, swirl=60)
for ac in (HERO, OBJ, OBJ2):
    ac.key(93.0, S(lambda n, t: point(n, HX, 540, 2.0)), 0.45, a=0.07, stag=0.15, swirl=30)
STAR.key(93.0, S(lambda n, t: STARP), 1.5, a=0.16)

# ---- ACT 3 ------------------------------------------------------------
BX = 820
OBJ.key(93.55, S(lambda n, t: polys_n(book_closed(BX, 470), n, 6)), 1.0, swirl=160)
HERO.key(93.55, S(lambda n, t: person("stand", 420, 640, HH, n)), 1.0, swirl=100)
OBJ2.key(93.55, hidden(OBJ2), 0.8, a=0.0)
OBJ.key(95.6, S(lambda n, t: polys_n(book_open(BX, 480), n, 6)), 0.9, stag=0.5, swirl=60)
TXT.key(97.8, S(lambda n, t: text_shape("西西弗是幸福的。", HX, 420, 96, n, "L", 3)), 1.3, a=1.0, col=WARM, stag=0.4, swirl=120)
OBJ.key(98.4, S(lambda n, t, prev: burst(prev, 260, 13)), 1.2, a=0.0)
HERO.key(98.4, S(lambda n, t: person("lookup", HX, 760, HH, n)), 1.4, swirl=60)
TXT.key(102.6, S(lambda n, t: text_shape("因为他看清了石头。", HX, 420, 92, n, "L", 3)), 1.3, a=1.0, col=WARM, stag=0.45, swirl=120)
TXT.key(107.6, S(lambda n, t, prev: burst(prev, 300, 14)), 1.0, a=0.0)
# mountain again 108.2
OBJ.key(108.0, S(lambda n, t: polys_n(mountain_polys(), n, 6)), 1.5, swirl=120)
OBJ2.key(108.2, S(lambda n, t: polys_n(boulder_polys(stone_c(0.04), 0.3), n, 5)), 1.3, swirl=100)
HERO.key(108.2, S(lambda n, t: person("stand", *slope_pt(0.04 - OFF - 0.02), HH * 0.9, n)), 1.3, swirl=60)
HERO.key(112.5, S(lambda n, t: person("reach", *slope_pt(0.04 - OFF + 0.01), HH * 0.9, n)), 0.7, stag=0.3, swirl=10)
OBJ2.key(114.8, S(lambda n, t: polys_n(boulder_polys(stone_c(0.04), 0.3), n, 5)), 1.0, col=WARM, stag=0.7, swirl=5)
TXT.key(112.6, S(lambda n, t: text_shape("选择", 1120, 230, 170, 1500, "L", 4)), 1.2, a=0.36, col=WARM, swirl=120)
TXT.key(116.5, S(lambda n, t, prev: burst(prev, 220, 15)), 1.0, a=0.0)
SK2 = [(116.7, 0.04), (127.4, 0.9), (129.0, 0.9)]
HK2 = [(116.7, 0.04 - OFF), (127.4, 0.9 - OFF), (129.0, 0.9 - OFF)]
OBJ2.key(116.7, A(lambda n, t: sis_stone(n, t, SK2)), 0.3, col=WARM, swirl=5)
HERO.key(116.7, A(lambda n, t: sis_hero(n, t, HK2, SK2) if t < 127.4 else person("stand", *slope_pt(0.9 - OFF), HH * 0.9, n)), 0.4, swirl=10)
CROWD.key(116.8, A(lambda n, t: np.concatenate([sample_polys(wheel_polys(330, 330, 95, -2.5 * t, 8), n // 2, 6),
                                               person("run", 330, 330 + 95 - 10, 75, n - n // 2, ph=run_ph(t, 3))])), 1.0, a=0.35, swirl=80)
CROWD.key(120.9, S(lambda n, t: np.concatenate([person("stand", 1270, GY, CH, n // 2, rot=math.pi / 2),
                                               sample_polys(boulder_polys((1270, GY - 34 - 46), 0.5, 46), n - n // 2, 6)])), 1.0, a=0.35, swirl=80)
CROWD.key(125.0, S(lambda n, t, prev: burst(prev, 200, 16)), 1.0, a=0.0)
# 129.2 own pace
OBJ.key(129.2, S(lambda n, t: dust_ground(n, 5, -100, 1540)), 1.2, a=0.25, swirl=60)
OBJ2.key(129.2, S(lambda n, t, prev: burst(prev, 160, 17)), 1.0, a=0.0)
HERO.key(129.2, A(lambda n, t: person("walk", HX, GY, HH, n, ph=run_ph(t, 0.9))), 1.0, swirl=60)
def sprinters(n, t):
    sp = []
    for k in range(7):
        x = -200 + ((t - 129.2) * (520 + 60 * k) + k * 260) % 1900
        sp.append(("run", x, GY, CH, run_ph(t, 3.4, k), 0, False))
    return crowd_people(n, sp)
CROWD.key(129.4, A(sprinters), 0.8, a=0.45, swirl=40)
CROWD.key(133.3, S(lambda n, t, prev: burst(prev, 160, 18)), 0.8, a=0.0)
# actions 133.5
ICX = [330, 560, 790, 1020]
def icons(k):
    ps = []
    if k >= 1: ps += clock_polys(ICX[0], GY - 62, 46, 0.0, math.pi)
    if k >= 2: ps += kite_polys(ICX[1], GY - 190, 0.95) + [np.array([[ICX[1], GY - 134 + 98], [ICX[1] + 6, GY]], np.float32)][:0]
    if k >= 3: ps += snail_polys(ICX[2], GY - 2)
    if k >= 4: ps += [heart_poly(ICX[3], GY - 64, 3.0)]
    return ps
HERO.key(133.5, S(lambda n, t: person("stand", 1250, GY, HH, n)), 1.0, swirl=60)
def _icon_fn(k): return lambda n, t: polys_n(icons(k), n, 5)
for i, tt in enumerate((133.5, 136.3, 139.0, 141.7)):
    OBJ.key(tt, S(_icon_fn(i + 1)), 0.9, stag=0.5, swirl=90)
# stone rolls again 144.9
SK3 = [(144.9, 0.93), (146.6, 0.03)]
OBJ.key(144.9, S(lambda n, t: polys_n(mountain_polys(), n, 6)), 1.3, swirl=120)
OBJ2.key(144.9, A(lambda n, t: sis_stone(n, t, SK3)), 0.5, swirl=30)
HERO.key(144.9, S(lambda n, t: person("stand", *slope_pt(-0.11), HH * 0.9, n)), 1.2, swirl=60)
OBJ2.key(147.7, S(lambda n, t: polys_n(boulder_polys(stone_c(0.03), 0.93 * SL / SR - (0.93 - 0.03) * SL / SR), n, 5)), 2.0, col=WARM, stag=0.6, swirl=5)
HERO.key(147.7, S(lambda n, t: person("lookup", *slope_pt(-0.11), HH * 0.9, n)), 1.2, col=WARM, swirl=10)
OBJ.key(148.2, S(lambda n, t: polys_n(mountain_polys(), n, 6)), 2.5, col=WARM, stag=0.7, swirl=0)
for ac in (HERO, OBJ, OBJ2, TXT, CROWD):
    ac.key(157.2, S(lambda n, t: point(n, HX, 540, 2.0)), 1.6, a=0.04, stag=0.3, swirl=50)

for a in ACTORS: a.prep()

# ------------------------------------------------------------ camera / fx timelines
CAM = [(0, 0.1, 1.0, 720, 540), (7.9, 3.6, 1.16, 720, 600), (11.9, 1.4, 1.0, 720, 540), (24.6, 2.2, 0.42, 720, 520),
       (27.2, 1.0, 1.0, 720, 540), (36.6, 0.6, 1.1, 760, 560), (41.3, 0.8, 1.0, 720, 540), (77.2, 2.0, 0.88, 720, 520),
       (81.4, 2.2, 1.0, 720, 540), (89.8, 2.6, 1.1, 640, 560), (92.4, 0.8, 1.0, 720, 540), (116.7, 10.7, 1.1, 820, 480),
       (129.2, 1.2, 1.0, 720, 540), (147.7, 6.0, 0.94, 720, 520)]
def camera(t):
    z, cx, cy = 1.0, 720.0, 540.0
    for (t0, d, z1, x1, y1) in CAM:
        if t < t0: break
        e = float(eio((t - t0) / d)); z, cx, cy = z + (z1 - z) * e, cx + (x1 - cx) * e, cy + (y1 - cy) * e
    for (t0, d, amp) in SHAKE:
        if t0 <= t < t0 + d:
            k = (1 - (t - t0) / d) ** 2 * amp; g = np.random.default_rng(int(t * 60)); cx += g.normal(0, k); cy += g.normal(0, k)
    return z, cx, cy
SHAKE = [(43.4, 0.4, 9), (56.0, 0.55, 14), (71.8, 0.35, 8), (90.7, 0.35, 8), (92.4, 0.6, 6), (146.6, 0.3, 6)]
WAVES = [(16.5, 720, 500, 900, 0), (43.5, 720, GY, 260, 0), (56.0, 720, 470, 1000, 0), (71.8, *stone_c(0.02), 200, 0),
         (93.5, 720, 540, 800, 0), (114.8, *stone_c(0.04), 240, 1), (127.4, *stone_c(0.9), 320, 1), (147.9, 520, GY, 900, 1)]
FLASH = [(16.5, 0.6, 0.75), (56.0, 0.3, 0.35), (93.45, 0.35, 0.55)]
GROUND = [(0.0, 1.2, -200, 1640, 0.8, 0), (11.9, 0.8, 720, 720, 0.0, 0), (20.7, 0.9, -200, 1640, 0.8, 0), (24.6, 2.2, -2600, 4000, 0.8, 0),
          (27.2, 1.0, -200, 1640, 0.8, 0), (33.7, 0.6, -200, 150, 0.8, 0), (41.3, 0.8, -200, 1640, 0.8, 0), (92.4, 0.5, 720, 720, 0, 0),
          (108.2, 1.3, -200, 1640, 0.8, 0), (147.7, 3.0, -200, 1640, 0.9, 1), (157.0, 1.6, 720, 720, 0.0, 1)]
def ground(t):
    x0, x1, a, wm = 720, 720, 0.0, 0.0
    for (t0, d, X0, X1, A_, WM) in GROUND:
        if t < t0: break
        e = float(eio((t - t0) / d)); x0, x1, a, wm = x0 + (X0 - x0) * e, x1 + (X1 - x1) * e, a + (A_ - a) * e, wm + (WM - wm) * e
    return x0, x1, a, wm

SUBS = [(3.9, 5.8, "卷，怕自己白卷。"), (5.8, 7.7, "躺，怕自己被落下。"), (7.9, 11.7, "你被夹在中间，哪边都不敢松手。"),
        (12.3, 16.2, "越转越快，越转越小。"), (20.9, 23.0, "所有人都在〔跑〕，"), (23.0, 24.8, "跑得越来越快。"),
        (24.9, 27.1, "每个人，都困在自己的轮子里。"), (27.2, 30.2, "没人知道，终点在哪。"), (30.3, 33.6, "一年，又一年。"),
        (33.8, 37.4, "你以为在往上爬，"), (37.5, 41.2, "其实只是〔原地加速〕。"), (41.3, 45.2, "于是有人说：躺平吧。"),
        (45.3, 49.6, "可躺下之后，心并没有放松。"), (49.8, 53.7, "因为卷和躺，问的是同一个问题："),
        (53.8, 55.9, "「我这样做，到底值不值？」"), (57.9, 62.0, "两千多年前，有个人也问过。"),
        (62.1, 66.4, "〔西西弗〕被诸神惩罚，推一块巨石上山。"), (66.5, 70.6, "石头到了山顶，就会滚下来。"),
        (70.8, 74.6, "日复一日，永无止境。"), (74.8, 77.0, "诸神以为，这是最可怕的刑罚。"), (77.4, 81.2, "徒劳，且无望。"),
        (81.4, 85.6, "加缪说，这就是「〔荒诞〕」：人渴望意义，世界沉默。"), (85.8, 89.6, "你的工位，就是你的山。"),
        (89.8, 93.0, "你的 KPI，就是你的石头。"), (93.6, 97.6, "但加缪写下了整本书最重要的一句："),
        (108.2, 112.4, "他不再等诸神给答案，不再问「值不值」。"), (112.5, 116.6, "推，是他自己的〔选择〕。这一刻，石头属于他。"),
        (116.8, 120.8, "内卷，是被推着跑。"), (120.9, 125.0, "躺平，是被石头压住。"), (125.1, 129.0, "而你，可以〔选择怎么推〕。"),
        (129.2, 133.4, "不必赢过所有人，也不必假装不在乎。"), (133.6, 136.2, "给努力，设一个〔下班时间〕。"),
        (136.3, 138.9, "做一件不为 KPI 的事。"), (139.0, 141.6, "允许自己〔慢一点〕。"), (141.7, 144.8, "把「值不值」，换成「〔想不想〕」。"),
        (144.9, 147.6, "石头，还会滚下来。"), (147.7, 153.3, "但山，是〔你的〕。"), (153.5, 157.4, "你现在推的石头，是你自己选的吗？")]
HOOK = [(0.25, 1.85, "白天拼命卷，"), (1.9, 3.75, "晚上想躺平。")]

# ------------------------------------------------------------ overlay text
SUBC = (214, 210, 202); GOLD = tuple(int(v) for v in WARM)
_sc = {}
def rich_sprite(text, sz, base=SUBC):
    k = (text, sz, base)
    if k in _sc: return _sc[k]
    f = font("L", sz); parts = []; gold = False; buf = ""
    for ch in text:
        if ch in "〔〕":
            if buf: parts.append((buf, gold)); buf = ""
            gold = ch == "〔"; continue
        buf += ch
    if buf: parts.append((buf, gold))
    d0 = ImageDraw.Draw(Image.new("L", (4, 4))); tw = int(sum(d0.textlength(p, font=f) for p, _ in parts)) + 20
    im = Image.new("RGBA", (tw, int(sz * 1.6)), (0, 0, 0, 0)); d = ImageDraw.Draw(im); x = 10
    for p, g in parts:
        d.text((x, sz * 0.8), p, font=f, fill=(GOLD if g else base) + (255,), anchor="lm"); x += d.textlength(p, font=f)
    a = np.array(im).astype(np.float32) / 255.0; _sc[k] = a; return a

def blit(canvas, spr, cx, cy, alpha):
    if alpha <= 0.003: return
    h, w = spr.shape[:2]; x0 = int(cx - w / 2); y0 = int(cy - h / 2)
    xa, ya = max(0, x0), max(0, y0); xb, yb = min(W, x0 + w), min(H, y0 + h)
    if xa >= xb or ya >= yb: return
    s = spr[ya - y0:yb - y0, xa - x0:xb - x0]; a = s[..., 3:4] * alpha
    canvas[ya:yb, xa:xb] = canvas[ya:yb, xa:xb] * (1 - a) + s[..., :3] * 255 * a

def fade(t, t0, t1, fi=0.35, fo=0.35): return float(min(cl((t - t0) / fi), cl((t1 - t) / fo)))

# ------------------------------------------------------------ static layers
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
VIG = (1 - 0.26 * (((xx - 720) / 720) ** 2 + ((yy - 540) / 540) ** 2) / 2)[..., None]
GR = [np.random.default_rng(100 + i).normal(0, 2.0, (H, W, 1)).astype(np.float32) for i in range(6)]
BG = np.array([7, 7, 9], np.float32)
SPOT = np.clip(1 - np.abs(xx - 720) / (40 + (yy / 760) * 190), 0, 1) * np.clip(1 - yy / 820, 0, 1) ** 0.6
SPOT = cv2.GaussianBlur(SPOT, (0, 0), 14)[..., None]
SUNG = np.exp(-(((xx - 520) / 620) ** 2 + ((yy - GY) / 380) ** 2))[..., None]

def W2S(p, cam):
    z, cx, cy = cam; return (np.asarray(p, np.float32) - [cx, cy]) * z + [720, 540]

def render(t):
    cam = camera(t); z = cam[0]
    Ps, Ws, Cs = [], [], []
    for ac in ACTORS:
        st = ac.state(t)
        if st is None: continue
        P, al, co = st
        b = P[:, 2] * al * ac.br * (0.86 + 0.14 * np.sin(ac.tw + t * 3.1))
        Ps.append(W2S(P[:, :2], cam)); Ws.append(b); Cs.append(co / 255.0)
    P = np.concatenate(Ps); Wt = np.concatenate(Ws) * (0.75 + 0.25 * min(1, z)); C = np.concatenate(Cs)
    img = splat(P, Wt, C)
    vec = np.zeros((H, W, 3), np.uint8)
    x0, x1, ga, wm = ground(t)
    if ga > 0.01:
        a = W2S([[x0, GY], [x1, GY]], cam); colr = (FG * (1 - wm) + WARM * wm) * ga
        cv2.line(vec, tuple(int(v) for v in a[0]), tuple(int(v) for v in a[1]), tuple(int(c) for c in colr), 2, cv2.LINE_AA)
    sh = fade(t, 24.9, 27.4, 1.2, 0.8)
    if sh > 0:
        for yv in (GY - 650, GY + 650):
            a = W2S([[-2600, yv], [4000, yv]], cam)
            cv2.line(vec, tuple(int(v) for v in a[0]), tuple(int(v) for v in a[1]), tuple(int(c * 0.6 * sh) for c in FG), 2, cv2.LINE_AA)
    pg = fade(t, 133.6, 145.2, 0.5, 0.6)
    if pg > 0:
        k = min(4, int((t - 133.5) / 2.7) + 1); xe = 230 + (ICX[min(k, 4) - 1] + 120 - 230) * float(eout((t - 133.5 - (k - 1) * 2.7) / 1.2)) if k >= 1 else 230
        xe = max(xe, 230); a = W2S([[230, GY + 1], [xe, GY + 1]], cam)
        cv2.line(vec, tuple(int(v) for v in a[0]), tuple(int(v) for v in a[1]), tuple(int(c * pg) for c in WARM), 3, cv2.LINE_AA)
    for (t0, wx, wy, R, warm) in WAVES:
        u = (t - t0) / 1.3
        if 0 <= u < 1:
            c = W2S([wx, wy], cam); colr = WARM if warm else FG
            for j, lag in enumerate((0.0, 0.12)):
                uu = u - lag
                if uu <= 0: continue
                r = int(R * float(eout(uu)) * z); al = (1 - uu) ** 1.6 * (0.9 - 0.4 * j)
                cv2.circle(vec, tuple(int(v) for v in c), max(1, r), tuple(int(v * al) for v in colr), 2, cv2.LINE_AA)
    sun = cl((t - 147.7) / 4.0) * cl((159.0 - t) / 1.5)
    if sun > 0:
        cy = GY + 150 - 175 * float(eout((t - 147.7) / 4.5)); sc = W2S([520, cy], cam)
        for i in range(17):
            a = math.pi + i * math.pi / 16; r0, r1 = 175, 175 + 60 + 25 * (i % 2)
            p0 = W2S([520 + r0 * math.cos(a), cy + r0 * math.sin(a)], cam); p1 = W2S([520 + r1 * math.cos(a), cy + r1 * math.sin(a)], cam)
            if p0[1] < W2S([0, GY], cam)[1]:
                cv2.line(vec, tuple(int(v) for v in p0), tuple(int(v) for v in p1), tuple(int(c * sun * 0.8) for c in WARM), 2, cv2.LINE_AA)
    vecf = vec.astype(np.float32)
    lit = img * 1500.0
    core = cv2.GaussianBlur(lit, (0, 0), 0.8)
    small = cv2.resize(core + vecf, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    glow = cv2.resize(cv2.GaussianBlur(small, (0, 0), 3.5), (W, H), interpolation=cv2.INTER_LINEAR)
    litt = core + vecf + glow * 0.9
    canvas = BG + 265.0 * (1 - np.exp(-litt / 265.0))
    # spotlight
    sp = max(fade(t, 97.9, 107.8, 1.0, 1.0), 0.8 * fade(t, 108.4, 116.6, 1.2, 1.2))
    if sp > 0:
        sx = 720 if t < 108 else W2S(slope_pt(0.04), cam)[0] - 20
        M = np.roll(SPOT, int(sx - 720), axis=1) if sx != 720 else SPOT
        canvas += M * np.array([60, 56, 50], np.float32) * sp
    if sun > 0:
        cy = GY + 150 - 175 * float(eout((t - 147.7) / 4.5)); sc = W2S([520, cy], cam); gy = W2S([0, GY], cam)[1]
        canvas += SUNG * WARM * 0.28 * sun
        disk = np.zeros((H, W), np.uint8); cv2.circle(disk, (int(sc[0]), int(sc[1])), int(150 * z), 255, -1, cv2.LINE_AA)
        disk[int(gy):] = 0; d = cv2.GaussianBlur(disk.astype(np.float32) / 255, (0, 0), 2.5)[..., None]
        canvas = canvas * (1 - d * 0.9 * sun) + np.array([255, 226, 170], np.float32) * d * 0.9 * sun
    # myth tone (warm sepia) 57.8 - 76
    mt = fade(t, 57.8, 76.2, 0.9, 1.4) * 0.7
    if mt > 0:
        lum = canvas.mean(2, keepdims=True)
        sep = lum * np.array([1.08, 0.94, 0.72], np.float32) + np.array([7, 5, 2], np.float32)
        canvas = canvas * (1 - mt) + sep * mt
    for (t0, d, s) in FLASH:
        if t0 <= t < t0 + d: canvas = canvas + (255 - canvas) * s * (1 - (t - t0) / d) ** 2
    if 56.0 <= t < 56.55:
        g = 1 - (t - 56.0) / 0.55; rg = np.random.default_rng(int(t * 30))
        canvas[..., 0] = np.roll(canvas[..., 0], int(14 * g), 1); canvas[..., 2] = np.roll(canvas[..., 2], -int(14 * g), 1)
        for _ in range(int(10 * g) + 2):
            y0 = int(rg.integers(200, 800)); hh = int(rg.integers(4, 26)); canvas[y0:y0 + hh] = np.roll(canvas[y0:y0 + hh], int(rg.normal(0, 90 * g)), 1)
    # text
    for (t0, t1, s) in HOOK:
        f = fade(t, t0, t1, 0.25, 0.25)
        if f > 0:
            blit(canvas, rich_sprite(s, 96, (240, 236, 228)), 720, 230 - 10 * (1 - f), f)
    for (t0, t1, s) in SUBS:
        f = fade(t, t0, t1, 0.3, 0.3)
        if f > 0: blit(canvas, rich_sprite(s, 38), 720, 975 + 8 * (1 - f), f)
    f = fade(t, 17.3, 20.4, 0.6, 0.5)
    if f > 0: blit(canvas, rich_sprite("第二期 · 给空心的人", 36, (150, 146, 140)), 720, 640, f)
    f = fade(t, 56.0, 57.7, 0.12, 0.4)
    if f > 0:
        k = float(eout((t - 56.0) / 0.45)); sz = int(150 + 70 * (1 - k))
        blit(canvas, rich_sprite("〔值不值？〕", sz), 720, 470, f)
    f = fade(t, 153.9, 157.4, 0.5, 0.5)
    if f > 0: blit(canvas, rich_sprite("评论区聊聊 · 下期：焦虑与自由", 28, (140, 136, 130)), 720, 1030, f)
    canvas = canvas * VIG + GR[int(t * 6) % 6] * 0.35
    return np.clip(canvas, 0, 255).astype(np.uint8)

# ------------------------------------------------------------ audio
def build_audio(path="sfx.wav"):
    SR_ = 44100; n = int(SR_ * DUR); tt = np.arange(n) / SR_; fx = np.zeros(n); rg = np.random.default_rng(5)
    def tone(s, f, amp, dec):
        m = tt >= s; x = tt[m] - s; fx[m] += amp * np.sin(2 * np.pi * f * x) * np.exp(-x * dec) * np.clip(x / 0.008, 0, 1)
    def whoosh(s, dur, amp, rev=False, fc=900):
        s = max(0.0, s); i0, i1 = int(s * SR_), min(n, int((s + dur) * SR_)); L = i1 - i0
        z_ = rg.normal(0, 1, L); sp = np.fft.rfft(z_); fr = np.fft.rfftfreq(L, 1 / SR_)
        sp *= np.exp(-((fr - fc) / fc) ** 2); z_ = np.fft.irfft(sp, L); z_ /= np.abs(z_).max() + 1e-9
        x = np.linspace(0, 1, L); en = (x ** 2) * np.clip((1 - x) / 0.15, 0, 1) if not rev else ((1 - x) ** 2) * np.clip(x / 0.08, 0, 1)
        fx[i0:i1] += amp * z_ * en
    def shimmer(s, dur, amp):
        for k in range(14): tone(s + k * dur / 14, 1800 + 300 * (k % 5), amp, 9)
    for t0 in (0.0, 3.9, 20.7, 33.7, 49.8, 57.8, 85.8, 93.55, 108.0, 133.5, 144.9):
        whoosh(t0 - 0.2, 1.3, 0.16, fc=1400)
    for t0 in (52.0, 53.8, 97.8, 102.6, 16.5): shimmer(t0, 1.1, 0.02)
    tone(1.95, 70, 0.25, 6); tone(43.45, 55, 0.4, 3); whoosh(43.0, 0.5, 0.15, rev=True, fc=500)
    whoosh(11.9, 4.6, 0.28, fc=700); tone(16.5, 45, 0.55, 1.2); tone(16.5, 440, 0.05, 1.0); whoosh(15.6, 0.9, 0.2)
    whoosh(24.6, 2.0, 0.2, fc=400)
    for i in range(22): tone(30.4 + i * 0.14, 2600, 0.04, 220)
    tone(33.4, 196, 0.1, 2)
    whoosh(36.6, 4.4, 0.18, fc=2500)
    whoosh(55.6, 0.5, 0.2, fc=3000); tone(56.0, 50, 0.55, 2); tone(56.0, 880, 0.06, 3)
    whoosh(69.3, 1.3, 0.3, fc=250); tone(71.8, 45, 0.45, 3)
    for k in range(3): whoosh(77.2 + k * 1.35 + 0.95, 0.4, 0.12, fc=300)
    tone(81.4, 1318, 0.03, 0.8); tone(81.4, 659, 0.03, 0.6)
    whoosh(89.9, 0.8, 0.15, rev=True, fc=600); tone(90.7, 55, 0.4, 4)
    whoosh(92.0, 1.0, 0.25); tone(93.45, 40, 0.5, 1.5); tone(93.45, 1760, 0.03, 2)
    tone(114.8, 659, 0.07, 1.2); tone(114.8, 988, 0.05, 1.2); tone(127.4, 784, 0.07, 1.0); tone(127.4, 1175, 0.05, 1.0)
    for tt0 in (133.6, 136.3, 139.0, 141.7): tone(tt0, 1046, 0.035, 3)
    whoosh(144.9, 1.8, 0.28, fc=250); tone(146.6, 45, 0.4, 3)
    tone(147.9, 523, 0.05, 0.5); tone(147.9, 784, 0.04, 0.5); tone(148.4, 1046, 0.03, 0.5)
    tone(157.4, 1760, 0.03, 1.2)
    fx = fx / max(1e-6, np.abs(fx).max()) * 0.5; st = np.stack([fx, np.roll(fx, 220)], 1)
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR_); w.writeframes((st * 32767).astype(np.int16).tobytes())

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "pv":
        for s in sys.argv[2:]: Image.fromarray(render(float(s))).save(f"pv_{s}.png")
    elif mode == "sheet":
        ts = [float(s) for s in sys.argv[3:]]; cells = []
        for s in ts:
            im = Image.fromarray(render(s)).resize((480, 360)); ImageDraw.Draw(im).text((6, 4), f"{s}", fill=(255, 60, 60)); cells.append(im)
        cols = 4; rows = (len(cells) + 3) // 4; sh = Image.new("RGB", (480 * cols, 360 * rows), (60, 60, 60))
        for i, c in enumerate(cells): sh.paste(c, ((i % cols) * 480, (i // cols) * 360))
        sh.save(sys.argv[2])
    elif mode == "seg":
        a, b, out = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
        p = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                              "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
        for i in range(a, b): p.stdin.write(render(i / FPS).tobytes())
        p.stdin.close(); p.wait(); print("seg done", out, p.returncode)
    elif mode == "audio": build_audio()
