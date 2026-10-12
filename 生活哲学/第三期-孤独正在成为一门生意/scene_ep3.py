# 第三期《孤独，正在成为一门生意》：点阵线描 + 粒子变形 + 全局粒子银河（GPU 渲染）
# 用法与第二期 GPU 版相同：
#   python scene_ep3.py bench
#   python scene_ep3.py gpu_pv 16.6 56.2 98 150
#   python scene_ep3.py sheet sheet.png 2 8 18 25 ...
#   python scene_ep3.py full            → ep3_gudu_4x3.mp4
import sys, subprocess, wave
from engine import *
import engine as _E

DUR = 160.0
HX = 720; HH = 170; CH = 150
OUT_NAME = "ep3_gudu_4x3.mp4"

# ------------------------------------------------------------ 画面参数
PARTICLE_LIGHT = 1100.0
EXPOSURE = 1.45
TONE_KNEE = 285.0
VIG_STRENGTH = 0.16
CORE_SHARP_SHARE = 0.65
CORE_SIGMA = 0.48
SUBJECT_GLOW_SIGMA = 2.8
SUBJECT_GLOW_GAIN = 0.22
GROUND_GLOW_SIGMA = 1.2
GROUND_GLOW_GAIN = 0.05
# 银河
GAL_LIGHT = 380.0          # 银河粒子亮度
GAL_GLOW = 0.30            # 小光晕
GAL_HAZE = 1.1             # 星云柔光（大半径）
GAL_HAZE_SIGMA = 22.0
# 拖尾（运动模糊）：时间偏移与权重
TRAIL = [(0.0, 0.46), (0.033, 0.24), (0.066, 0.17), (0.10, 0.13)]
# 字幕：比第二期更小、更靠下，把画面让给粒子
SUB_SIZE = 34; SUB_Y = 1024
SUBC = (246, 243, 236); GOLD = tuple(int(v) for v in WARM)

# ------------------------------------------------------------ 几何工具
def lines(*segs): return [np.array(s, np.float32) for s in segs]
def scale_pts(p, cx, cy, s, ox=0, oy=0):
    q = p.copy(); q[:, 0] = (q[:, 0] - ox) * s + cx; q[:, 1] = (q[:, 1] - oy) * s + cy; return q
def run_ph(t, f=2.4, k=0): return 2 * math.pi * f * t + k * 0.9
def rect(x0, y0, x1, y1): return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]], np.float32)
def rrect(cx, cy, w, h, r):
    ps = []
    for (ox, oy, a0) in ((w / 2 - r, -h / 2 + r, -math.pi / 2), (w / 2 - r, h / 2 - r, 0), (-w / 2 + r, h / 2 - r, math.pi / 2), (-w / 2 + r, -h / 2 + r, math.pi)):
        ps.append(circle(cx + ox, cy + oy, r, a0, a0 + math.pi / 2, 10))
    p = np.concatenate(ps); return np.vstack([p, p[:1]]).astype(np.float32)

def heart_poly(cx, cy, s):
    a = np.linspace(0, 2 * math.pi, 120)
    x = 16 * np.sin(a) ** 3; y = -(13 * np.cos(a) - 5 * np.cos(2 * a) - 2 * np.cos(3 * a) - np.cos(4 * a))
    return np.stack([cx + x * s, cy + y * s], 1).astype(np.float32)

def phone_polys(cx, cy, w=46, h=80):
    return [rrect(cx, cy, w, h, max(4, w * 0.16)), np.array([[cx - w * 0.15, cy - h * 0.40], [cx + w * 0.15, cy - h * 0.40]], np.float32)]

def clock_polys(cx, cy, R, mh, hh):
    ps = [circle(cx, cy, R)]
    for i in range(12):
        a = i * math.pi / 6; r0 = R * 0.84
        ps.append(np.array([[cx + r0 * math.cos(a), cy + r0 * math.sin(a)], [cx + R * 0.94 * math.cos(a), cy + R * 0.94 * math.sin(a)]], np.float32))
    ps.append(np.array([[cx, cy], [cx + R * 0.78 * math.sin(mh), cy - R * 0.78 * math.cos(mh)]], np.float32))
    ps.append(np.array([[cx, cy], [cx + R * 0.5 * math.sin(hh), cy - R * 0.5 * math.cos(hh)]], np.float32))
    return ps

def bubble_polys(cx, cy, t=0.0):
    b = rrect(cx, cy, 150, 84, 26)
    tail = np.array([[cx - 40, cy + 40], [cx - 58, cy + 70], [cx - 14, cy + 42]], np.float32)
    ps = [b, tail]
    for i in range(9):
        x = cx - 52 + i * 13; hgt = 8 + 22 * abs(math.sin(i * 1.7 + t * 6))
        ps.append(np.array([[x, cy - hgt / 2], [x, cy + hgt / 2]], np.float32))
    return ps

def bowl_polys(cx, cy, flip=False):
    arc = circle(cx, cy - 30, 46, 0.0, math.pi, 30)
    rim = np.array([[cx - 52, cy - 30], [cx + 52, cy - 30]], np.float32)
    base = np.array([[cx - 18, cy + 18], [cx + 18, cy + 18]], np.float32)
    d = -1 if flip else 1
    sticks = [np.array([[cx + d * 10, cy - 34], [cx + d * 66, cy - 92]], np.float32), np.array([[cx + d * 22, cy - 34], [cx + d * 76, cy - 84]], np.float32)]
    steam = [np.array([[cx - 12 + 6 * math.sin(k * 0.6), cy - 46 - k * 6] for k in range(9)], np.float32)]
    return [arc, rim, base] + sticks + steam

def tag_polys(cx, cy, w=240, h=132):
    x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
    body = np.array([[x0 + 30, y0], [x1, y0], [x1, y1], [x0 + 30, y1], [x0, cy], [x0 + 30, y0]], np.float32)
    return [body, circle(x0 + 24, cy, 7, k=16)]

def table_polys(x0, x1, y=600):
    return lines([[x0, y], [x1, y]], [[x0 + 30, y], [x0 + 30, GY]], [[x1 - 30, y], [x1 - 30, GY]])

def building_polys(x, w, top, cols, rows, seed):
    g = np.random.default_rng(seed)
    ps = [np.array([[x, GY], [x, top], [x + w, top], [x + w, GY]], np.float32)]
    ww = (w - 20) / cols; hh = (GY - top - 30) / rows; lit = []
    for i in range(cols):
        for j in range(rows):
            if g.random() < 0.72:
                wx = x + 10 + i * ww + ww * 0.2; wy = top + 16 + j * hh + hh * 0.2
                ps.append(rect(wx, wy, wx + ww * 0.6, wy + hh * 0.55)); lit.append((wx + ww * 0.3, wy + hh * 0.55))
    return ps, lit

def merge(n, *arrs):
    pts = []
    for a in arrs:
        a = np.asarray(a, np.float32)
        if a.shape[1] == 3:
            k = max(1, min(len(a), int(round(a[:, 2].mean() * len(a))))); a = a[:k, :2]
        pts.append(a[:, :2])
    return spread(np.concatenate(pts), n, 0.18)

def polys_n(polys, n, sp=6.0): return sample_polys(polys, n, max(2.8, sp * 0.6), jitter=0.18)
def dust_ground(n, seed=3, x0=100, x1=1340):
    r = np.random.default_rng(seed); return spread(np.stack([r.uniform(x0, x1, n), GY - np.abs(r.normal(0, 3, n))], 1).astype(np.float32), n)
def point(n, x, y, s=2.5):
    r = np.random.default_rng(n); return spread(np.stack([x + r.normal(0, s, n), y + r.normal(0, s, n)], 1).astype(np.float32), n)
def burst(prev, r=420, seed=1, cx=None, cy=None):
    g = np.random.default_rng(seed); p = prev.copy()
    c = np.array([cx if cx is not None else p[:, 0].mean(), cy if cy is not None else p[:, 1].mean()])
    v = p[:, :2] - c; v = v / (np.hypot(v[:, 0], v[:, 1])[:, None] + 1e-3)
    p[:, :2] += v * g.uniform(0.3, 1.0, (len(p), 1)) * r + g.normal(0, 40, (len(p), 2)); return p
def burst_fn(r, sd, cx=None, cy=None): return lambda n, t, prev: burst(prev, r, sd, cx, cy)
def hidden(actor): return S(lambda n, t: actor.dust.copy())
def refl(a, gy=GY):
    """水面倒影：沿地平线上下翻转"""
    b = a.copy(); b[:, 1] = 2 * gy - b[:, 1]; return b

# ------------------------------------------------------------ 加密点阵字
_orig_text_pts = _E.text_pts_local
_dtc = {}
def text_pts_dense(text, w, sz, spacing):
    k = (text, w, sz, spacing)
    if k in _dtc: return _dtc[k]
    outline = _orig_text_pts(text, w, sz, 1.6)
    f = font(w, sz); d0 = ImageDraw.Draw(Image.new("L", (8, 8)))
    bb = d0.textbbox((0, 0), text, font=f); tw, th = bb[2] - bb[0] + 20, bb[3] - bb[1] + 20
    im = Image.new("L", (tw, th), 0); ImageDraw.Draw(im).text((10 - bb[0], 10 - bb[1]), text, font=f, fill=255)
    a = cv2.erode((np.array(im) > 128).astype(np.uint8), np.ones((2, 2), np.uint8))
    g = max(2.6, sz / 52.0)
    gy, gx = np.mgrid[0:th:g, 0:tw:g]; gx = gx + (np.round(gy / g) % 2) * g / 2
    gx = gx.ravel(); gy = gy.ravel(); ok = (gx < tw - 1) & (gy < th - 1)
    gx, gy = gx[ok], gy[ok]; inside = a[gy.astype(int), gx.astype(int)] > 0
    fill = np.stack([gx[inside], gy[inside]], 1).astype(np.float32) - np.array([tw / 2, th / 2], np.float32)
    pts = np.concatenate([outline, fill]) if len(fill) else outline
    _dtc[k] = pts; return pts
_E.text_pts_local = text_pts_dense
text_pts_local = text_pts_dense

def text_polys_pts(text, cx, cy, sz):
    return text_pts_dense(text, "L", sz, 2) + [cx, cy]

# ------------------------------------------------------------ actors
HERO = Actor("hero", 620, 1); OBJ = Actor("obj", 2800, 2); OBJ2 = Actor("obj2", 1400, 3)
CROWD = Actor("crowd", 3600, 4); TXT = Actor("txt", 4600, 5); KW = Actor("kw", 3200, 7)
MIR = Actor("mir", 620, 8)     # 倒影 / 另一个自己
ACTORS = [CROWD, OBJ, OBJ2, MIR, KW, TXT, HERO]

def crowd_people(n, specs):
    per = n // len(specs); out = []
    for i, (pz, x, y, h, ph, rot, fl) in enumerate(specs):
        k = per if i < len(specs) - 1 else n - per * (len(specs) - 1)
        out.append(person(pz, x, y, h, k, ph=ph, rot=rot, flip=fl))
    return np.concatenate(out)

def person_polys(name, x, y, h, ph=0.0, flip=False):
    return [p + [x, y] for p in person_contour(name, ph, h, 0.0, flip)]

# 坐姿时手的大致位置（相对髋部）：用于把手机放到手里
def hand_xy(x, y, h=HH, flip=False):
    return (x + (0.43 * h if not flip else -0.43 * h), y - 0.47 * h)

# ---- 关键词点阵（与字幕同步出现 / 炸散）
KWS = [(9.4, 11.7, "¥6", 1110, 300, 130, 1),
       (20.9, 24.8, "1 亿 +", 1120, 150, 96, 0),
       (37.6, 41.1, "续费", 720, 200, 120, 1),
       (45.4, 49.5, "真人", 1100, 300, 110, 0),
       (49.9, 53.6, "不用冒险", 1110, 260, 84, 0),
       (57.9, 61.9, "Hannah Arendt", 1040, 260, 64, 1),
       (62.2, 66.3, "寂寞", 1120, 240, 120, 0),
       (66.6, 70.5, "独处", 1120, 240, 120, 1),
       (70.9, 74.5, "好卖", 300, 240, 120, 0),
       (77.5, 81.1, "解药", 300, 240, 120, 0),
       (85.9, 89.5, "01:00", 1080, 290, 110, 0),
       (108.4, 112.3, "不是病", 1120, 220, 110, 0),
       (116.9, 120.8, "没有错", 720, 170, 96, 0),
       (120.9, 125.0, "所有的安静", 720, 170, 88, 0),
       (125.1, 129.0, "拿回来", 720, 170, 104, 1),
       (129.2, 133.4, "一个人", 720, 220, 104, 0),
       (133.6, 136.2, "十分钟", 720, 220, 96, 1),
       (136.3, 138.9, "一条语音", 720, 220, 96, 0),
       (139.0, 141.6, "允许冷场", 720, 220, 96, 1),
       (141.7, 144.8, "我陪我", 720, 220, 104, 1)]
def _kw_fn(txt, x, y, sz): return lambda n, t: text_shape(txt, x, y, sz, n, "L", 4)
def _kw_burst(seed): return lambda n, t, prev: burst(prev, 240, seed)
for i, (t0, t1, txt, x, y, sz, wm) in enumerate(KWS):
    KW.key(t0, S(_kw_fn(txt, x, y, sz)), 0.9, a=0.9, col=WARM if wm else FG, stag=0.35, swirl=110)
    KW.key(t1 - 0.25, S(_kw_burst(40 + i)), 0.8, a=0.0, stag=0.25)

# ================================================================ ACT 1  现象
# 0–3.9 凌晨一点，一个人坐在地平线上看手机
HERO.key(0.0, S(lambda n, t: person("sit", HX, GY, HH, n)), 1.6, swirl=90)
def phone_hand(x, y, h=HH): hx, hy = hand_xy(x, y, h); return phone_polys(hx + 6, hy - 22, 30, 52)
def hearts_rising(t, cx, cy, t0, count=7, span=300, s=1.6):
    ps, ws = [], []
    for k in range(count):
        q = ((t - t0) * 0.55 + k / count) % 1.0
        if t < t0 + k * 0.25: continue
        x = cx + 46 * math.sin(q * 7 + k * 1.3) * q + (k - count / 2) * 10; y = cy - q * span
        sc = s * (0.6 + 0.6 * q)
        ps.append(heart_poly(x, y, sc)); ws.append(math.sin(math.pi * q) ** 0.8)
    return ps, ws
def weighted_polys(ps, ws, n, sp=5.0):
    if not ps: return point(n, HX, GY, 1.0)
    L = [float(np.hypot(*np.diff(p, axis=0).T).sum()) for p in ps]; tot = sum(L)
    out = []
    for i, (p, w) in enumerate(zip(ps, ws)):
        k = max(8, int(n * L[i] / tot)); a = sample_polys([p], k, 3.0); a[:, 2] *= w; out.append(a)
    a = np.concatenate(out)
    if len(a) >= n: return a[:n]
    b = a[np.arange(n) % len(a)].copy(); return b
def phone_hearts(n, t):
    base = sample_polys(phone_hand(HX, GY), n // 3, 3.0)
    if t < 1.9: return np.concatenate([base, point(n - len(base), *hand_xy(HX, GY, HH), 2.0)])
    hx, hy = hand_xy(HX, GY, HH); ps, ws = hearts_rising(t, hx + 6, hy - 50, 1.9, 6, 260, 1.1)
    return np.concatenate([base, weighted_polys(ps, ws, n - len(base))])
OBJ2.key(0.0, A(phone_hearts), 1.4, a=0.95, col=WARM, swirl=60)

# 3.9–11.7 主播的大屏幕；礼物爱心不断往上冒
SX, SY, SW, SH_ = 1040, 420, 330, 560
HERO.key(3.9, S(lambda n, t: person("sit", 420, GY, HH, n)), 1.0, swirl=60)
OBJ.key(3.9, S(lambda n, t: polys_n([rrect(SX, SY, SW, SH_, 34), np.array([[SX - 40, SY - SH_ / 2 + 22], [SX + 40, SY - SH_ / 2 + 22]], np.float32),
                                      rrect(SX - 90, SY - SH_ / 2 + 60, 110, 26, 12)], n, 5)), 1.1, a=0.85, swirl=140)
CROWD.key(3.9, A(lambda n, t: crowd_people(n, [("tpose" if int(t * 2.2) % 2 else "stand", SX, SY + 180, 190, 0, 0, False)])), 1.0, a=0.55, swirl=100)
def stream_hearts(n, t):
    base = sample_polys(phone_hand(420, GY), n // 4, 3.0)
    ps, ws = hearts_rising(t, SX + 100, SY + 220, 3.9, 9, 420, 1.5)
    hx, hy = hand_xy(420, GY, HH)
    # 一条从手机飞向屏幕的点线
    u = np.linspace(0, 1, 40); fly = ((t * 0.8) % 1.0)
    arc = np.stack([hx + (SX - 140 - hx) * u, hy - 20 + (SY + 120 - hy) * u - 120 * np.sin(np.pi * u)], 1).astype(np.float32)
    m = np.abs(u - fly) < 0.12
    ps2 = ps + ([arc[m]] if m.sum() >= 2 else []); ws2 = ws + ([1.0] if m.sum() >= 2 else [])
    return np.concatenate([base, weighted_polys(ps2, ws2, n - len(base))])
OBJ2.key(3.9, A(stream_hearts), 1.0, a=1.0, col=WARM, swirl=60)

# 11.9–16.5 "一人经济"的词螺旋卷入，收成一点
SPW = ["一人食", "陪聊", "搭子", "树洞", "AI 恋人", "直播间", "哄睡", "云陪伴", "一人份", "深夜电台", "付费自习", "租个朋友"]
SPP = [_orig_text_pts(w, "L", 60, 1.6) for w in SPW]
SP_T0, SP_STEP, SP_SPIN = 10.4, 0.26, 14.4
def spiral_u(t): return cl((t - SP_SPIN) / 2.0)
def spiral_words(n, t):
    u = spiral_u(t); r0 = 470 * (1 - eio(u)) ** 0.9 + 4
    om = 0.08 * (t - SP_T0) + 1.4 * u + 8.0 * u ** 2
    per = n // len(SPP); out = []
    for k, p in enumerate(SPP):
        a = 2 * math.pi * k / 12 + om - math.pi / 2; r = r0 * (0.74 + 0.26 * ((k * 5) % 3) / 2)
        sc = 0.25 + 0.75 * (r / 470)
        ap = float(eout(cl((t - SP_T0 - k * SP_STEP) / 0.45)))
        cx = HX + r * math.cos(a); cy = 470 + r * math.sin(a) * 0.82
        cx = HX + (cx - HX) * (0.35 + 0.65 * ap); cy = 470 + (cy - 470) * (0.35 + 0.65 * ap)
        q = spread(p * (sc * (0.4 + 0.6 * ap)) + [cx, cy], per if k < 11 else n - per * 11, 0.3)
        q[:, 2] *= ap * (1.25 - 0.3 * (ap - 1) ** 2) * (1 - 0.9 * u)
        out.append(q)
    return np.concatenate(out)
TXT.key(SP_T0, A(spiral_words), 0.6, a=1.0, swirl=80)
for ac, sd in ((OBJ, 2), (OBJ2, 3), (CROWD, 4)):
    ac.key(11.9, S(burst_fn(300, sd)), 1.2, a=0.0, stag=0.3)
HERO.key(12.4, A(lambda n, t: (lambda u: np.column_stack([scale_pts(person("sit", 0, 0, HH, n)[:, :2], 420 + (HX - 420) * u, GY - 250 * u, 1 - 0.95 * u),
                                                            np.full(n, 1.0 - 0.9 * u)]))(eio(cl((t - SP_SPIN) / 1.9)))), 1.0, swirl=30)
# 16.5 冲击：标题由粒子拼出
TXT.key(16.5, S(lambda n, t: text_shape("孤独，正在成为一门生意", HX, 440, 80, n, "L", 3)), 1.1, a=1.0, swirl=160, stag=0.4)
HERO.key(16.5, S(lambda n, t, prev: burst(prev, 500, 5, HX, 470)), 1.0, a=0.0)
TXT.key(20.5, S(lambda n, t, prev: burst(prev, 380, 6)), 1.0, a=0.0, stag=0.35)

# 20.7–27.2 城市的窗：一亿多个一人户
CITY = []; CITY_LIT = []
_bx = -1150; _g = np.random.default_rng(12)
while _bx < 2600:
    bw = float(_g.uniform(110, 190)); top = float(_g.uniform(330, 560)); cols = int(bw // 46); rows = int((GY - top) // 54)
    ps, lit = building_polys(_bx, bw, top, cols, rows, int(_bx + 5000)); CITY.append(ps); CITY_LIT += [(x, y, abs(_bx - 650) < 650) for (x, y) in lit]
    _bx += bw + float(_g.uniform(14, 40))
CITY_ALL = [p for ps in CITY for p in ps]
CITY_NEAR = [p for ps in CITY for p in ps if p[:, 0].min() > -60 and p[:, 0].max() < 1500]
_near = [(x, y) for (x, y, ok) in CITY_LIT if ok and 60 < x < 1380]
_ng = np.random.default_rng(3); WIN_P = [_near[i] for i in sorted(_ng.choice(len(_near), min(16, len(_near)), replace=False))]
OBJ.key(20.7, S(lambda n, t: polys_n(CITY_NEAR, n, 6)), 1.5, a=0.8, swirl=120)
CROWD.key(20.9, S(lambda n, t: crowd_people(n, [("sit", x - 6, y, 26, 0, 0, i % 2 == 1) for i, (x, y) in enumerate(WIN_P)])), 1.3, a=0.7, col=WARM, swirl=60)
OBJ.key(24.6, S(lambda n, t: polys_n(CITY_ALL, n, 6)), 2.2, a=0.8, stag=0.5, swirl=30)

# 27.2–41.3 价签：陪伴被明码标价，绕着人转
TAGS = [("陪聊", "¥30 / 小时"), ("哄睡", "¥19.9 / 月"), ("AI 恋人", "¥68 / 月"), ("租个朋友", "¥200 / 天")]
def tag_ring(n, t, k, rx=440, ry=160, spd=0.16, close=0.0):
    ps = []
    for i in range(k):
        a = spd * t + i * 2 * math.pi / max(k, 3) + 0.4
        r = 1 - close; x = HX + rx * r * math.cos(a); y = 470 + ry * r * math.sin(a) + 10 * math.sin(t * 1.3 + i)
        ps.append((x, y))
    out = []; per = n // max(k, 1)
    for i, (x, y) in enumerate(ps):
        m = per if i < k - 1 else n - per * (k - 1)
        body = sample_polys(tag_polys(x, y), m // 2, 3.0)[:, :2]
        txt = np.concatenate([text_polys_pts(TAGS[i][0], x + 14, y - 24, 38), text_polys_pts(TAGS[i][1], x + 14, y + 26, 28)])
        out.append(spread(np.concatenate([body[: m // 2], txt]), m, 0.15))
    return np.concatenate(out)
OBJ.key(27.2, A(lambda n, t: tag_ring(n, t, 2)), 1.3, a=0.9, swirl=140)
OBJ.key(30.3, A(lambda n, t: tag_ring(n, t, 3)), 1.0, a=0.9, swirl=80)
OBJ.key(33.8, A(lambda n, t: tag_ring(n, t, 4)), 1.0, a=0.9, swirl=80)
OBJ.key(37.5, A(lambda n, t: tag_ring(n, t, 4, spd=0.16 + 0.9 * float(eio((t - 37.5) / 3.0)), close=0.28 * float(eio((t - 37.5) / 3.5)))), 1.0, a=1.0, col=WARM, swirl=40)
CROWD.key(27.2, S(lambda n, t, prev: burst(prev, 260, 9)), 1.0, a=0.0)
HERO.key(27.2, S(lambda n, t: person("stand", HX, GY, HH, n)), 1.2, swirl=90)
HERO.key(37.6, S(lambda n, t: person("sad", HX, GY, HH, n)), 1.4, swirl=20)

# 41.3–49.6 两个人坐在桌子两端，桌子越拉越长；对面的人变成一块屏幕
def tbl_x(t): return 260 * float(eio((t - 41.3) / 7.5))
OBJ.key(41.3, A(lambda n, t: polys_n(table_polys(420 - tbl_x(t), 1020 + tbl_x(t)) + bowl_polys(500 - tbl_x(t), 600) + bowl_polys(940 + tbl_x(t), 600, True), n, 6)), 1.3, swirl=140)
HERO.key(41.3, A(lambda n, t: person("sit", 360 - tbl_x(t), GY, HH, n)), 1.2, swirl=60)
CROWD.key(41.4, A(lambda n, t: crowd_people(n, [("sit", 1080 + tbl_x(t), GY, HH, 0, 0, True)])), 1.2, a=0.7, swirl=60)
CROWD.key(45.3, A(lambda n, t: polys_n(phone_polys(1140 + tbl_x(t), 560, 80, 140) + [heart_poly(1140 + tbl_x(t), 560, 1.6)], n, 4)), 1.2, a=0.8, col=WARM, stag=0.5, swirl=120)

# 49.8–55.9 手机成了一只笼子
OBJ.key(49.8, S(lambda n, t: polys_n([rrect(HX, 430, 380, 640, 46), np.array([[HX - 50, 140], [HX + 50, 140]], np.float32)] +
                                    [np.array([[HX - 150 + i * 50, 140], [HX - 150 + i * 50, 720]], np.float32) for i in range(7)], n, 5)), 1.4, a=0.85, swirl=150)
HERO.key(49.8, S(lambda n, t: person("stand", HX, 690, HH * 1.05, n)), 1.2, swirl=80)
CROWD.key(49.8, A(lambda n, t: crowd_people(n, [("walk", (x + 70 * (t - 49.8) * d) % 1500 - 30, GY, CH * 0.8, run_ph(t, 1.0, i), 0, d < 0)
                                               for i, (x, d) in enumerate(((100, 1), (240, 1), (1200, -1), (1340, -1), (60, -1), (1290, 1)))])), 1.2, a=0.32, swirl=60)
OBJ2.key(49.8, hidden(OBJ2), 0.5, a=0.0)
# 56.0 冲击："谁在陪你？"
for ac, sd in ((OBJ, 21), (CROWD, 22), (HERO, 23)):
    ac.key(56.0, S(burst_fn(520, sd, HX, 450)), 0.9, a=0.0, stag=0.2, swirl=80)

# ================================================================ ACT 2  阿伦特：寂寞与独处
DX = 700
def desk_like(x):
    T_ = 640
    ps = lines([[x - 34, GY], [x - 34, GY - 46], [x + 26, GY - 46]], [[x - 34, GY - 46], [x - 40, GY - 150]],
               [[x + 40, T_], [x + 380, T_]], [[x + 60, T_], [x + 60, GY]], [[x + 360, T_], [x + 360, GY]])
    # 书堆 + 台灯
    for i in range(4): ps.append(rect(x + 230 + i * 4, T_ - 22 * (i + 1), x + 330 - i * 6, T_ - 22 * i))
    ps += lines([[x + 120, T_], [x + 150, T_ - 120], [x + 210, T_ - 150]])
    ps.append(circle(x + 216, T_ - 140, 26, -0.4, math.pi * 0.9, 20))
    return ps
OBJ.key(57.8, S(lambda n, t: polys_n(desk_like(DX - 200), n, 6)), 1.6, a=0.9, swirl=120)
HERO.key(57.9, S(lambda n, t: person("sit", DX - 200, GY, HH, n)), 1.4, swirl=80)
def smoke(n, t):   # 台灯光里的尘埃缓缓旋转
    g = np.random.default_rng(4); k = n
    a = g.uniform(0, 6.28, k) + t * g.uniform(0.2, 0.6, k); r = g.uniform(10, 120, k) ** 1.0
    x = DX + 16 + r * np.cos(a) * 1.2; y = 640 - 230 + r * np.sin(a) * 0.6 - 30
    return np.stack([x, y, np.full(k, 0.5)], 1).astype(np.float32)
OBJ2.key(58.5, A(smoke), 1.5, a=0.55, col=WARM, swirl=60)
# 62.1 寂寞：人潮从他身边穿过
OBJ.key(62.1, S(lambda n, t, prev: burst(prev, 260, 31)), 1.0, a=0.0)
OBJ2.key(62.1, S(lambda n, t, prev: burst(prev, 200, 32)), 0.8, a=0.0)
HERO.key(62.1, S(lambda n, t: person("stand", HX, GY, HH, n)), 1.0, swirl=60)
def passers(n, t):
    sp = []
    for k in range(10):
        d = 1 if k % 2 == 0 else -1; v = 120 + 25 * (k % 4)
        x = ((k * 173 + (t - 62.1) * v * d) % 1700) - 130
        if abs(x - HX) < 50: x += 100 * d
        sp.append(("walk", x, GY, CH * (0.85 + 0.1 * (k % 3)), run_ph(t, 1.1, k), 0, d < 0))
    return crowd_people(n, sp)
CROWD.key(62.2, A(passers), 1.2, a=0.45, swirl=50)
# 66.5 独处：人潮散去，他坐下，地面出现倒影
CROWD.key(66.5, S(lambda n, t, prev: burst(prev, 300, 33)), 1.0, a=0.0)
HERO.key(66.5, S(lambda n, t: person("sit", HX, GY, HH, n)), 1.3, swirl=40)
MIR.key(66.8, S(lambda n, t: refl(person("sit", HX, GY, HH, n))), 1.6, a=0.42, col=WARM, swirl=40, stag=0.6)
OBJ.key(66.6, S(lambda n, t: dust_ground(n, 5, -100, 1540)), 1.2, a=0.22, swirl=60)
# 70.8 寂寞好卖：屏幕成群围过来
def screens(n, t, k, r0, r1, t0, dur, spd=0.5):
    u = float(eio((t - t0) / dur)); rr = r0 + (r1 - r0) * u; ps = []
    for i in range(k):
        a = spd * t + i * 2 * math.pi / k; rad = rr * (0.85 + 0.15 * math.sin(i * 2.3))
        x = HX + rad * math.cos(a) * 1.25; y = 560 + rad * math.sin(a) * 0.62
        sz = 0.8 + 0.3 * math.sin(i * 1.7)
        ps += phone_polys(x, y, 40 * sz, 70 * sz)
    return polys_n(ps, n, 4)
MIR.key(70.8, S(lambda n, t, prev: burst(prev, 120, 34)), 1.0, a=0.0)
OBJ2.key(70.8, A(lambda n, t: screens(n, t, 8, 620, 380, 70.8, 6.0)), 1.3, a=0.9, col=WARM, swirl=120)
OBJ2.key(77.2, A(lambda n, t: screens(n, t, 14, 380, 230, 77.2, 4.0, 0.8)), 1.0, a=1.0, col=WARM, swirl=60)
HERO.key(77.4, S(lambda n, t: person("sad", HX, GY, HH, n)), 1.2, a=0.6, swirl=20)
# 81.4 解药越多：屏幕停住，银河亮起——他错过的那片安静
OBJ2.key(81.4, A(lambda n, t: screens(n, t, 14, 230, 200, 81.4, 3.0, 0.15)), 1.0, a=0.55, col=WARM, swirl=20)
# 85.8 刷到凌晨
OBJ2.key(85.8, A(lambda n, t: polys_n(phone_hand(HX, GY) + [circle(hand_xy(HX, GY)[0] + 6, hand_xy(HX, GY)[1] - 22, 30 + 6 * math.sin(t * 9), k=30)], n, 3)), 1.2, a=1.0, col=WARM, swirl=80)
HERO.key(85.8, S(lambda n, t: person("sit", HX, GY, HH, n)), 1.2, a=1.0, swirl=40)
OBJ.key(85.8, A(lambda n, t: polys_n(clock_polys(330, 330, 90, t * 2.2, t * 0.18), n, 5)), 1.2, a=0.6, swirl=100)
# 90.7 屏幕熄灭；92.4 全部收成一点
OBJ2.key(90.7, S(lambda n, t, prev: burst(prev, 160, 35)), 0.6, a=0.0, stag=0.15)
for ac in (HERO, OBJ, OBJ2, MIR, CROWD):
    ac.key(92.4, S(lambda n, t: point(n, HX, 540, 2.0)), 1.0, a=0.6, stag=0.3, swirl=90)

# ================================================================ ACT 3  把安静拿回来
# 93.55 一点展开成两个人：他，和另一个自己
HERO.key(93.55, S(lambda n, t: person("stand", 560, GY, HH, n)), 1.4, swirl=140)
MIR.key(93.55, S(lambda n, t: person("stand", 880, GY, HH, n, flip=True)), 1.6, a=0.55, col=WARM, swirl=140)
for ac in (OBJ, OBJ2, CROWD): ac.key(93.55, hidden(ac), 0.8, a=0.0)
TXT.key(97.8, S(lambda n, t: text_shape("独处，是和自己在一起。", HX, 390, 82, n, "L", 3)), 1.3, a=1.0, col=WARM, stag=0.4, swirl=120)
HERO.key(98.4, S(lambda n, t: person("reach", 560, GY, HH, n)), 1.2, swirl=20)
TXT.key(102.6, S(lambda n, t: text_shape("寂寞，是连自己也弄丢了。", HX, 390, 78, n, "L", 3)), 1.3, a=1.0, col=FG, stag=0.45, swirl=120)
MIR.key(102.8, S(lambda n, t, prev: burst(prev, 340, 41, 880, 600)), 2.2, a=0.0, stag=0.7, swirl=160)
HERO.key(103.2, S(lambda n, t: person("sad", 560, GY, HH, n)), 1.2, swirl=20)
TXT.key(107.6, S(lambda n, t, prev: burst(prev, 300, 14)), 1.0, a=0.0)
# 108.2 坐在银河下
HERO.key(108.2, S(lambda n, t: person("lookup", HX, GY, HH, n)), 1.4, swirl=60)
MIR.key(108.4, S(lambda n, t: refl(person("lookup", HX, GY, HH, n))), 1.6, a=0.30, col=FG, swirl=60, stag=0.6)
OBJ.key(108.2, S(lambda n, t: dust_ground(n, 7, -100, 1540)), 1.3, a=0.22, swirl=60)
TXT.key(112.6, S(lambda n, t: text_shape("能力", 1130, 240, 170, n, "L", 4)), 1.2, a=1.0, col=WARM, swirl=120)
HERO.key(114.8, S(lambda n, t: person("lookup", HX, GY, HH, n)), 1.0, col=WARM, stag=0.7, swirl=5)
MIR.key(114.8, S(lambda n, t: refl(person("lookup", HX, GY, HH, n))), 1.0, a=0.42, col=WARM, stag=0.7, swirl=5)
TXT.key(116.5, S(lambda n, t, prev: burst(prev, 220, 15)), 1.0, a=0.0)
# 116.8 买陪伴没有错；120.9 只是别让它替你回答所有的安静
CROWD.key(116.8, A(lambda n, t: polys_n(phone_polys(330, 330, 90, 150) + [heart_poly(330, 335, 1.8 * (1 + 0.12 * abs(math.sin(t * 6)) ** 4))], n, 4)), 1.0, a=0.5, col=WARM, swirl=80)
CROWD.key(120.9, A(lambda n, t: polys_n(person_polys("sit", 1110, 420, 90) +
                                       sum([phone_polys(1110 + 120 * math.cos(t * 0.8 + i * 1.257), 360 + 60 * math.sin(t * 0.8 + i * 1.257), 30, 50) for i in range(5)], []), n, 4)), 1.0, a=0.5, swirl=80)
CROWD.key(125.0, S(lambda n, t, prev: burst(prev, 200, 16)), 1.0, a=0.0)
# 129.2 一个人吃饭，也不必假装很忙：他慢慢走，身边是低头赶路的人
HERO.key(129.2, A(lambda n, t: person("walk", HX, GY, HH, n, ph=run_ph(t, 0.8))), 1.0, col=FG, swirl=60)
MIR.key(129.2, S(lambda n, t, prev: burst(prev, 120, 42)), 1.0, a=0.0)
def rushers(n, t):
    sp = []
    for k in range(7):
        x = -200 + ((t - 129.2) * (300 + 50 * k) + k * 260) % 1900
        sp.append(("sad" if k % 2 else "walk", x, GY, CH, run_ph(t, 2.2, k), 0, False))
    return crowd_people(n, sp)
CROWD.key(129.4, A(rushers), 0.8, a=0.40, swirl=40)
CROWD.key(133.3, S(lambda n, t, prev: burst(prev, 160, 18)), 0.8, a=0.0)
# 133.5 四个小行动
ICX = [330, 560, 790, 1020]
def icons(k, t=0.0):
    ps = []
    if k >= 1: ps += clock_polys(ICX[0], GY - 62, 46, 0.0, math.pi / 3)
    if k >= 2: ps += bubble_polys(ICX[1], GY - 90, t)
    if k >= 3: ps += bowl_polys(ICX[2] - 40, GY - 20) + bowl_polys(ICX[2] + 48, GY - 20, True)
    if k >= 4: ps += person_polys("stand", ICX[3] - 34, GY, 92) + person_polys("stand", ICX[3] + 34, GY, 92, flip=True) + [heart_poly(ICX[3], GY - 128, 0.9)]
    return ps
HERO.key(133.5, S(lambda n, t: person("stand", 1250, GY, HH, n)), 1.0, swirl=60)
def _icon_fn(k): return lambda n, t: polys_n(icons(k, t), n, 5)
for i, tt in enumerate((133.5, 136.3, 139.0, 141.7)):
    OBJ.key(tt, A(_icon_fn(i + 1)), 0.9, a=0.95, stag=0.5, swirl=90)
# 144.9 安静还会再来：坐回地平线，银河铺满
OBJ.key(144.9, S(lambda n, t: dust_ground(n, 9, -100, 1540)), 1.3, a=0.22, swirl=120)
HERO.key(144.9, S(lambda n, t: person("sit", HX, GY, HH, n)), 1.4, swirl=60)
MIR.key(145.2, S(lambda n, t: refl(person("sit", HX, GY, HH, n))), 1.6, a=0.30, col=FG, swirl=60, stag=0.6)
HERO.key(147.7, S(lambda n, t: person("lookup", HX, GY, HH, n)), 1.6, col=WARM, stag=0.6, swirl=10)
MIR.key(147.7, S(lambda n, t: refl(person("lookup", HX, GY, HH, n))), 1.8, a=0.45, col=WARM, stag=0.6, swirl=10)
for ac in (HERO, OBJ, OBJ2, TXT, CROWD, MIR, KW):
    ac.key(157.2, S(lambda n, t: point(n, HX, 540, 2.0)), 1.6, a=0.04, stag=0.3, swirl=50)

for a in ACTORS: a.prep()

# ------------------------------------------------------------ 镜头 / 冲击时间轴
CAM = [(0, 0.1, 1.08, 720, 560), (3.9, 2.0, 1.0, 720, 540), (11.9, 1.4, 1.0, 720, 520), (16.5, 3.5, 1.04, 720, 520),
       (20.7, 2.0, 1.0, 720, 540), (24.6, 2.4, 0.42, 720, 470), (27.2, 1.2, 1.0, 720, 540), (37.5, 3.6, 1.12, 720, 520),
       (41.3, 1.0, 1.0, 720, 540), (45.3, 4.2, 0.9, 720, 540), (49.8, 1.0, 1.0, 720, 500), (53.8, 2.0, 1.12, 720, 470),
       (56.0, 0.4, 1.0, 720, 540), (66.5, 4.0, 1.1, 720, 620), (70.8, 1.5, 1.0, 720, 560), (77.2, 4.0, 1.18, 720, 600),
       (81.4, 2.2, 0.86, 720, 480), (85.8, 1.5, 1.08, 720, 600), (92.4, 0.8, 1.0, 720, 540), (108.2, 8.0, 0.9, 720, 470),
       (116.8, 1.5, 1.0, 720, 540), (129.2, 1.2, 1.0, 720, 540), (144.9, 3.0, 0.92, 720, 500), (147.7, 9.0, 0.82, 720, 460)]
SHAKE = [(1.95, 0.3, 4), (43.4, 0.4, 8), (56.0, 0.55, 14), (90.7, 0.35, 7), (92.4, 0.6, 6), (114.8, 0.3, 4)]
def camera(t):
    z, cx, cy = 1.0, 720.0, 540.0
    for (t0, d, z1, x1, y1) in CAM:
        if t < t0: break
        e = float(eio((t - t0) / d)); z, cx, cy = z + (z1 - z) * e, cx + (x1 - cx) * e, cy + (y1 - cy) * e
    for (t0, d, amp) in SHAKE:
        if t0 <= t < t0 + d:
            k = (1 - (t - t0) / d) ** 2 * amp; g = np.random.default_rng(int(t * 60)); cx += g.normal(0, k); cy += g.normal(0, k)
    return z, cx, cy
# (时间, 世界x, 世界y, 半径, 暖色)
WAVES = [(1.95, 760, 580, 260, 1), (16.5, 720, 440, 1000, 0), (43.5, 720, GY, 300, 0), (56.0, 720, 450, 1100, 0),
         (66.8, 720, GY, 420, 1), (90.7, 720, 560, 380, 0), (93.5, 720, 540, 900, 0), (114.8, 720, GY - 80, 520, 1),
         (127.4, 720, GY, 700, 1), (147.9, 720, GY - 80, 1100, 1)]
FLASH = [(16.5, 0.6, 0.75), (56.0, 0.3, 0.35), (93.45, 0.35, 0.55)]
GROUND = [(0.0, 1.5, -200, 1640, 0.8, 0), (11.9, 0.8, 720, 720, 0.0, 0), (20.7, 1.0, -200, 1640, 0.8, 0), (24.6, 2.4, -2600, 4000, 0.8, 0),
          (27.2, 1.0, -200, 1640, 0.8, 0), (49.8, 1.0, 400, 1040, 0.4, 0), (55.6, 0.4, 720, 720, 0.0, 0), (57.8, 1.0, -200, 1640, 0.8, 0),
          (66.6, 2.0, -200, 1640, 0.9, 0.6), (70.8, 1.5, -200, 1640, 0.8, 0), (92.4, 0.5, 720, 720, 0, 0), (93.6, 1.4, -200, 1640, 0.8, 0),
          (114.8, 1.2, -200, 1640, 0.9, 0.7), (116.8, 1.5, -200, 1640, 0.8, 0), (147.7, 3.0, -200, 1640, 0.9, 1), (157.0, 1.6, 720, 720, 0.0, 1)]
def ground(t):
    x0, x1, a, wm = 720, 720, 0.0, 0.0
    for (t0, d, X0, X1, A_, WM) in GROUND:
        if t < t0: break
        e = float(eio((t - t0) / d)); x0, x1, a, wm = x0 + (X0 - x0) * e, x1 + (X1 - x1) * e, a + (A_ - a) * e, wm + (WM - wm) * e
    return x0, x1, a, wm

# ---- 银河时间轴：(开始, 过渡, 中心x, 中心y, 尺度, 倾角, 方位角, 亮度, 暖度)
GAL = [(0.0, 0.1, 900, 420, 1.55, 0.42, -0.35, 0.42, 0.0), (11.9, 4.6, 720, 480, 1.9, 0.55, -0.20, 0.80, 0.0),
       (16.5, 3.0, 720, 640, 1.8, 0.40, -0.30, 0.50, 0.0), (20.7, 2.0, 720, 300, 2.0, 0.30, -0.12, 0.50, 0.0),
       (27.2, 1.5, 720, 430, 1.7, 0.42, -0.30, 0.50, 0.0), (41.3, 1.5, 720, 360, 1.8, 0.38, -0.25, 0.45, 0.0),
       (49.8, 1.5, 720, 430, 1.5, 0.45, -0.30, 0.32, 0.0), (57.8, 1.5, 900, 360, 1.8, 0.40, -0.30, 0.45, 0.2),
       (66.5, 2.0, 720, 380, 1.9, 0.42, -0.25, 0.65, 0.25), (70.8, 2.0, 720, 400, 1.6, 0.42, -0.25, 0.40, 0.0),
       (81.4, 2.2, 720, 420, 2.3, 0.50, -0.25, 1.00, 0.10), (85.8, 1.5, 720, 400, 1.7, 0.42, -0.25, 0.32, 0.0),
       (92.0, 1.4, 720, 540, 0.03, 0.42, -0.25, 0.0, 0.0), (93.6, 3.0, 720, 400, 1.8, 0.40, -0.30, 0.50, 0.1),
       (108.2, 3.0, 720, 380, 2.3, 0.42, -0.22, 0.90, 0.15), (114.8, 2.0, 720, 380, 2.3, 0.42, -0.22, 1.0, 0.45),
       (116.8, 2.0, 760, 430, 1.9, 0.40, -0.25, 0.55, 0.2), (129.2, 2.0, 760, 440, 1.8, 0.36, -0.20, 0.50, 0.1),
       (144.9, 3.0, 720, 380, 2.4, 0.45, -0.25, 0.90, 0.2), (147.7, 5.0, 720, 400, 2.6, 0.45, -0.25, 1.10, 0.8),
       (157.0, 1.8, 720, 540, 0.02, 0.45, -0.25, 0.0, 0.8)]
def galaxy_state(t):
    v = list(GAL[0][2:])
    for k in GAL:
        if t < k[0]: break
        e = float(eio((t - k[0]) / k[1])); v = [a + (b - a) * e for a, b in zip(v, k[2:])]
    phase = 0.045 * t + 2.6 * float(eio((t - 11.9) / 4.6)) + 0.9 * float(eio((t - 81.4) / 4.0)) \
            + 2.0 * float(eio((t - 92.0) / 1.4)) + 1.6 * float(eio((t - 144.9) / 12.0)) + 2.5 * float(eio((t - 157.0) / 1.8))
    field = (0.45 + 0.55 * min(1.0, v[5])) * float(cl((159.6 - t) / 2.0))
    return v, phase, field
METEORS = [(5.2, 1250, 70, -520, 230, 1.0), (23.5, 260, 110, 440, 170, 0.9), (68.6, 1260, 120, -480, 190, 1.0),
           (110.4, 1150, 80, -540, 210, 1.1), (150.4, 1280, 60, -580, 250, 1.2), (152.9, 330, 90, 400, 180, 1.0)]
EMB_W = Embers(900, 31); EMB_C = Embers(700, 37, y0=H + 30, span=H + 80)
def ember_level(t):
    warm = max(0.55 * fade(t, 114.8, 129.0, 1.5, 1.5), 1.0 * fade(t, 147.7, 157.4, 2.0, 1.5), 0.4 * fade(t, 1.9, 11.8, 1.0, 1.0))
    cool = 0.22 * float(cl((158.0 - t) / 2.0))
    return warm, cool

# ------------------------------------------------------------ 字幕
HOOK = [(0.25, 1.85, "凌晨一点，"), (1.9, 3.75, "有人叫了你的名字。")]
SUBS = [(3.9, 6.6, "是主播，在感谢你刷的礼物。"), (6.7, 9.3, "那一秒，你觉得自己〔被看见〕了。"), (9.4, 11.7, "哪怕，只花了六块钱。"),
        (14.4, 16.4, "孤独，有了价格。"), (20.9, 24.8, "一亿多个家庭，只住着一个人。"), (24.9, 27.1, "每扇窗后面，都亮着一块屏幕。"),
        (27.2, 30.2, "有人陪你聊天，按小时计费。"), (30.3, 33.6, "有人哄你入睡，按月订阅。"), (33.8, 37.4, "AI 恋人永远在线，永远温柔，"),
        (37.5, 41.2, "只要你记得〔续费〕。"), (41.3, 45.2, "我们好像什么都买得到，"), (45.3, 49.6, "却越来越难，和一个真人吃顿饭。"),
        (49.8, 53.7, "因为这门生意卖给你的，"), (53.8, 55.9, "不是陪伴，是「〔不用冒险〕」。"),
        (57.9, 62.0, "哲学家〔阿伦特〕，分清过两种状态。"), (62.1, 66.4, "一种叫寂寞：在人群里，却没人和你在一起。"),
        (66.5, 70.6, "一种叫〔独处〕：一个人，却和自己在一起。"), (70.8, 74.6, "寂寞让人害怕，所以它好卖。"),
        (74.8, 77.2, "直播间、陪聊、AI 恋人，"), (77.4, 81.2, "卖的都是同一种解药：别让你一个人待着。"),
        (81.4, 85.6, "可解药越多，我们越没机会，和自己待一会儿。"), (85.8, 89.6, "刷到凌晨，不是因为喜欢，"),
        (89.8, 93.0, "是因为害怕〔安静〕。"), (93.6, 97.6, "在阿伦特看来，两者只差一件事："),
        (108.2, 112.4, "所以，孤独本身不是病，不需要被治好。"), (112.5, 116.6, "真正被卖掉的，是你和自己相处的〔能力〕。"),
        (116.8, 120.8, "花钱买陪伴，没有错。"), (120.9, 125.0, "只是别让它，替你回答所有的安静。"), (125.1, 129.0, "你可以，〔把安静拿回来〕。"),
        (129.2, 133.4, "一个人吃饭，也不必假装很忙。"), (133.6, 136.2, "每天留〔十分钟〕，什么都不看。"),
        (136.3, 138.9, "给一个老朋友，发一条语音。"), (139.0, 141.6, "约人吃顿饭，〔允许冷场〕。"),
        (141.7, 144.8, "把「谁来陪我」，换成「〔我陪我〕」。"), (144.9, 147.6, "安静，还会再来。"), (147.7, 153.3, "但这一次，它是〔你的〕。"),
        (153.5, 157.4, "你上一次不花钱被人陪着，是什么时候？")]

_sc = {}
def rich_sprite(text, sz, base=SUBC, shadow=0.55):
    k = (text, sz, base, shadow)
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
    a = np.array(im).astype(np.float32) / 255.0
    ta = a[..., 3] ** 0.72; pad = max(6, sz // 6)
    ta = np.pad(ta, pad); rgb = np.pad(a[..., :3], ((pad, pad), (pad, pad), (0, 0)))
    sh = np.clip(cv2.GaussianBlur(ta, (0, 0), max(2.0, sz / 14)) * 1.6, 0, 1) * shadow
    A_ = ta + sh * (1 - ta)
    out = np.zeros(ta.shape + (4,), np.float32)
    out[..., :3] = rgb * (ta / np.maximum(A_, 1e-4))[..., None]; out[..., 3] = A_
    _sc[k] = out; return out

def fade(t, t0, t1, fi=0.35, fo=0.35): return float(min(cl((t - t0) / fi), cl((t1 - t) / fo)))

# ------------------------------------------------------------ 静态层
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
VIG = (1 - VIG_STRENGTH * (((xx - 720) / 720) ** 2 + ((yy - 540) / 540) ** 2) / 2)[..., None]
GR = [np.random.default_rng(100 + i).normal(0, 2.0, (H, W, 1)).astype(np.float32) for i in range(6)]
BG = np.array([6, 6, 9], np.float32)
SPOT = np.clip(1 - np.abs(xx - 720) / (40 + (yy / 760) * 190), 0, 1) * np.clip(1 - yy / 820, 0, 1) ** 0.6
SPOT = cv2.GaussianBlur(SPOT, (0, 0), 14)[..., None]
# 字幕区轻微压暗：让小字幕在银河上也清楚
SUBBAND = (1 - 0.35 * np.exp(-((yy - SUB_Y) / 46.0) ** 2))[..., None]

def W2S(p, cam):
    z, cx, cy = cam; return (np.asarray(p, np.float32) - [cx, cy]) * z + [720, 540]

def vector_layer(t, cam):
    z = cam[0]; vec = np.zeros((H, W, 3), np.uint8)
    x0, x1, ga, wm = ground(t)
    if ga > 0.01:
        a = W2S([[x0, GY], [x1, GY]], cam); colr = (FG * (1 - wm) + WARM * wm) * ga
        cv2.line(vec, tuple(int(v) for v in a[0]), tuple(int(v) for v in a[1]), tuple(int(c) for c in colr), 2, cv2.LINE_AA)
    pg = fade(t, 133.6, 145.2, 0.5, 0.6)
    if pg > 0:
        k = min(4, int((t - 133.5) / 2.7) + 1); xe = 230 + (ICX[k - 1] + 120 - 230) * float(eout((t - 133.5 - (k - 1) * 2.7) / 1.2))
        a = W2S([[230, GY + 1], [max(xe, 230), GY + 1]], cam)
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
    return vec

# ------------------------------------------------------------ 渲染（单一路径：CUDA / MPS / CPU 均可）
_G = {}
def engine_torch():
    if _E.DEV is None: gpu_init()
    return _E._TORCH

def _gpu_static():
    if _G: return _G
    engine_torch()
    _G["VIG"] = T(VIG[..., 0])[None]; _G["GR"] = [T(g[..., 0])[None] for g in GR]
    _G["BG"] = T(BG).view(3, 1, 1); _G["SPOT"] = T(SPOT[..., 0])[None]; _G["SUB"] = T(SUBBAND[..., 0])[None]
    _G["SPOTC"] = T(np.array([60, 56, 50], np.float32)).view(3, 1, 1)
    _G["SEPM"] = T(np.array([1.08, 0.94, 0.72], np.float32)).view(3, 1, 1); _G["SEPA"] = T(np.array([7, 5, 2], np.float32)).view(3, 1, 1)
    _G["GAL"] = Galaxy(); _G["spr"] = {}
    return _G

def blit_t(canvas, spr, cx, cy, alpha):
    if alpha <= 0.003: return canvas
    G = _G; key = id(spr)
    if key not in G["spr"]: G["spr"][key] = T(spr).permute(2, 0, 1).contiguous()
    s = G["spr"][key]; h, w = s.shape[1:]
    x0 = int(cx - w / 2); y0 = int(cy - h / 2)
    xa, ya = max(0, x0), max(0, y0); xb, yb = min(W, x0 + w), min(H, y0 + h)
    if xa >= xb or ya >= yb: return canvas
    ss = s[:, ya - y0:yb - y0, xa - x0:xb - x0]; a = ss[3:4] * alpha
    canvas[:, ya:yb, xa:xb] = canvas[:, ya:yb, xa:xb] * (1 - a) + ss[:3] * 255 * a
    return canvas

def actor_particles(t, cam):
    """所有前景演员 + 拖尾（多个时间偏移的加权叠加）"""
    z = cam[0]; Ps, Ws, Cs = [], [], []
    for ac in ACTORS:
        for (dt, wgt) in TRAIL:
            st = ac.state(t - dt)
            if st is None: continue
            P, al, co = st
            b = P[:, 2] * al * ac.br * (0.86 + 0.14 * np.sin(ac.tw + t * 3.1)) * wgt
            Ps.append(W2S(P[:, :2], cam)); Ws.append(b); Cs.append(co / 255.0)
    warm, cool = ember_level(t)
    for emb, lv, col in ((EMB_W, warm, (255, 200, 130)), (EMB_C, cool, (220, 224, 236))):
        if lv > 0.005:
            P, Wt, C = emb.points(t, lv * 2.2, col); Ps.append(P); Ws.append(Wt); Cs.append(C)
    P = np.concatenate(Ps); Wt = np.concatenate(Ws) * (0.75 + 0.25 * min(1, z)); C = np.concatenate(Cs)
    return P, Wt, C

def render_gpu(t, as_tensor=False):
    torch = engine_torch(); G = _gpu_static()
    cam = camera(t); z = cam[0]
    # ---- 前景粒子
    P, Wt, C = actor_particles(t, cam)
    lit = splat_t(P, Wt, C) * PARTICLE_LIGHT
    core = lit * CORE_SHARP_SHARE + gblur_t(lit, CORE_SIGMA) * (1 - CORE_SHARP_SHARE)
    glow = gblur_t(lit, SUBJECT_GLOW_SIGMA) * SUBJECT_GLOW_GAIN
    # ---- 银河
    gv, phase, field = galaxy_state(t)
    waves = []
    for (t0, wx, wy, R, warm) in WAVES:
        u = (t - t0) / 1.6
        if 0 <= u < 1: sw = W2S([wx, wy], cam); waves.append((u, float(sw[0]), float(sw[1]), R * 1.1, 1.0))
    gp, gw, gc = G["GAL"].points(t, gv[0], gv[1], gv[2], gv[3], gv[4], phase, gv[5], gv[6], waves, cam, field)
    gal = splat_t(gp, gw, gc)
    mt = meteor_pts(t, METEORS)
    if mt is not None: gal = gal + splat_t(*mt)
    gal = gal * GAL_LIGHT
    gyS = int(np.clip(W2S([0, GY], cam)[1], 0, H))      # 地平线以下的银河压暗，像夜空落在地面之后
    if gyS < H: gal[:, gyS:] *= 0.28
    galc = gblur_t(gal, 0.45)
    galt = galc + gblur_t(gal, 1.6) * GAL_GLOW + haze_t(gal, GAL_HAZE_SIGMA) * GAL_HAZE
    # ---- 矢量层（地平线 / 冲击波）
    vec = vector_layer(t, cam); vecf = T(vec).permute(2, 0, 1)
    litt = (core + glow + galt + vecf + gblur_t(vecf, GROUND_GLOW_SIGMA) * GROUND_GLOW_GAIN) * EXPOSURE
    canvas = G["BG"] + TONE_KNEE * (1 - torch.exp(-litt / TONE_KNEE))
    # 聚光灯（金句段）
    sp = fade(t, 97.9, 107.8, 1.0, 1.0)
    if sp > 0: canvas = canvas + G["SPOT"] * G["SPOTC"] * sp
    # 历史段暖褐色调（阿伦特）
    mtone = fade(t, 57.8, 66.4, 0.9, 1.2) * 0.6
    if mtone > 0:
        lum = canvas.mean(0, keepdim=True); canvas = canvas * (1 - mtone) + (lum * G["SEPM"] + G["SEPA"]) * mtone
    for (t0, d, s) in FLASH:
        if t0 <= t < t0 + d: canvas = canvas + (255 - canvas) * s * (1 - (t - t0) / d) ** 2
    if 56.0 <= t < 56.55:
        g = 1 - (t - 56.0) / 0.55; rg = np.random.default_rng(int(t * 30)); canvas = canvas.clone()
        canvas[0] = torch.roll(canvas[0], int(14 * g), 1); canvas[2] = torch.roll(canvas[2], -int(14 * g), 1)
        for _ in range(int(10 * g) + 2):
            y0 = int(rg.integers(200, 800)); hh = int(rg.integers(4, 26))
            canvas[:, y0:y0 + hh] = torch.roll(canvas[:, y0:y0 + hh], int(rg.normal(0, 90 * g)), 2)
    canvas = canvas * G["VIG"] * G["SUB"] + G["GR"][int(t * 6) % 6] * 0.35
    # ---- 文字
    for (t0, t1, s) in HOOK:
        f = fade(t, t0, t1, 0.25, 0.25)
        if f > 0: canvas = blit_t(canvas, rich_sprite(s, 84, (252, 249, 242)), 720, 200 - 10 * (1 - f), f)
    for (t0, t1, s) in SUBS:
        f = fade(t, t0, t1, 0.3, 0.3)
        if f > 0: canvas = blit_t(canvas, rich_sprite(s, SUB_SIZE), 720, SUB_Y + 6 * (1 - f), f)
    f = fade(t, 17.3, 20.4, 0.6, 0.5)
    if f > 0: canvas = blit_t(canvas, rich_sprite("第三期", 34, (210, 205, 196)), 720, 548, f)
    f = fade(t, 56.0, 57.7, 0.12, 0.4)
    if f > 0:
        k = float(eout((t - 56.0) / 0.45)); sz = int(140 + 70 * (1 - k))
        canvas = blit_t(canvas, rich_sprite("〔谁在陪你？〕", sz), 720, 450, f)
    f = fade(t, 153.9, 157.4, 0.5, 0.5)
    if f > 0: canvas = blit_t(canvas, rich_sprite("评论区聊聊 ", 26, (200, 196, 188)), 720, 1064, f)
    out = canvas.clamp(0, 255).to(torch.uint8).permute(1, 2, 0).contiguous()
    return out if as_tensor else out.cpu().numpy()

render = render_gpu   # CPU 预览也走同一路径（设备自动退回 CPU）

# ------------------------------------------------------------ 音效
def build_audio(path="sfx.wav"):
    SR_ = 44100; n = int(SR_ * DUR); tt = np.arange(n) / SR_; fx = np.zeros(n); rg = np.random.default_rng(5)
    def tone(s, f, amp, dec):
        i0 = int(s * SR_); x = tt[i0:] - s; fx[i0:] += amp * np.sin(2 * np.pi * f * x) * np.exp(-x * dec) * np.clip(x / 0.008, 0, 1)
    def whoosh(s, dur, amp, rev=False, fc=900):
        s = max(0.0, s); i0, i1 = int(s * SR_), min(n, int((s + dur) * SR_)); L = i1 - i0
        z_ = rg.normal(0, 1, L); sp = np.fft.rfft(z_); fr = np.fft.rfftfreq(L, 1 / SR_)
        sp *= np.exp(-((fr - fc) / fc) ** 2); z_ = np.fft.irfft(sp, L); z_ /= np.abs(z_).max() + 1e-9
        x = np.linspace(0, 1, L); en = (x ** 2) * np.clip((1 - x) / 0.15, 0, 1) if not rev else ((1 - x) ** 2) * np.clip(x / 0.08, 0, 1)
        fx[i0:i1] += amp * z_ * en
    def shimmer(s, dur, amp):
        for k in range(14): tone(s + k * dur / 14, 1800 + 300 * (k % 5), amp, 9)
    def bubble(s, amp=0.03):            # 礼物 / 消息提示的小“啵”
        tone(s, 1320, amp, 18); tone(s + 0.05, 1760, amp * 0.7, 20)
    for t0 in (0.0, 3.9, 20.7, 27.2, 41.3, 49.8, 57.8, 62.1, 70.8, 85.8, 93.55, 108.2, 129.2, 133.5, 144.9):
        whoosh(t0 - 0.2, 1.3, 0.15, fc=1400)
    tone(1.95, 70, 0.25, 6); bubble(1.95, 0.05)
    for k in range(12): bubble(4.4 + k * 0.62, 0.022)
    tone(9.4, 1046, 0.04, 2.5)
    whoosh(11.9, 4.6, 0.28, fc=700); tone(16.5, 45, 0.55, 1.2); tone(16.5, 440, 0.05, 1.0); whoosh(15.6, 0.9, 0.2)
    shimmer(16.5, 1.1, 0.02); whoosh(24.6, 2.2, 0.2, fc=400)
    for t0 in (27.3, 30.4, 33.9): tone(t0, 2093, 0.03, 6); tone(t0 + 0.08, 2637, 0.025, 6)   # 收银“叮”
    whoosh(37.5, 3.6, 0.18, fc=2500); tone(37.6, 1568, 0.04, 3)
    tone(43.45, 55, 0.4, 3); whoosh(43.0, 0.5, 0.15, rev=True, fc=500)
    whoosh(45.3, 1.0, 0.12, fc=1800); shimmer(52.0, 1.1, 0.02); shimmer(53.8, 1.1, 0.02)
    whoosh(55.6, 0.5, 0.2, fc=3000); tone(56.0, 50, 0.55, 2); tone(56.0, 880, 0.06, 3)
    tone(66.8, 659, 0.05, 0.9); tone(66.8, 988, 0.035, 0.9)
    for k in range(10): bubble(71.0 + k * 0.6 + 0.2 * math.sin(k), 0.018)
    for k in range(14): bubble(77.3 + k * 0.28, 0.016)
    tone(81.4, 1318, 0.03, 0.8); tone(81.4, 659, 0.03, 0.6)
    whoosh(89.9, 0.8, 0.15, rev=True, fc=600); tone(90.7, 55, 0.4, 4)
    whoosh(92.0, 1.0, 0.25); tone(93.45, 40, 0.5, 1.5); tone(93.45, 1760, 0.03, 2)
    shimmer(97.8, 1.1, 0.02); shimmer(102.6, 1.1, 0.016); whoosh(102.8, 2.2, 0.12, fc=600)
    tone(114.8, 659, 0.07, 1.2); tone(114.8, 988, 0.05, 1.2); tone(127.4, 784, 0.07, 1.0); tone(127.4, 1175, 0.05, 1.0)
    for tt0 in (133.6, 136.3, 139.0, 141.7): tone(tt0, 1046, 0.035, 3)
    tone(147.9, 523, 0.05, 0.5); tone(147.9, 784, 0.04, 0.5); tone(148.4, 1046, 0.03, 0.5)
    for t0 in (150.4, 152.9): whoosh(t0, 0.9, 0.06, fc=3200)
    tone(157.4, 1760, 0.03, 1.2)
    fx = fx / max(1e-6, np.abs(fx).max()) * 0.5; st = np.stack([fx, np.roll(fx, 220)], 1)
    with wave.open(path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR_); w.writeframes((st * 32767).astype(np.int16).tobytes())

# ------------------------------------------------------------ 编码 / 混音
def encode(frames, out, enc=None):
    import threading, queue, time
    engine_torch(); enc = enc or pick_encoder()
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-"] + enc + ["-pix_fmt", "yuv420p", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE); q = queue.Queue(maxsize=8)
    def writer():
        while True:
            b = q.get()
            if b is None: break
            p.stdin.write(b)
    th = threading.Thread(target=writer, daemon=True); th.start()
    t0 = time.time(); N = len(frames)
    for j, i in enumerate(frames):
        q.put(render_gpu(i / FPS).tobytes())
        if (j + 1) % 150 == 0 or j + 1 == N:
            el = time.time() - t0; print(f"\r  {j + 1}/{N} 帧  {el:.0f}s  剩余约 {el / (j + 1) * (N - j - 1):.0f}s", end="", flush=True)
    q.put(None); th.join(); p.stdin.close(); p.wait(); print()
    if p.returncode != 0 and enc[1] != "libx264":
        print("硬件编码失败，改用 libx264 重试"); return encode(frames, out, ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"])
    return p.returncode

def mux(video, out, bgm="bgm.mp3", sfx="sfx.wav"):
    import os
    if not os.path.exists(sfx): build_audio(sfx)
    fc = ("[2:a]volume=0.55[s];[1:a][s]amix=inputs=2:normalize=0:duration=first,atrim=0:160,"
          "afade=t=out:st=157.5:d=2.5,loudnorm=I=-14:TP=-1.5:LRA=11[a]")
    return subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", video, "-i", bgm, "-i", sfx, "-filter_complex", fc,
                           "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-t", "160", out]).returncode

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    if mode in ("pv", "gpu_pv"):
        print("设备:", engine_torch() and _E.DEV)
        for s in sys.argv[2:]: Image.fromarray(render_gpu(float(s))).save(f"pv_{s}.png")
    elif mode == "sheet":
        print("设备:", engine_torch() and _E.DEV)
        ts = [float(s) for s in sys.argv[3:]]; cells = []
        for s in ts:
            im = Image.fromarray(render_gpu(s)).resize((480, 360)); ImageDraw.Draw(im).text((6, 4), f"{s}", fill=(255, 60, 60)); cells.append(im)
        cols = 4; rows = (len(cells) + 3) // 4; sh = Image.new("RGB", (480 * cols, 360 * rows), (60, 60, 60))
        for i, c in enumerate(cells): sh.paste(c, ((i % cols) * 480, (i // cols) * 360))
        sh.save(sys.argv[2])
    elif mode == "audio": build_audio()
    elif mode in ("seg", "gpu_seg"):
        a, b, out = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
        print("设备:", engine_torch() and _E.DEV); encode(range(a, b), out); print("seg done", out)
    elif mode == "full":
        out = sys.argv[2] if len(sys.argv) > 2 else OUT_NAME
        engine_torch(); print("设备:", _E.DEV, "| 编码器:", pick_encoder()[1])
        rc = encode(range(int(DUR * FPS)), "_video.mp4")
        if rc == 0 and mux("_video.mp4", out) == 0: print("完成:", out)
        else: print("出错，返回码", rc)
    elif mode == "bench":
        import time
        engine_torch(); print("设备:", _E.DEV); render_gpu(1.0); t0 = time.time()
        for k in range(30): render_gpu(10 + k * 4.7)
        ms = (time.time() - t0) / 30
        print(f"渲染 {ms * 1000:.1f} ms/帧，全片约 {ms * DUR * FPS / 60:.1f} 分钟（不含编码）")
