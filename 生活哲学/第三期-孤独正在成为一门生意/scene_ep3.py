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
GAL_LIGHT = 230.0          # 星野亮度（退后一层，只做氛围）
GAL_GLOW = 0.12            # 小光晕
GAL_HAZE = 0.06            # 柔光
GAL_HAZE_SIGMA = 22.0
# 背景：去掉巨型旋臂银河，只保留向左缓慢流动的三层视差星野
STAR_N = 9000              # 星点数量
# 景深：环境线稿（远景/场景）单独成层，虚一点、暗一点，把视线让给人物和文字
BG_LAYER_GAIN = 0.55
BG_LAYER_SIGMA = 1.1
# 凝字：粒子汇聚后浮现完整发光字
GLYPH_FILL = 1.10          # 实体字亮度（粒子让位后由实体字撑起亮度，保持原先的高亮）
GLYPH_GLOW = 0.42          # 字外光晕
GLYPH_CORE_WHITE = 0.35    # 笔画本体向暖白靠拢的比例
GLYPH_BOLD = 1.0           # 笔画加粗（0 = 字体原粗细）
GLYPH_SHEEN = 0.55         # 定格后扫过一次的高光
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

def poly_pts(polys, sp=1.2):
    """按固定间距沿线稿取点，返回 (m,2)，点数随线长变化"""
    L = sum(float(np.hypot(*np.diff(np.asarray(p, np.float32), axis=0).T).sum()) for p in polys if len(p) >= 2)
    m = max(1, int(L / sp)); return sample_polys(polys, m, sp, jitter=0.0)[:, :2]
def polys_n(polys, n, sp=6.0): return sample_polys(polys, n, max(1.6, sp * 0.4), jitter=0.12)
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
    outline = _orig_text_pts(text, w, sz, 1.0)
    f = font(w, sz); d0 = ImageDraw.Draw(Image.new("L", (8, 8)))
    bb = d0.textbbox((0, 0), text, font=f); tw, th = bb[2] - bb[0] + 20, bb[3] - bb[1] + 20
    im = Image.new("L", (tw, th), 0); ImageDraw.Draw(im).text((10 - bb[0], 10 - bb[1]), text, font=f, fill=255)
    a = cv2.erode((np.array(im) > 128).astype(np.uint8), np.ones((2, 2), np.uint8))
    g = max(1.35, sz / 100.0) if sz >= 40 else 1.3   # 填充网格更细，粒子字接近实心
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
HERO = Actor("hero", 1000, 1); OBJ = Actor("obj", 6000, 2); OBJ2 = Actor("obj2", 3000, 3)
CROWD = Actor("crowd", 6000, 4); TXT = Actor("txt", 16000, 5); KW = Actor("kw", 13000, 7)
MIR = Actor("mir", 1000, 8)    # 倒影 / 另一个自己
SET = Actor("set", 8000, 9)    # 场景环境（窗、灯、书架、树…）
FAR = Actor("far", 4000, 10)   # 远景天际线 / 山
BUB = Actor("bub", 14000, 11)  # 漂浮物（弹幕、金币、消息气泡）
# 粒子变密后整体会变亮，按演员压一点增益；字类演员略压，保证笔画清楚不糊成一片
for _ac, _g in ((TXT, 0.5), (KW, 0.5), (BUB, 0.5), (OBJ, 0.8), (SET, 0.8), (FAR, 0.85), (CROWD, 0.85)): _ac.br *= _g
# 会快速移动的漂浮物不做拖尾，否则字会出现重影
BUB.trail = [(0.0, 1.0)]
# 价签绕圈很快，拖尾缩短，避免重影堆叠
OBJ.trail = [(0.0, 0.72), (0.033, 0.28)]; OBJ2.trail = [(0.0, 1.0)]   # OBJ2 多是细线小物件（屏幕、爱心），不拖尾才清楚
# 螺旋词在飞的时候是小字，拖尾缩短，避免重影糊字
TXT.trail = [(0.0, 0.75), (0.033, 0.25)]
ACTORS = [FAR, SET, BUB, CROWD, OBJ, OBJ2, MIR, KW, TXT, HERO]

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
KWS = [(6.7, 9.3, "被看见", 430, 330, 110, 1),
       (9.4, 11.7, "¥6", 1110, 300, 130, 1),
       (14.4, 16.4, "价格", 720, 170, 120, 1),
       (24.9, 27.1, "屏幕", 720, -150, 240, 0),
       (27.2, 30.2, "按小时", 720, 140, 100, 0),
       (30.3, 33.6, "按月", 720, 140, 110, 0),
       (33.8, 37.4, "永远在线", 720, 140, 96, 0),
       (41.3, 45.2, "什么都买得到", 720, 250, 80, 0),
       (81.4, 85.6, "待一会儿", 720, 190, 96, 1),
       (89.8, 92.3, "安静", 720, 170, 130, 1),
       (93.6, 97.6, "只差一件事", 720, 250, 90, 0),
       (144.9, 147.6, "还会再来", 1120, 200, 96, 0),
       (147.8, 153.3, "你的", 1120, 200, 140, 1),
       (20.9, 24.8, "1 亿 +", 1120, 150, 96, 0),
       (37.6, 41.1, "续费", 720, 200, 120, 1),
       (45.4, 49.5, "真人", 1100, 300, 110, 0),
       (49.9, 53.6, "不用冒险", 1110, 260, 84, 0),
       (57.9, 61.9, "Hannah Arendt", 1040, 260, 64, 1),
       (62.2, 66.3, "寂寞", 1120, 240, 120, 0),
       (66.6, 70.5, "独处", 1120, 240, 120, 1),
       (70.9, 74.5, "好卖", 720, 200, 120, 0),
       (77.5, 81.1, "解药", 720, 200, 120, 0),
       (85.9, 89.5, "01:00", 720, 170, 110, 0),
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
KWS.sort()
GLYPHS = []   # (入场开始, 入场时长, 散场开始, 文字, x, y, 字号, 颜色, 入场风格, 散场风格)
KW_IN = ["write", "gather", "write", "rise"]
for i, (t0, t1, txt, x, y, sz, wm) in enumerate(KWS):
    sin_, sout = KW_IN[i % 4], ("ash" if wm else "wind")
    KW.key(t0, S(_kw_fn(txt, x, y, sz)), 1.0, a=0.9, col=WARM if wm else FG, stag=0.55, swirl=110); KW.keys[-1]["style"] = sin_
    KW.key(t1 - 0.45, S(_kw_burst(40 + i)), 1.0, a=0.0, stag=0.6); KW.keys[-1]["style"] = sout
    GLYPHS.append((t0, 1.0, t1 - 0.45, txt, x, y, sz, WARM if wm else FG, sin_, sout, bool(wm)))

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
TXT.key(16.5, S(lambda n, t: text_shape("孤独，正在成为一门生意", HX, 440, 80, n, "L", 3)), 1.4, a=1.0, swirl=160, stag=0.5); TXT.keys[-1]["style"] = "gather"
GLYPHS.append((16.5, 1.4, 20.3, "孤独，正在成为一门生意", HX, 440, 80, FG, "gather", "wind", False))
HERO.key(16.5, S(lambda n, t, prev: burst(prev, 500, 5, HX, 470)), 1.0, a=0.0)
TXT.key(20.3, S(lambda n, t, prev: burst(prev, 380, 6)), 1.2, a=0.0, stag=0.6); TXT.keys[-1]["style"] = "wind"

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
def tag_phase(t):
    """价签公转角：37.5 秒后逐渐加速。对速度积分，角度连续，不会跳变出第二套价签"""
    r = float(cl((t - 37.5) / 3.0)); extra = 0.9 * (3.0 * r * r / 2 + max(0.0, t - 40.5))
    return 0.16 * t + extra
def tag_close(t): return 0.28 * float(eio((t - 37.5) / 3.5))
def tag_items(t, k):
    out = []
    for i in range(k):
        a = tag_phase(t) + i * 2 * math.pi / max(k, 3) + 0.4
        r = 1 - tag_close(t); x = HX + 440 * r * math.cos(a); y = 470 + 160 * r * math.sin(a) + 10 * math.sin(t * 1.3 + i)
        out.append((x, y, i))
    return out
def tag_ring(n, t, k):
    out = []; per = n // max(k, 1)
    for (x, y, i) in tag_items(t, k):
        m = per if i < k - 1 else n - per * (k - 1)
        body = poly_pts(tag_polys(x, y), 1.4)
        txt = np.concatenate([text_polys_pts(TAGS[i][0], x + 14, y - 24, 38), text_polys_pts(TAGS[i][1], x + 14, y + 26, 28)])
        out.append(spread(np.concatenate([body, txt]), m, 0.15))
    return np.concatenate(out)
# 粒子只做闪烁质感（亮度压低），清晰的边框和文字由 draw_ui 实线叠加
OBJ.key(27.2, A(lambda n, t: tag_ring(n, t, 2)), 1.3, a=0.45, swirl=140)
OBJ.key(30.3, A(lambda n, t: tag_ring(n, t, 3)), 1.0, a=0.45, swirl=80)
OBJ.key(33.8, A(lambda n, t: tag_ring(n, t, 4)), 1.0, a=0.45, swirl=80)
OBJ.key(37.5, A(lambda n, t: tag_ring(n, t, 4)), 1.0, a=0.5, col=WARM, swirl=10, stag=0.1)
TAG_T = [(27.2, 2), (30.3, 3), (33.8, 4)]
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

# ================================================================ 场景环境 / 远景 / 漂浮物（让画面不空）
def moon_polys(cx, cy, r):
    return [circle(cx, cy, r, -1.9, 1.9, 40), circle(cx + r * 0.45, cy, r * 0.82, -1.6, 1.6, 36)[::-1]]
def window_polys(cx, cy, w=220, h=260, moon=True):
    ps = [rect(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), np.array([[cx, cy - h / 2], [cx, cy + h / 2]], np.float32),
          np.array([[cx - w / 2, cy], [cx + w / 2, cy]], np.float32), np.array([[cx - w / 2 - 18, cy + h / 2 + 8], [cx + w / 2 + 18, cy + h / 2 + 8]], np.float32)]
    # 窗帘
    for sgn in (-1, 1):
        x0 = cx + sgn * (w / 2 + 6); ps.append(np.array([[x0 + sgn * 6 * math.sin(k * 0.9), cy - h / 2 - 20 + k * (h + 30) / 12] for k in range(13)], np.float32))
    if moon: ps += moon_polys(cx + w * 0.22, cy - h * 0.24, 22)
    return ps
def lamp_polys(x):
    return lines([[x, GY], [x, GY - 230]], [[x - 26, GY], [x + 26, GY]], [[x - 40, GY - 230], [x + 40, GY - 230], [x + 26, GY - 290], [x - 26, GY - 290], [x - 40, GY - 230]])
def nightstand(x):
    return lines([[x - 50, GY], [x - 50, GY - 80], [x + 50, GY - 80], [x + 50, GY]], [[x - 50, GY - 40], [x + 50, GY - 40]])
def pendant_polys(cx, cord=330):
    ps = lines([[cx, 0], [cx, cord]])
    ps.append(circle(cx, cord + 46, 46, math.pi, 2 * math.pi, 30)); ps += lines([[cx - 46, cord + 46], [cx + 46, cord + 46]])
    for k in (-1, 0, 1):   # 光锥（虚线）
        p0 = np.array([cx + k * 30, cord + 60]); p1 = np.array([cx + k * 120, cord + 200])
        for j in range(6): ps.append(np.stack([p0 + (p1 - p0) * (j / 6), p0 + (p1 - p0) * (j / 6 + 0.08)]).astype(np.float32))
    return ps
def bookshelf_polys(x0, x1, top):
    ps = [rect(x0, top, x1, GY)]; g = np.random.default_rng(int(x0))
    sh = np.linspace(top, GY, 5)
    for y in sh[1:-1]: ps.append(np.array([[x0, y], [x1, y]], np.float32))
    for a, b in zip(sh[:-1], sh[1:]):
        x = x0 + 8
        while x < x1 - 16:
            w = float(g.uniform(8, 18)); hgt = float(g.uniform(0.55, 0.9)) * (b - a)
            ps.append(rect(x, b - hgt, x + w, b)); x += w + float(g.uniform(1, 6))
    return ps
def streetlamp_polys(x, flip=False):
    d = -1 if flip else 1
    ps = lines([[x, GY], [x, GY - 330]], [[x, GY - 330], [x + d * 26, GY - 352], [x + d * 70, GY - 352]], [[x - 18, GY], [x + 18, GY]])
    ps += lines([[x + d * 52, GY - 352], [x + d * 88, GY - 352], [x + d * 80, GY - 340], [x + d * 60, GY - 340], [x + d * 52, GY - 352]])
    for k in (-1, 0, 1):
        p0 = np.array([x + d * 70 + k * 8, GY - 336]); p1 = np.array([x + d * 70 + k * 90, GY - 10])
        for j in range(8): ps.append(np.stack([p0 + (p1 - p0) * (j / 8), p0 + (p1 - p0) * (j / 8 + 0.06)]).astype(np.float32))
    return ps
def skyline_poly(x0, x1, seed, hmin=60, hmax=230, base=GY):
    g = np.random.default_rng(seed); pts = [[x0, base]]; x = x0
    while x < x1:
        w = float(g.uniform(30, 90)); h = float(g.uniform(hmin, hmax)); pts += [[x, base - h], [x + w, base - h]]
        if g.random() < 0.2: pts += [[x + w * 0.5, base - h], [x + w * 0.5, base - h - 40], [x + w * 0.5, base - h]]
        x += w
    pts.append([x, base]); return [np.array(pts, np.float32)]
def hills_polys(off=0.0):
    x = np.linspace(-200, 1640, 120)
    a = GY - 90 - 60 * np.sin(x / 260 + 0.6 + off) - 30 * np.sin(x / 97 + off * 2)
    b = GY - 40 - 40 * np.sin(x / 180 + 2.2 + off * 1.5)
    return [np.stack([x, a], 1).astype(np.float32), np.stack([x, b], 1).astype(np.float32)]
def tree_polys(cx, base, u, size=1.0):
    ps = []
    def br(p, ang, L, d):
        if d > 6: return
        grow = cl(u * 7 - d)
        if grow <= 0: return
        q = p + np.array([math.sin(ang), -math.cos(ang)]) * L * grow
        ps.append(np.stack([p, q]).astype(np.float32))
        if grow >= 1:
            br(q, ang - 0.42 - 0.05 * d, L * 0.74, d + 1); br(q, ang + 0.36 + 0.04 * d, L * 0.70, d + 1)
    br(np.array([cx, base], float), 0.0, 92 * size, 0)
    return ps
def ripple_rings(t, cx, n_r=4, rmax=520, period=3.2):
    ps = []
    for k in range(n_r):
        q = ((t / period) + k / n_r) % 1.0; r = 40 + q * rmax
        e = circle(cx, GY, r, 0, 2 * math.pi, 90); e[:, 1] = GY + (e[:, 1] - GY) * 0.12 + 6
        ps.append((e, (1 - q) ** 1.2))
    return ps

# ---- SET 时间轴
SET.key(0.0, S(lambda n, t: polys_n(window_polys(240, 330) + lamp_polys(1060) + nightstand(1060 - 110 * 0), n, 5)), 1.8, a=0.55, swirl=100)
SET.key(3.9, S(lambda n, t: polys_n(window_polys(170, 300, 200, 240) + lamp_polys(660), n, 5)), 1.2, a=0.5, swirl=80)
SET.key(11.9, S(burst_fn(320, 51)), 1.0, a=0.0)
SET.key(41.3, A(lambda n, t: polys_n(pendant_polys(360 - tbl_x(t) + 80) + pendant_polys(1080 + tbl_x(t) - 80) +
                                    [rect(640, 260, 800, 470), np.array([[720, 260], [720, 470]], np.float32)], n, 5)), 1.3, a=0.5, swirl=100)
SET.key(49.8, S(burst_fn(260, 52)), 1.0, a=0.0)
SET.key(57.8, S(lambda n, t: polys_n(bookshelf_polys(960, 1320, 360) + window_polys(250, 330, 180, 220, False), n, 5)), 1.6, a=0.5, swirl=120)
SET.key(62.1, S(lambda n, t: polys_n(streetlamp_polys(170) + streetlamp_polys(1270, True), n, 5)), 1.2, a=0.55, swirl=100)
def moon_ripples(n, t):
    m = sample_polys(moon_polys(1120, 200, 60), n // 3, 3.0)
    rs = ripple_rings(t, HX); out = [m]; per = (n - len(m)) // len(rs)
    for i, (e, w) in enumerate(rs):
        k = per if i < len(rs) - 1 else n - len(m) - per * (len(rs) - 1)
        a = sample_polys([e], k, 3.0); a[:, 2] *= w; out.append(a)
    return np.concatenate(out)
SET.key(66.5, A(moon_ripples), 1.5, a=0.6, col=WARM, swirl=80)
SET.key(70.8, S(burst_fn(260, 53)), 1.0, a=0.0)
SET.key(85.8, S(lambda n, t: polys_n(window_polys(1120, 330, 220, 260) + nightstand(1000), n, 5)), 1.2, a=0.5, swirl=100)
def stage(n, t):
    e = circle(HX, GY, 330, 0, 2 * math.pi, 120); e[:, 1] = GY + (e[:, 1] - GY) * 0.10
    e2 = circle(HX, GY, 420 + 10 * math.sin(t * 2), 0, 2 * math.pi, 140); e2[:, 1] = GY + (e2[:, 1] - GY) * 0.12
    return polys_n([e, e2], n, 4)
SET.key(93.6, A(stage), 1.6, a=0.45, col=WARM, swirl=120)
SET.key(108.2, A(lambda n, t: polys_n(hills_polys(0.02 * (t - 108)) + moon_polys(250, 210, 46), n, 5)), 1.8, a=0.45, swirl=120)
def lamps_moving(n, t):
    ps = []
    for k in range(5):
        x = ((k * 420 - (t - 129.2) * 160) % 2100) - 300
        ps += streetlamp_polys(x)
    return polys_n(ps, n, 5)
SET.key(129.2, A(lamps_moving), 1.2, a=0.5, swirl=80)
def tree_grow(n, t):
    u = 0.15 + 0.85 * float(cl((t - 133.5) / 11.0))
    return polys_n(tree_polys(150, GY, u, 0.85) + [circle(150, GY, 6, k=12)], n, 4)
SET.key(133.5, A(tree_grow), 1.0, a=0.6, col=WARM, swirl=80)
SET.key(144.9, S(lambda n, t: polys_n(hills_polys(0.3) + tree_polys(1180, GY, 1.0, 1.0), n, 5)), 1.8, a=0.5, swirl=100)
SET.key(147.7, S(lambda n, t: polys_n(hills_polys(0.3) + tree_polys(1180, GY, 1.0, 1.0), n, 5)), 2.0, a=0.6, col=WARM, stag=0.6, swirl=5)

# ---- FAR：远处天际线（很淡）
SKY = skyline_poly(-300, 1740, 77, 50, 200)
def skyline_s(a=0.0): return lambda n, t: polys_n([p + [a, 0] for p in SKY], n, 5)
FAR.key(27.2, S(skyline_s()), 1.8, a=0.22, swirl=100)
FAR.key(41.3, S(burst_fn(200, 61)), 1.0, a=0.0)
FAR.key(49.8, S(skyline_s()), 1.5, a=0.25, swirl=100)
FAR.key(56.0, S(burst_fn(400, 62, HX, 450)), 0.9, a=0.0)
FAR.key(62.1, S(skyline_s()), 1.5, a=0.25, swirl=100)
FAR.key(66.5, S(burst_fn(200, 63)), 1.0, a=0.0)
FAR.key(70.8, S(skyline_s()), 1.5, a=0.16, swirl=100)
FAR.key(85.8, S(burst_fn(200, 64)), 1.0, a=0.0)
FAR.key(129.2, A(lambda n, t: polys_n([p + [-(t - 129.2) * 30, 0] for p in skyline_poly(-300, 2400, 78, 40, 180)], n, 5)), 1.5, a=0.22, swirl=80)
FAR.key(133.4, S(burst_fn(200, 65)), 1.0, a=0.0)

# ---- BUB：直播弹幕 / 金币雨 / 消息气泡
DANMU = ["来了来了", "主播晚安", "谢谢老板的礼物", "666", "爱了爱了", "有人陪我聊聊吗", "一个人好无聊", "又失眠了",
         "刷个小心心", "今天也好累", "主播念念我的名字", "陪我到天亮", "睡不着+1", "有人吗"]
DM_SLOTS = 5
def danmu_items(t):
    """[(槽位, 文字, 中心x, y)]：每个槽位当前那条弹幕"""
    out = []; k = int((t - 2.0) / 0.55)
    for sl in range(DM_SLOTS):
        for kk in range(k, k - 14, -1):
            if kk < 0 or kk % DM_SLOTS != sl: continue
            age = t - 2.0 - kk * 0.55
            out.append((sl, DANMU[kk % len(DANMU)], W + 200 - age * 230, [78, 132, 186, 105, 159][sl])); break
    return out
def danmaku(n, t):
    per = n // DM_SLOTS; out = []; cur = {sl: (txt, x, y) for (sl, txt, x, y) in danmu_items(t)}
    for sl in range(DM_SLOTS):
        m = per if sl < DM_SLOTS - 1 else n - per * (DM_SLOTS - 1)
        if sl not in cur:
            q = np.zeros((m, 3), np.float32); q[:, 0] = W + 300; q[:, 1] = 120; out.append(q); continue
        txt, x, y = cur[sl]
        q = spread(text_pts_dense(txt, "L", 36, 2) + [x, y], m, 0.1); q[:, 2] = np.where((q[:, 0] > -260), 0.42, 0.0)
        out.append(q)
    return np.concatenate(out)
BUB.key(2.2, A(danmaku), 0.8, a=0.35, swirl=40)
BUB.key(11.9, S(burst_fn(260, 71)), 1.0, a=0.0)
_YEN = text_pts_dense("¥", "L", 58, 2)
_CG = np.random.default_rng(91)
COIN_X = np.linspace(110, 1330, 10) + _CG.uniform(-40, 40, 10); _CG.shuffle(COIN_X)
COIN_SP = _CG.uniform(90, 150, 10); COIN_PH = _CG.uniform(0, 1, 10); COIN_ROT = _CG.uniform(1.2, 2.6, 10)
def coin_items(t):
    """[(x, y, 横向缩放c, 生命周期亮度)]"""
    out = []
    for i in range(10):
        q_ = ((t - 27.2) * COIN_SP[i] / 860 + COIN_PH[i]) % 1.0
        c = abs(math.cos(COIN_ROT[i] * t + i)) * 0.88 + 0.12
        out.append((COIN_X[i], -70 + q_ * 860, c, math.sin(math.pi * q_) ** 0.6))
    return out
def coin_clear(t, x, y):
    """金币靠近价签时变淡，别挡住价签上的字"""
    if not (27.2 <= t < 41.6): return 1.0
    k = max(k_ for (t0, k_) in TAG_T if t >= t0 - 0.01) if t >= 27.2 else 2
    d = min(math.hypot((x - tx) / 1.6, y - ty) for (tx, ty, _) in tag_items(t, k))
    f = float(cl((d - 70) / 60))
    for g in GLYPHS:   # 也别挡住关键词
        if g[0] - 0.2 <= t <= g[2] + 1.0:
            tw = len(g[3]) * g[6] * 0.95
            dx = max(0.0, abs(x - g[4]) - tw / 2); dy = max(0.0, abs(y - g[5]) - g[6] * 0.6)
            f = min(f, float(cl((math.hypot(dx, dy) - 30) / 50)))
    return f
def coin_rain(n, t):
    K = 10; per = n // K; out = []
    for i, (x, y, c, life) in enumerate(coin_items(t)):
        m = per if i < K - 1 else n - per * (K - 1)
        rims = []
        for R in (44, 37):
            e = circle(x, y, R, 0, 2 * math.pi, 90); e[:, 0] = x + (e[:, 0] - x) * c; rims.append(e)
        ring = poly_pts(rims, 1.1); yen = _YEN * [c, 1] + [x, y]
        q = spread(np.concatenate([ring, yen]), m, 0.1); q[:, 2] = life * 0.55 * coin_clear(t, x, y); out.append(q)
    return np.concatenate(out)
BUB.key(27.2, A(coin_rain), 1.2, a=0.35, col=WARM, swirl=60)
BUB.key(41.3, S(burst_fn(260, 72)), 1.0, a=0.0)
NOTE = ["在吗", "晚安", "想你了", "宝宝早", "我一直在", "今天怎么样", "陪你聊聊", "别睡太晚", "抱抱", "你最特别"]
BUB_X = np.array([190, 1250, 330, 1110, 170, 1270], float)
BUB_PH = np.arange(6) / 6 + np.random.default_rng(93).uniform(-0.04, 0.04, 6)
def bubble_items(t):
    """[(x, y, 宽, 文字, 生命周期亮度)]"""
    out = []
    for i in range(6):
        q_ = ((t - 70.8) / 7.5 + BUB_PH[i]) % 1.0; y = GY - 50 - q_ * 580; x = BUB_X[i] + 16 * math.sin(t * 0.7 + i)
        txt = NOTE[(i + int((t - 70.8) / 7.5 + BUB_PH[i]) * 3) % len(NOTE)]
        out.append((x, y, 36 * len(txt) + 54, txt, math.sin(math.pi * q_) ** 0.7))
    return out
def bubble_polys(x, y, tw):
    return [rrect(x, y, tw, 66, 24), np.array([[x - tw / 2 + 24, y + 31], [x - tw / 2 + 10, y + 52], [x - tw / 2 + 44, y + 33]], np.float32)]
def note_bubbles(n, t):
    K = 6; per = n // K; out = []
    for i, (x, y, tw, txt, life) in enumerate(bubble_items(t)):
        m = per if i < K - 1 else n - per * (K - 1)
        q = spread(np.concatenate([poly_pts(bubble_polys(x, y, tw), 1.2), text_pts_dense(txt, "L", 34, 2) + [x, y]]), m, 0.1)
        q[:, 2] = life * 0.48; out.append(q)
    return np.concatenate(out)
BUB.key(70.8, A(note_bubbles), 1.4, a=0.35, swirl=80)
BUB.key(85.8, S(burst_fn(240, 73)), 1.0, a=0.0)

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
# 屏幕公转角：对速度积分（0.5 → 0.8 → 0.15 rad/s），换挡时角度连续，不会出现两套屏幕
_ST = np.arange(0, 100, 0.01)
_SV = 0.5 + 0.3 * eio((_ST - 77.2) / 1.0) - 0.65 * eio((_ST - 81.4) / 1.5)
_SPH = np.cumsum(_SV) * 0.01
def screen_phase(t): return float(np.interp(t, _ST, _SPH))
def screens(n, t, k, r0, r1, t0, dur, spd=0.5):
    u = float(eio((t - t0) / dur)); rr = r0 + (r1 - r0) * u; ps = []
    for i in range(k):
        a = screen_phase(t) + i * 2 * math.pi / k; rad = rr * (0.85 + 0.15 * math.sin(i * 2.3))
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
for ac in (HERO, OBJ, OBJ2, MIR, CROWD, SET):
    ac.key(92.4, S(lambda n, t: point(n, HX, 540, 2.0)), 1.0, a=0.6, stag=0.3, swirl=90)

# ================================================================ ACT 3  把安静拿回来
# 93.55 一点展开成两个人：他，和另一个自己
HERO.key(93.55, S(lambda n, t: person("stand", 560, GY, HH, n)), 1.4, swirl=140)
MIR.key(93.55, S(lambda n, t: person("stand", 880, GY, HH, n, flip=True)), 1.6, a=0.55, col=WARM, swirl=140)
for ac in (OBJ, OBJ2, CROWD): ac.key(93.55, hidden(ac), 0.8, a=0.0)
TXT.key(97.8, S(lambda n, t: text_shape("独处，是和自己在一起。", HX, 390, 82, n, "L", 3)), 1.6, a=1.0, col=WARM, stag=0.6, swirl=120); TXT.keys[-1]["style"] = "write"
GLYPHS.append((97.8, 1.6, 102.2, "独处，是和自己在一起。", HX, 390, 82, WARM, "write", "ash", True))
HERO.key(98.4, S(lambda n, t: person("reach", 560, GY, HH, n)), 1.2, swirl=20)
TXT.key(102.6, S(lambda n, t: text_shape("寂寞，是连自己也弄丢了。", HX, 390, 78, n, "L", 3)), 1.6, a=1.0, col=FG, stag=0.6, swirl=120); TXT.keys[-1]["style"] = "write"
GLYPHS.append((102.6, 1.6, 107.4, "寂寞，是连自己也弄丢了。", HX, 390, 78, FG, "write", "wind", False))
MIR.key(102.8, S(lambda n, t, prev: burst(prev, 340, 41, 880, 600)), 2.2, a=0.0, stag=0.7, swirl=160)
HERO.key(103.2, S(lambda n, t: person("sad", 560, GY, HH, n)), 1.2, swirl=20)
TXT.key(107.4, S(lambda n, t, prev: burst(prev, 300, 14)), 1.2, a=0.0, stag=0.6); TXT.keys[-1]["style"] = "wind"
# 108.2 坐在银河下
HERO.key(108.2, S(lambda n, t: person("lookup", HX, GY, HH, n)), 1.4, swirl=60)
MIR.key(108.4, S(lambda n, t: refl(person("lookup", HX, GY, HH, n))), 1.6, a=0.30, col=FG, swirl=60, stag=0.6)
OBJ.key(108.2, S(lambda n, t: dust_ground(n, 7, -100, 1540)), 1.3, a=0.22, swirl=60)
TXT.key(112.6, S(lambda n, t: text_shape("能力", 1130, 240, 170, n, "L", 4)), 1.3, a=1.0, col=WARM, swirl=120, stag=0.5); TXT.keys[-1]["style"] = "gather"
GLYPHS.append((112.6, 1.3, 116.2, "能力", 1130, 240, 170, WARM, "gather", "ash", True))
HERO.key(114.8, S(lambda n, t: person("lookup", HX, GY, HH, n)), 1.0, col=WARM, stag=0.7, swirl=5)
MIR.key(114.8, S(lambda n, t: refl(person("lookup", HX, GY, HH, n))), 1.0, a=0.42, col=WARM, stag=0.7, swirl=5)
TXT.key(116.2, S(lambda n, t, prev: burst(prev, 220, 15)), 1.2, a=0.0, stag=0.6); TXT.keys[-1]["style"] = "ash"
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
for ac in (HERO, OBJ, OBJ2, TXT, CROWD, MIR, KW, SET, FAR, BUB):
    ac.key(157.2, S(lambda n, t: point(n, HX, 540, 2.0)), 1.6, a=0.04, stag=0.3, swirl=50)


# ================================================================ 粒子文字的入场 / 散场风格
#  write  ：自左向右"书写"，粒子沿流动的墨迹曲线落笔
#  gather ：每个粒子绕目标点旋入，像星尘被吸进字里
#  rise   ：自下而上升起，像水汽凝结成字
#  wind   ：自左向右被风吹散成沙，向右上飘走
#  ash    ：化作暖色灰烬缓缓上升
_orig_eval = Actor._eval
def _noise2(p, t, sw):
    a = p[:, 0] * 0.011 + p[:, 1] * 0.017 + sw * 1.7 + t * 1.3
    b = p[:, 0] * 0.015 - p[:, 1] * 0.009 - sw * 1.1 + t * 1.1
    return np.stack([np.sin(a) + 0.5 * np.sin(2.3 * b), np.cos(b) + 0.5 * np.cos(1.9 * a)], 1)
def _norm(v):
    lo, hi = float(v.min()), float(v.max()); return (v - lo) / max(hi - lo, 1e-3)
def _styled_eval(self, t, i):
    k = self.keys[i]; st = k.get("style")
    if st is None: return _orig_eval(self, t, i)
    prev = k["prevP"]
    if i > 0:
        pk = self.keys[i - 1]
        if k["t"] >= pk["t"] + pk["dur"] and pk["shape"].anim: prev = pk["shape"].get(self.n, t)[pk["perm"]]
    exiting = st in ("wind", "ash")
    if exiting:
        if "_tgt" not in k:
            g = np.random.default_rng(int(k["t"] * 97) + self.n); P0 = k["prevP"].copy()
            if st == "wind":
                P0[:, 0] += g.uniform(180, 520, self.n); P0[:, 1] += -g.uniform(40, 220, self.n) + g.normal(0, 30, self.n)
            else:
                P0[:, 0] += g.normal(0, 60, self.n); P0[:, 1] -= g.uniform(160, 420, self.n)
            k["_tgt"] = P0
        cur = k["_tgt"]
    else:
        cur = k["shape"].get(self.n, t, k["prevP"])[k["perm"]]
    # 连笔判断用的参照形状：入场用目标字形；散场用原来的字形（还没飞走的部分仍保持实线）
    if exiting and i > 0:
        pk = self.keys[i - 1]
        if "inv" not in pk: pk["inv"] = np.argsort(pk["perm"])
        self._solid = (k["prevP"], pk["inv"])
    else:
        if "inv" not in k: k["inv"] = np.argsort(k["perm"])
        self._solid = (cur, k["inv"])
    if "_dly" not in k:
        ref = prev if exiting else cur; rnd = self.delay
        if st in ("write", "wind"): d = _norm(ref[:, 0]) * 0.8 + rnd * 0.2
        elif st == "rise": d = (1 - _norm(ref[:, 1])) * 0.75 + rnd * 0.25
        else: d = rnd
        k["_dly"] = d.astype(np.float32)
    u = (t - k["t"]) / k["dur"]
    if u >= 1 and not exiting:
        return cur, np.full(self.n, k["a"], np.float32), np.broadcast_to(k["col"], (self.n, 3))
    sg = k["stag"]; e = eio((u - k["_dly"] * sg) / (1 - sg)).astype(np.float32); E = e[:, None]
    if st == "write":
        P = prev + (cur - prev) * E
        P[:, :2] += _noise2(cur, t, self.sw)[:, :] * (np.sin(np.pi * e) * 70)[:, None]
    elif st == "gather":
        th = (1 - e) ** 1.5 * 4.2 * np.sign(self.sw + 1e-3); c, s_ = np.cos(th), np.sin(th)
        d = (prev[:, :2] - cur[:, :2]) * (1 - E) ** 1.2
        P = cur.copy(); P[:, 0] += d[:, 0] * c - d[:, 1] * s_; P[:, 1] += d[:, 0] * s_ + d[:, 1] * c
    elif st == "rise":
        start = cur.copy(); start[:, 1] += 160 + 80 * np.abs(self.sw); start[:, 0] += self.sw * 20
        mix = np.clip(1.6 * (1 - E), 0, 1); src = prev * (1 - mix) * 0 + (prev * 0.35 + start * 0.65)
        P = src + (cur - src) * E; P[:, :2] += _noise2(cur, t, self.sw) * (np.sin(np.pi * e) * 26)[:, None]
    else:  # wind / ash
        ee = E ** 1.4
        P = prev + (cur - prev) * ee
        P[:, :2] += _noise2(prev, t * 1.6, self.sw) * (e * (90 if st == "wind" else 50))[:, None]
    pa = k["prevA"] if np.ndim(k["prevA"]) else np.full(self.n, k["prevA"], np.float32)
    al = pa + (k["a"] - pa) * (e ** 0.7 if exiting else e ** 2.2)   # 飞行中的粒子偏暗，落定才亮，画面不脏
    pc = np.broadcast_to(k["prevC"], (self.n, 3)); tc = WARM if st == "ash" else k["col"]
    if exiting and st == "ash": co = pc + (tc - pc) * E
    else: co = pc + (k["col"] - pc) * E
    return P.astype(np.float32), al.astype(np.float32), co
Actor._eval = _styled_eval

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
    cool = 0.10 * float(cl((158.0 - t) / 2.0))
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
    _G["GAL"] = Galaxy(n_arm=0, n_core=0, n_field=STAR_N); _G["spr"] = {}
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

BG_ACTORS = (FAR, SET)
# ------------------------------------------------------------ 连笔：粒子落位后连成实线
# 每个形状的点本来就是沿笔画顺序采样的。粒子在飞时保持离散；当相邻两点都接近各自的目标、
# 且它们在目标形状里本就是笔画上的邻居时，就在两点之间补满细密的点，连成一条平滑实线。
# 能量守恒：被连起来的端点把亮度让给线段，所以线条亮度与原粒子一致，看不出一颗颗的点。
SOLID_STEP = 0.55     # 补点间距（px），越小线越顺
SOLID_THR = 2.4       # 相邻点距离 < 中位间距 × 该值，才认为是同一笔画
def _eval_keep(self, t, i):
    """与引擎 Actor._eval 相同，另外记下目标形状（用于判断笔画邻接）"""
    k = self.keys[i]
    cur = k["shape"].get(self.n, t, k["prevP"])[k["perm"]]
    if "inv" not in k: k["inv"] = np.argsort(k["perm"])
    self._solid = (cur, k["inv"])
    prev = k["prevP"]
    if i > 0:
        pk = self.keys[i - 1]
        if k["t"] >= pk["t"] + pk["dur"] and pk["shape"].anim:
            prev = pk["shape"].get(self.n, t)[pk["perm"]]
    u = (t - k["t"]) / k["dur"]
    if u >= 1:
        return cur, np.full(self.n, k["a"], np.float32), np.broadcast_to(k["col"], (self.n, 3))
    st = k["stag"]; e = eio((u - self.delay * st) / (1 - st)).astype(np.float32)
    v = cur[:, :2] - prev[:, :2]; L = np.hypot(v[:, 0], v[:, 1]) + 1e-3
    perp = np.stack([-v[:, 1], v[:, 0]], 1) / L[:, None]
    bend = (np.sin(np.pi * e) * self.sw * np.minimum(L, k["swirl"] * 3) * k["swirl"] / 200.0)[:, None]
    P = prev + (cur - prev) * e[:, None]
    P[:, :2] += perp * bend
    pa = k["prevA"] if np.ndim(k["prevA"]) else np.full(self.n, k["prevA"], np.float32)
    al = pa + (k["a"] - pa) * e
    pc = np.broadcast_to(k["prevC"], (self.n, 3))
    co = pc + (k["col"] - pc) * e[:, None]
    return P, al, co
_orig_eval = _eval_keep   # 风格化求值（_styled_eval）的兜底分支改用带记录的版本

_SOFT = []            # 本帧的柔光孤点（世界坐标）
SOFT_SIGMA = 4.5      # 柔光层模糊半径
SOFT_GAIN = 1.25      # 柔光层亮度补偿（模糊后峰值会降低）
def solidify(P, b, bl, co, cur, inv):
    """P 世界坐标点，b 粒子亮度，bl 连线用亮度（不含闪烁），返回 (点, 亮度, 颜色)"""
    Po = P[inv, :2]; To = cur[inv, :2]
    dT = np.hypot(*(To[1:] - To[:-1]).T); dP = np.hypot(*(Po[1:] - Po[:-1]).T)
    pos = dT[dT > 1e-3]
    if len(pos) < 8:
        st = np.hypot(*(P[:, :2] - cur[:, :2]).T) < 2.0
        if st.any(): _SOFT.append((P[st, :2], b[st], np.asarray(co)[st]))
        return P[~st, :2], b[~st], np.asarray(co)[~st]
    med = float(np.median(pos)); thr = float(np.clip(med * SOLID_THR, 2.5, 12.0))
    # 只有"笔画型"形状才连线：沿笔画采样的点间距几乎恒定；随机散布的云雾、浮尘间距杂乱，不连
    stroke = np.mean(np.abs(pos - med) < 0.4 * med + 0.3) > 0.55
    ok = (dT < thr) & (dP < thr * 1.25) & (dT > 1e-3) & stroke
    seg = np.nonzero(ok)[0]
    if len(seg) == 0:
        st = np.hypot(*(P[:, :2] - cur[:, :2]).T) < 2.0
        if st.any(): _SOFT.append((P[st, :2], b[st], np.asarray(co)[st]))
        return P[~st, :2], b[~st], np.asarray(co)[~st]
    bo = b[inv]; blo = bl[inv]; co_o = np.asarray(co)[inv]
    hasR = np.zeros(len(P), bool); hasL = np.zeros(len(P), bool); hasR[seg] = True; hasL[seg + 1] = True
    wpt = bo * (1 - 0.5 * hasL - 0.5 * hasR)
    K = np.maximum(1, np.ceil(dP[seg] / SOLID_STEP)).astype(np.int64)
    rs = np.repeat(seg, K); Kr = np.repeat(K, K)
    jj = np.arange(len(rs)) - np.repeat(np.cumsum(K) - K, K)
    f = ((jj + 0.5) / Kr)[:, None].astype(np.float32)
    A_ = Po[rs]; B_ = Po[rs + 1]
    lp = A_ + (B_ - A_) * f
    lw = (0.5 * (blo[rs] + blo[rs + 1]) / Kr).astype(np.float32)
    lc = co_o[rs] * (1 - f) + co_o[rs + 1] * f
    keep = wpt > 1e-5
    # 已落位却没有笔画邻居的孤点（星云、地面浮尘等）→ 交给柔光层，化成一片光雾而不是一颗颗点
    iso = keep & ~hasL & ~hasR & (np.hypot(*(Po - To).T) < 2.0)
    if iso.any():
        _SOFT.append((Po[iso], wpt[iso], co_o[iso])); keep = keep & ~iso
    return (np.concatenate([Po[keep], lp]), np.concatenate([wpt[keep], lw]), np.concatenate([co_o[keep], lc]))

def glyph_state(g, t):
    """一个凝字条目在 t 时刻的 (整体亮度, 显现方向, 显现进度)，与 draw_glyphs 共用"""
    (t0, din, tout, txt, x, y, sz, col, sin_, sout, wm) = g
    if t < t0 + din * 0.35 or t > tout + 1.2: return 0.0, None
    u = (t - t0) / din
    if sin_ == "write": a_in, rv = 1.0, (float(cl((u - 0.35) / 0.75)), "in")
    else: a_in, rv = float(ease((u - 0.45) / 0.6)), None
    if t > tout:
        v = (t - tout) / 1.0
        if sout == "wind": rv = (float(cl(v)), "out"); a_in = min(a_in, 1.0)
        else: a_in = a_in * (1 - float(ease(v / 0.7)))
    return a_in, rv
def glyph_cover(P, cur, t):
    """落在已浮现实体字里、且已落位的文字粒子，让位给实体字（否则粒子与字形会叠出毛边和双影）"""
    f = np.ones(len(P), np.float32)
    settled = np.hypot(P[:, 0] - cur[:, 0], P[:, 1] - cur[:, 1]) < 3.0
    for g in GLYPHS:
        a_in, rv = glyph_state(g, t)
        if a_in <= 0.01: continue
        txt, x, y, sz = g[3], g[4], g[5], g[6]; tw = glyph_sprite(txt, int(sz))[2]
        inb = settled & (np.abs(P[:, 0] - x) < tw / 2 + 4) & (np.abs(P[:, 1] - y) < sz * 0.8)
        if not inb.any(): continue
        a = np.full(len(P), a_in, np.float32)
        if rv is not None:
            pr, d = rv; xr = (P[:, 0] - (x - tw / 2)) / max(tw, 1)
            a = a * (cl((pr * 1.3 - xr) / 0.3) if d == "in" else cl((xr - (pr * 1.3 - 0.3)) / 0.3))
        f = np.where(inb, f * (1 - a), f)
    return f

def actor_particles(t, cam, group=None):
    """前景演员 + 拖尾（多个时间偏移的加权叠加）。group="bg" 只取环境层"""
    z = cam[0]; Ps, Ws, Cs = [], [], []
    for ac in ACTORS:
        if (group == "bg") != (ac in BG_ACTORS): continue
        for (dt, wgt) in getattr(ac, "trail", TRAIL):
            st = ac.state(t - dt)
            if st is None: continue
            P, al, co = st
            b = P[:, 2] * al * ac.br * (0.86 + 0.14 * np.sin(ac.tw + t * 3.1)) * wgt
            bl = P[:, 2] * al * 0.95 * wgt
            if ac is TXT or ac is KW:
                gc_ = glyph_cover(P, ac._solid[0], t - dt); b = b * gc_; bl = bl * gc_
            ns = len(_SOFT); q, w, c = solidify(P, b, bl, co, *ac._solid)
            for j in range(ns, len(_SOFT)):
                sp_, sw_, sc_ = _SOFT[j]; _SOFT[j] = (W2S(sp_, cam), sw_, np.asarray(sc_) / 255.0, group == "bg")
            Ps.append(W2S(q, cam)); Ws.append(w); Cs.append(np.asarray(c) / 255.0)
    warm, cool = ember_level(t)
    for emb, lv, col in ((EMB_W, warm, (255, 200, 130)), (EMB_C, cool, (220, 224, 236))):
        if lv > 0.005 and group != "bg":
            P, Wt, C = emb.points(t, lv * 2.2, col); Ps.append(P); Ws.append(Wt); Cs.append(C)
    if not Ps: return None
    P = np.concatenate(Ps); Wt = np.concatenate(Ws) * (0.75 + 0.25 * min(1, z)); C = np.concatenate(Cs)
    return P, Wt, C

# ------------------------------------------------------------ 凝字：粒子汇聚后浮现完整发光字
_gs = {}
def glyph_sprite(text, sz):
    """与 text_pts_dense 同一坐标系（图像中心 = 粒子字中心），返回 (alpha, glow) 两张灰度图"""
    k = (text, sz)
    if k in _gs: return _gs[k]
    f = font("L", sz); d0 = ImageDraw.Draw(Image.new("L", (8, 8)))
    bb = d0.textbbox((0, 0), text, font=f); tw, th = bb[2] - bb[0] + 20, bb[3] - bb[1] + 20
    pad = int(sz * 0.5) + 8
    im = Image.new("L", (tw + 2 * pad, th + 2 * pad), 0)
    ImageDraw.Draw(im).text((pad + 10 - bb[0], pad + 10 - bb[1]), text, font=f, fill=255)
    a = np.array(im).astype(np.float32) / 255.0
    if GLYPH_BOLD > 0 and sz >= 40:   # 大字略加粗：边缘仍保留抗锯齿
        a = np.clip(cv2.GaussianBlur(a, (0, 0), GLYPH_BOLD * sz / 110.0) * 1.7, 0, 1)
    gl = cv2.GaussianBlur(a, (0, 0), max(2.0, sz / 16.0)) * 0.9 + cv2.GaussianBlur(a, (0, 0), max(5.0, sz / 5.0)) * 0.6
    _gs[k] = (a, gl, tw); return _gs[k]
_gt = {}
def blit_glyph(canvas, text, cx, cy, sz, col, alpha, reveal=None, sheen=None, line=0.0):
    """加色混合（发光字）。reveal=(进度, 方向)：write 自左向右显现、wind 自左向右消散；sheen 为高光扫过的进度"""
    torch = engine_torch()
    if alpha <= 0.003 or sz < 6: return canvas
    a, gl, tw = glyph_sprite(text, int(sz))
    key = (text, int(sz))
    if key not in _gt: _gt[key] = (T(a), T(gl))
    A_, GL = _gt[key]; h, w = A_.shape
    x0 = int(round(cx - w / 2)); y0 = int(round(cy - h / 2))
    xa, ya = max(0, x0), max(0, y0); xb, yb = min(W, x0 + w), min(H, y0 + h)
    if xa >= xb or ya >= yb: return canvas
    sa = A_[ya - y0:yb - y0, xa - x0:xb - x0]; sg = GL[ya - y0:yb - y0, xa - x0:xb - x0]
    xs = (torch.arange(xa, xb, device=sa.device, dtype=torch.float32) - (cx - tw / 2)) / max(tw, 1)   # 0..1 跨字宽
    m = torch.ones_like(xs) * alpha
    if reveal is not None:
        p, mode = reveal
        if mode == "in": m = m * torch.clamp((p * 1.3 - xs) / 0.3, 0, 1)
        else: m = m * torch.clamp((xs - (p * 1.3 - 0.3)) / 0.3, 0, 1)
    lit = sa * GLYPH_FILL + sg * GLYPH_GLOW
    if sheen is not None and 0 < sheen < 1:
        ys = torch.arange(ya, yb, device=sa.device, dtype=torch.float32)[:, None]
        band = torch.exp(-(((xs[None, :] * tw + (ys - cy) * 0.5) / tw - (sheen * 1.6 - 0.3)) / 0.07) ** 2)
        lit = lit * (1 + GLYPH_SHEEN * band)
    # 笔画本体偏向暖白（像发光体的"热核"），外光晕保留原色：金色字显得通透，不会发黄发闷
    c_ = np.asarray(col, np.float32); core_c = c_ + (np.array([255, 250, 240], np.float32) - c_) * GLYPH_CORE_WHITE
    colt = T(c_).view(3, 1, 1); colc = T(core_c).view(3, 1, 1)
    canvas[:, ya:yb, xa:xb] = canvas[:, ya:yb, xa:xb] + (colc * (sa * GLYPH_FILL)[None] + colt * (sg * GLYPH_GLOW)[None]) * \
        ((lit / (sa * GLYPH_FILL + sg * GLYPH_GLOW + 1e-6)) * m[None, :])[None]
    if line > 0:   # 暖色关键词下方的一道细金线
        ly = int(cy + sz * 0.62); hw = int(tw * 0.32 * float(eout(line)))
        if 0 <= ly < H - 2 and hw > 1:
            lx0, lx1 = max(0, int(cx) - hw), min(W, int(cx) + hw)
            xx_ = torch.arange(lx0, lx1, device=sa.device, dtype=torch.float32)
            fall = torch.clamp(1 - torch.abs(xx_ - cx) / max(hw, 1), 0, 1) ** 0.6 * alpha * 150
            canvas[:, ly:ly + 2, lx0:lx1] = canvas[:, ly:ly + 2, lx0:lx1] + colt / 255 * fall[None, None, :]
    return canvas


# ------------------------------------------------------------ 清晰叠加层：价签 / 弹幕 / 金币 / 气泡
# 小号文字和细边框只靠粒子很难清楚（粒子会稀、会抖、会拖影）。这里用实线和实体字叠一层，
# 粒子降为闪烁质感；进场时实线比粒子晚一步浮现，看起来仍是"粒子凝结成形"。
UI_LINE = 0.85     # 实线亮度
UI_TEXT = 0.95     # 实体小字亮度
def _poly_cv(layer, polys, cam, col, a, th=1):
    if a <= 0.01: return
    c = tuple(float(v) * a for v in col)
    for p in polys:
        q = W2S(np.asarray(p, np.float32), cam); q = np.round(q * 4).astype(np.int32)
        cv2.polylines(layer, [q.reshape(-1, 1, 2)], False, c, th, cv2.LINE_AA, shift=2)
def draw_ui(canvas, t, cam):
    torch = engine_torch(); z = cam[0]; texts = []
    layer = np.zeros((H, W, 3), np.float32); used = False
    # 价签 27.2–41.3
    ft = fade(t, 27.9, 41.6, 0.8, 0.5)
    if ft > 0:
        k = max(k_ for (t0, k_) in TAG_T if t >= t0 - 0.01) if t >= 27.2 else 2
        wc = float(cl((t - 37.5) / 1.2)); col = FG * (1 - wc) + WARM * wc
        for (x, y, i) in tag_items(t, k):
            born = [t0 for (t0, k_) in TAG_T if k_ > i][0]
            a = ft * float(ease((t - born - 0.7) / 0.8))
            if a <= 0.01: continue
            _poly_cv(layer, tag_polys(x, y), cam, col * UI_LINE, a, 2); used = True
            texts += [(TAGS[i][0], x + 14, y - 24, 38, col, a), (TAGS[i][1], x + 14, y + 26, 28, col, a * 0.9)]
    # 弹幕 2.2–11.7
    fd = fade(t, 2.6, 11.8, 0.6, 0.4)
    if fd > 0:
        for (sl, txt, x, y) in danmu_items(t): texts.append((txt, x, y, 36, FG, fd * 0.6))
    # 金币 27.2–41.3
    fc = fade(t, 27.8, 41.4, 0.8, 0.4)
    if fc > 0:
        for (x, y, c, life) in coin_items(t):
            a = fc * life * 0.75 * coin_clear(t, x, y)
            for R in (44, 37):
                e = circle(x, y, R, 0, 2 * math.pi, 90); e[:, 0] = x + (e[:, 0] - x) * c
                _poly_cv(layer, [e], cam, WARM * UI_LINE, a, 2)
            used = True
            if c > 0.45: texts.append(("¥", x, y, 58, WARM, a * float(cl((c - 0.45) / 0.35))))
    # 消息气泡 70.8–85.6
    fb = fade(t, 71.6, 85.7, 0.8, 0.4)
    if fb > 0:
        for (x, y, tw, txt, life) in bubble_items(t):
            a = fb * life * 0.8
            _poly_cv(layer, bubble_polys(x, y, tw), cam, FG * UI_LINE, a, 2); used = True
            texts.append((txt, x, y, 34, FG, a))
    if used:
        L = T(layer).permute(2, 0, 1); canvas = canvas + L + gblur_t(L, 2.2) * 0.6
    for (txt, x, y, sz, col, a) in texts:
        sc = W2S([x, y], cam)
        canvas = blit_glyph(canvas, txt, float(sc[0]), float(sc[1]), max(8, int(round(sz * z / 2) * 2)), col, a * UI_TEXT)
    return canvas

def draw_glyphs(canvas, t, cam):
    z = cam[0]
    for (t0, din, tout, txt, x, y, sz, col, sin_, sout, wm) in GLYPHS:
        a_in, rv = glyph_state((t0, din, tout, txt, x, y, sz, col, sin_, sout, wm), t)
        if a_in <= 0.003: continue
        sc = W2S([x, y], cam); zs = max(8, int(round(sz * z / 2) * 2))
        sheen = (t - (t0 + din + 0.3)) / 1.4
        canvas = blit_glyph(canvas, txt, float(sc[0]), float(sc[1]), zs, col, a_in, rv, sheen,
                            line=float(cl((t - t0 - din) / 0.8)) * (1 - float(cl((t - tout) / 0.4))) if wm else 0.0)
    return canvas

def render_gpu(t, as_tensor=False):
    torch = engine_torch(); G = _gpu_static()
    cam = camera(t); z = cam[0]
    # ---- 前景粒子
    _SOFT.clear()
    P, Wt, C = actor_particles(t, cam)
    lit = splat_t(P, Wt, C) * PARTICLE_LIGHT
    core = lit * CORE_SHARP_SHARE + gblur_t(lit, CORE_SIGMA) * (1 - CORE_SHARP_SHARE)
    glow = gblur_t(lit, SUBJECT_GLOW_SIGMA) * SUBJECT_GLOW_GAIN
    bgp = actor_particles(t, cam, "bg")
    if bgp is not None:   # 环境层：轻微失焦 + 压暗，形成景深
        core = core + gblur_t(splat_t(*bgp) * PARTICLE_LIGHT, BG_LAYER_SIGMA) * BG_LAYER_GAIN
    if _SOFT:
        sp_ = np.concatenate([q[0] for q in _SOFT]); sc_ = np.concatenate([q[2] for q in _SOFT])
        sw_ = np.concatenate([q[1] * (BG_LAYER_GAIN if q[3] else 1.0) for q in _SOFT])
        core = core + gblur_t(splat_t(sp_.astype(np.float32), sw_.astype(np.float32), sc_.astype(np.float32)) * PARTICLE_LIGHT, SOFT_SIGMA) * SOFT_GAIN
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
    canvas = draw_ui(canvas, t, cam)
    canvas = draw_glyphs(canvas, t, cam)
    canvas = canvas * G["VIG"] * G["SUB"] + G["GR"][int(t * 6) % 6] * 0.28
    # ---- 文字
    for (t0, t1, s) in HOOK:
        f = fade(t, t0, t1, 0.25, 0.25)
        if t0 - 0.05 <= t <= t1 + 0.6:
            rv = (float(cl((t - t0) / 0.9)), "in") if t < t1 else (float(cl((t - t1) / 0.6)), "out")
            canvas = blit_glyph(canvas, s, 720, 200, 84, (252, 249, 242), 1.0, rv, (t - t0 - 0.9) / 1.2)
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
    if f > 0: canvas = blit_t(canvas, rich_sprite("评论区聊聊", 26, (200, 196, 188)), 720, 1064, f)
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
