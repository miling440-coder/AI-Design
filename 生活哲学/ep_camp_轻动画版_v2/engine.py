# -*- coding: utf-8 -*-
# 《生活智慧》系列 · 电影感渲染引擎（PyTorch GPU）
# 3D 自然场景用逐像素着色（天空/山/湖/草/星空/雾/体积光），2D 叙事元素用矢量蒙版，
# 统一在 GPU 上合成、泛光、调色、颗粒、字幕，再用 NVENC 编码。
import os, math, json, subprocess, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

DW, DH = 1440, 1080                      # 设计坐标（所有场景都按 1440×1080 写）
SCALE = float(os.environ.get("EP_SCALE", "1"))   # 预览可用 0.5 加速
W, H = int(round(DW * SCALE / 2)) * 2, int(round(DH * SCALE / 2)) * 2
PX = W / DW                              # 设计单位 -> 像素
FPS = 30
FONT_PATH = "font.otf"

torch = None; F = None; DEV = None
X = Y = None                              # (H,W) 设计坐标网格

def init(prefer=None):
    global torch, F, DEV, X, Y
    import torch as _t
    torch = _t; F = _t.nn.functional
    want = prefer or os.environ.get("EP_DEVICE")
    if want: DEV = torch.device(want)
    elif torch.cuda.is_available(): DEV = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available(): DEV = torch.device("mps")
    else: DEV = torch.device("cpu")
    torch.set_grad_enabled(False)
    if DEV.type == "cuda":
        torch.backends.cudnn.benchmark = True
    xs = (torch.arange(W, device=DEV, dtype=torch.float32) + 0.5) / PX
    ys = (torch.arange(H, device=DEV, dtype=torch.float32) + 0.5) / PX
    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    return DEV

# ------------------------------------------------------------------ 数学
def cl(x, a=0.0, b=1.0):
    if isinstance(x, (int, float)): return min(max(x, a), b)
    return x.clamp(a, b)
def sm(e0, e1, x):
    if isinstance(x, (int, float)):
        u = min(max((x - e0) / (e1 - e0 + 1e-9), 0.0), 1.0); return u * u * (3 - 2 * u)
    u = ((x - e0) / (e1 - e0 + 1e-9)).clamp(0, 1); return u * u * (3 - 2 * u)
def ease(u): u = cl(u); return u * u * (3 - 2 * u)
def eio(u):
    u = cl(u); return 4 * u ** 3 if u < .5 else 1 - (-2 * u + 2) ** 3 / 2
def eout(u): u = cl(u); return 1 - (1 - u) ** 3
def ein(u): u = cl(u); return u ** 3
def lerp(a, b, u): return a + (b - a) * u
def win(t, a, b, fi=0.5, fo=0.5):
    """时间窗包络：a 起淡入 fi 秒，b 前淡出 fo 秒"""
    return ease((t - a) / max(fi, 1e-6)) * ease((b - t) / max(fo, 1e-6))

def kf(t, keys, fn=eio):
    """关键帧插值：keys=[(t, 值或元组)]，缓入缓出"""
    if t <= keys[0][0]: return keys[0][1]
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if t < t1:
            u = fn((t - t0) / (t1 - t0))
            if isinstance(v0, (tuple, list)): return tuple(lerp(a, b, u) for a, b in zip(v0, v1))
            return lerp(v0, v1, u)
    return keys[-1][1]

def kf_spline(t, keys):
    """单调三次样条插值（Fritsch–Carlson）：经过每个关键帧但不在关键帧处“刹停”，
    相邻两段同向运动时镜头速度连续（消除走走停停的卡顿感）；值不变的保持段仍然静止。"""
    if t <= keys[0][0]: return keys[0][1]
    if t >= keys[-1][0]: return keys[-1][1]
    ts = [k[0] for k in keys]; vs = [k[1] if isinstance(k[1], (tuple, list)) else (k[1],) for k in keys]
    i = max(0, min(len(ts) - 2, next(j for j in range(len(ts) - 1) if t < ts[j + 1])))
    h = ts[i + 1] - ts[i]; u = (t - ts[i]) / h
    out = []
    for c in range(len(vs[0])):
        def slope(j): return (vs[j + 1][c] - vs[j][c]) / (ts[j + 1] - ts[j])
        def tan(j):
            if j == 0 or j == len(ts) - 1: return 0.0
            a, b = slope(j - 1), slope(j)
            if a * b <= 0: return 0.0
            w1 = 2 * (ts[j + 1] - ts[j]) + (ts[j] - ts[j - 1]); w2 = (ts[j + 1] - ts[j]) + 2 * (ts[j] - ts[j - 1])
            return (w1 + w2) / (w1 / a + w2 / b)
        p0, p1 = vs[i][c], vs[i + 1][c]; m0, m1 = tan(i) * h, tan(i + 1) * h
        h00 = 2 * u ** 3 - 3 * u ** 2 + 1; h10 = u ** 3 - 2 * u ** 2 + u; h01 = -2 * u ** 3 + 3 * u ** 2; h11 = u ** 3 - u ** 2
        out.append(h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1)
    return tuple(out) if isinstance(keys[0][1], (tuple, list)) else out[0]

def kf_dict(t, keys, fn=ease):
    if t <= keys[0][0]: return dict(keys[0][1])
    for (t0, d0), (t1, d1) in zip(keys, keys[1:]):
        if t < t1:
            u = fn((t - t0) / (t1 - t0)); out = {}
            for k in d0:
                a, b = d0[k], d1.get(k, d0[k])
                out[k] = tuple(lerp(x, y, u) for x, y in zip(a, b)) if isinstance(a, tuple) else lerp(a, b, u)
            return out
    return dict(keys[-1][1])

def C(rgb):
    """颜色 -> (3,1,1) 张量，输入 0..1 或 0..255"""
    r = [v / 255.0 for v in rgb] if max(rgb) > 1.5 else list(rgb)
    return torch.tensor(r, device=DEV, dtype=torch.float32).view(3, 1, 1)

def mix(a, b, u):
    return a + (b - a) * u

# ------------------------------------------------------------------ 噪声（整数哈希，GPU 上可复现）
M32 = 0xFFFFFFFF
def _s32(v):
    v &= M32
    return v - (1 << 32) if v >= (1 << 31) else v
# 32 位整数哈希：结果与旧版 64 位写法逐位一致，但在 GTX 16 系 / 消费级显卡上快得多
# （int64 乘法在这些卡上要拆成多条指令）。设置 EP_HASH64=1 可退回旧实现做对比。
_HASH64 = os.environ.get("EP_HASH64", "0") == "1"
def ihash(ix, iy, seed=0):
    if _HASH64:
        h = (ix.long() * 374761393 + iy.long() * 668265263 + seed * 1442695041 + 1013904223) & M32
        h = ((h ^ (h >> 13)) * 1274126177) & M32
        h = h ^ (h >> 16)
        return (h & 0xFFFFFF).float() * (1.0 / 16777216.0)
    ix = ix.to(torch.int32); iy = iy.to(torch.int32)
    h = ix * 374761393 + iy * 668265263 + _s32(seed * 1442695041 + 1013904223)   # int32 回绕 = mod 2^32
    h = (h ^ ((h >> 13) & 0x7FFFF)) * 1274126177                                  # 逻辑右移
    h = h ^ ((h >> 16) & 0xFFFF)
    return (h & 0xFFFFFF).float() * (1.0 / 16777216.0)

def vnoise(x, y, seed=0):
    xf = torch.floor(x); yf = torch.floor(y)
    fx = x - xf; fy = y - yf
    ix = xf.int(); iy = yf.int()
    ux = fx * fx * (3 - 2 * fx); uy = fy * fy * (3 - 2 * fy)
    a = ihash(ix, iy, seed); b = ihash(ix + 1, iy, seed)
    c = ihash(ix, iy + 1, seed); d = ihash(ix + 1, iy + 1, seed)
    return (a + (b - a) * ux) + ((c + (d - c) * ux) - (a + (b - a) * ux)) * uy

def fbm(x, y, oct=5, seed=0, lac=2.03, gain=0.5):
    s = 0.0; a = 0.5; norm = 0.0
    for i in range(oct):
        s = s + a * vnoise(x, y, seed + i * 17); norm += a
        x = x * lac + 3.1; y = y * lac + 1.7; a *= gain
    return s / norm

def noise1(x, seed=0):
    xf = torch.floor(x); f = x - xf; i = xf.int()
    u = f * f * (3 - 2 * f)
    a = ihash(i, i * 0 + 7, seed); b = ihash(i + 1, i * 0 + 7, seed)
    return a + (b - a) * u

def fbm1(x, oct=5, seed=0, gain=0.5):
    s = 0.0; a = 0.5; n = 0.0
    for i in range(oct):
        s = s + a * noise1(x, seed + i * 31); n += a; x = x * 2.07 + 5.3; a *= gain
    return s / n

def hash1(i, seed=0):
    """numpy 端的确定性哈希（CPU 侧布置物件用）"""
    i = np.asarray(i, np.int64)
    h = (i * 374761393 + seed * 668265263 + 1013904223) & M32
    h = ((h ^ (h >> 13)) * 1274126177) & M32
    h = h ^ (h >> 16)
    return (h & 0xFFFFFF) / 16777216.0

# ------------------------------------------------------------------ 矢量蒙版（CPU 画，GPU 合成）
SH = 3; SHM = 1 << SH
class Mask:
    def __init__(self):
        self.a = np.zeros((H, W), np.uint8)
    def _p(self, pts):
        return np.round(np.asarray(pts, np.float64) * PX * SHM).astype(np.int32).reshape(-1, 1, 2)
    def poly(self, pts, v=255):
        cv2.fillPoly(self.a, [self._p(pts)], int(v), cv2.LINE_AA, SH); return self
    def polys(self, ps, v=255):
        if ps: cv2.fillPoly(self.a, [self._p(p) for p in ps], int(v), cv2.LINE_AA, SH)
        return self
    def line(self, pts, w, v=255, closed=False):
        cv2.polylines(self.a, [self._p(pts)], closed, int(v), max(1, int(round(w * PX))), cv2.LINE_AA, SH); return self
    def lines(self, ps, w, v=255):
        if ps: cv2.polylines(self.a, [self._p(p) for p in ps], False, int(v), max(1, int(round(w * PX))), cv2.LINE_AA, SH)
        return self
    def circle(self, c, r, v=255, w=-1):
        cc = tuple(int(round(q * PX * SHM)) for q in c)
        cv2.circle(self.a, cc, max(1, int(round(r * PX * SHM))), int(v), -1 if w < 0 else max(1, int(w * PX)), cv2.LINE_AA, SH); return self
    def ellipse(self, c, ax, ang=0, v=255, a0=0, a1=360, w=-1):
        cc = tuple(int(round(q * PX * SHM)) for q in c); aa = tuple(max(1, int(round(q * PX * SHM))) for q in ax)
        cv2.ellipse(self.a, cc, aa, ang, a0, a1, int(v), -1 if w < 0 else max(1, int(w * PX)), cv2.LINE_AA, SH); return self
    def rect(self, x0, y0, x1, y1, v=255):
        return self.poly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], v)
    def rrect(self, x0, y0, x1, y1, r, v=255):
        r = min(r, (x1 - x0) / 2, (y1 - y0) / 2); k = 8; pts = []
        for cx, cy, a0 in ((x1 - r, y0 + r, -90), (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180)):
            for j in range(k + 1):
                a = math.radians(a0 + 90 * j / k); pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        return self.poly(pts, v)
    def t(self, blur=0.0):
        m = torch.from_numpy(self.a).to(DEV).float().mul_(1 / 255.0)
        if blur > 0: m = gblur(m[None], blur * PX)[0]
        return m

def over(img, col, a):
    """img(3,H,W) 上叠加颜色 col（(3,1,1) 或 (3,H,W)），alpha a（(H,W)）"""
    if not torch.is_tensor(a): a = torch.tensor(float(a), device=DEV)
    return img + (col - img) * a.unsqueeze(0) if a.dim() == 2 else img + (col - img) * a

def add(img, col, a):
    return img + col * (a.unsqueeze(0) if torch.is_tensor(a) and a.dim() == 2 else a)

# ------------------------------------------------------------------ 模糊 / 泛光
_gk = {}
def gblur(img, sigma):
    """可分离高斯模糊，img (C,H,W)，sigma 为像素"""
    if sigma <= 0.05: return img
    key = round(float(sigma), 3)
    if key not in _gk:
        r = max(1, int(math.ceil(sigma * 3)))
        xs = torch.arange(-r, r + 1, dtype=torch.float32, device=DEV)
        k = torch.exp(-xs ** 2 / (2 * sigma * sigma)); _gk[key] = (k / k.sum(), r)
    k, r = _gk[key]; c = img.shape[0]; x = img[None]
    pm = "reflect" if (r < img.shape[1] and r < img.shape[2]) else "replicate"
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode=pm), k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode=pm), k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)
    return x[0]

def haze(img, sigma, down=4):
    """大半径柔光：缩小-模糊-放大"""
    down = max(1, int(round(down * PX))) if PX > 0.6 else max(1, int(down / 2))
    x = F.avg_pool2d(img[None], down)
    x = gblur(x[0], sigma * PX / down)[None]
    return F.interpolate(x, size=img.shape[1:], mode="bilinear", align_corners=False)[0]

def dir_blur(img, dx, dy, n=12):
    """方向模糊（甩镜/运动），dx,dy 为设计单位总长度"""
    if abs(dx) + abs(dy) < 0.5: return img
    acc = torch.zeros_like(img)
    for i in range(n):
        s = (i / (n - 1) - 0.5)
        acc += shift(img, dx * s, dy * s)
    return acc / n

def shift(img, dx, dy):
    """整数像素平移（边缘复制）；对很窄的图块（高或宽小于平移量）也安全"""
    ix = int(round(dx * PX)); iy = int(round(dy * PX))
    if ix == 0 and iy == 0: return img
    Hh, Ww = img.shape[1], img.shape[2]
    ix = max(-Ww, min(Ww, ix)); iy = max(-Hh, min(Hh, iy))
    out = torch.roll(img, (iy, ix), (1, 2))
    if ix > 0: out[:, :, :ix] = out[:, :, ix:ix + 1] if ix < Ww else img[:, :, :1]
    elif ix < 0: out[:, :, ix:] = out[:, :, ix - 1:ix] if -ix < Ww else img[:, :, -1:]
    if iy > 0: out[:, :iy, :] = out[:, iy:iy + 1, :] if iy < Hh else out[:, :1, :]
    elif iy < 0: out[:, iy:, :] = out[:, iy - 1:iy, :] if -iy < Hh else out[:, -1:, :]
    return out

def warp_rect(src, x0, y0, x1, y1):
    """把整帧 src 缩放放进屏幕矩形 [x0,x1]×[y0,y1]（设计单位），返回 (img, mask)"""
    u = (X - x0) / max(x1 - x0, 1e-3) * 2 - 1; v = (Y - y0) / max(y1 - y0, 1e-3) * 2 - 1
    g = torch.stack([u, v], -1)[None]
    out = F.grid_sample(src[None], g, mode="bilinear", padding_mode="border", align_corners=False)[0]
    m = ((u.abs() <= 1) & (v.abs() <= 1)).float()
    return out, m

def zoom_about(src, z, cx, cy, ox=0.0, oy=0.0):
    """以 (cx,cy) 为中心放大 z 倍并平移（设计单位）"""
    u = ((X - DW / 2 - ox) / z + cx) / DW * 2 - 1
    v = ((Y - DH / 2 - oy) / z + cy) / DH * 2 - 1
    g = torch.stack([u, v], -1)[None]
    return F.grid_sample(src[None], g, mode="bilinear", padding_mode="border", align_corners=False)[0]

# ------------------------------------------------------------------ 粒子
def splat(P, Wt, Cc):
    """P (N,2) 设计坐标；Wt (N,)；Cc (N,3) 0..1 —— 双线性溅射到 (3,H,W)"""
    P = torch.as_tensor(P, dtype=torch.float32, device=DEV) * PX
    Wt = torch.as_tensor(Wt, dtype=torch.float32, device=DEV)
    Cc = torch.as_tensor(np.array(Cc, np.float32) if not torch.is_tensor(Cc) else Cc, dtype=torch.float32, device=DEV)
    if Cc.dim() == 1: Cc = Cc.view(1, 3).expand(len(P), 3)
    x, y = P[:, 0], P[:, 1]
    m = (x > 1) & (x < W - 2) & (y > 1) & (y < H - 2) & (Wt > 1e-4)
    out = torch.zeros(3, H * W, device=DEV)
    if not bool(m.any()): return out.view(3, H, W)
    x, y, Wt, Cc = x[m], y[m], Wt[m], Cc[m]
    x0 = torch.floor(x); y0 = torch.floor(y); fx = x - x0; fy = y - y0
    base = y0.long() * W + x0.long()
    idx = torch.cat([base, base + 1, base + W, base + W + 1])
    w = torch.cat([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy]) * Wt.repeat(4)
    out.index_add_(1, idx, w[None, :] * Cc.t().repeat(1, 4))
    return out.view(3, H, W)

def glow_points(P, Wt, Cc, sigma=2.0, core=0.6):
    s = splat(P, Wt, Cc)
    return s * core + gblur(s, sigma * PX) * (1 - core) * (sigma * PX) ** 2 * 2.0

# ------------------------------------------------------------------ 文字
_fc = {}
def font(sz):
    k = int(sz)
    if k not in _fc: _fc[k] = ImageFont.truetype(FONT_PATH, k)
    return _fc[k]

_tc = {}
def text_alpha(text, sz, spacing=0):
    """返回文字 alpha（numpy float, 像素尺度）与其设计尺寸"""
    k = (text, int(sz), spacing, W)
    if k in _tc: return _tc[k]
    f = font(max(6, int(round(sz * PX))))
    d0 = ImageDraw.Draw(Image.new("L", (8, 8)))
    if spacing:
        ws = [d0.textlength(ch, font=f) for ch in text]; tw = int(sum(ws) + spacing * PX * (len(text) - 1)) + 8
    else:
        tw = int(d0.textlength(text, font=f)) + 8
    asc, desc = f.getmetrics(); th = asc + desc + 8
    im = Image.new("L", (tw, th), 0); dr = ImageDraw.Draw(im)
    if spacing:
        x = 4
        for ch, wch in zip(text, ws): dr.text((x, 4), ch, font=f, fill=255); x += wch + spacing * PX
    else:
        dr.text((4, 4), text, font=f, fill=255)
    a = np.asarray(im, np.float32) / 255.0
    _tc[k] = a; return a

def draw_text(img, text, cx, cy, sz, col, alpha=1.0, glow=0.0, shadow=0.0, reveal=1.0, blur=0.0, spacing=0, anchor="c", rise=0.0):
    """在 (cx,cy) 绘制文字。reveal: 0..1 由左至右柔和显现；blur: 失焦程度（像素）"""
    if alpha <= 0.003: return img
    a = text_alpha(text, sz, spacing)
    h, w = a.shape
    if anchor == "c": x0 = int(round(cx * PX - w / 2)); y0 = int(round((cy + rise) * PX - h / 2))
    elif anchor == "l": x0 = int(round(cx * PX)); y0 = int(round((cy + rise) * PX - h / 2))
    else: x0 = int(round(cx * PX - w)); y0 = int(round((cy + rise) * PX - h / 2))
    pad = int(max(blur, glow * 8, shadow * 6) * PX + 4)
    X0, Y0 = max(0, x0 - pad), max(0, y0 - pad); X1, Y1 = min(W, x0 + w + pad), min(H, y0 + h + pad)
    if X1 <= X0 or Y1 <= Y0: return img
    full = np.zeros((Y1 - Y0, X1 - X0), np.float32)
    sx0, sy0 = max(0, X0 - x0 + 0), max(0, Y0 - y0)
    dx0, dy0 = max(0, x0 - X0), max(0, y0 - Y0)
    hh = min(h - sy0, full.shape[0] - dy0); ww = min(w - sx0, full.shape[1] - dx0)
    if hh <= 0 or ww <= 0: return img
    full[dy0:dy0 + hh, dx0:dx0 + ww] = a[sy0:sy0 + hh, sx0:sx0 + ww]
    m = torch.from_numpy(full).to(DEV)
    if reveal < 1.0:
        xs = torch.arange(m.shape[1], device=DEV, dtype=torch.float32)
        edge = (dx0 + w * reveal * 1.15) - 0.15 * w
        m = m * ((edge - xs) / (0.15 * w + 1)).clamp(0, 1)[None, :]
    if blur > 0.05: m = gblur(m[None], blur * PX)[0]
    reg = img[:, Y0:Y1, X0:X1]
    if shadow > 0:
        sh = gblur(m[None], 3.0 * PX)[0] * shadow
        reg = reg * (1 - sh.clamp(0, 1) * alpha * 0.8)
    if glow > 0:
        g = gblur(m[None], 7.0 * PX)[0]
        reg = reg + col * (g * glow * alpha)[None]
    reg = reg + (col - reg) * (m * alpha)[None].clamp(0, 1)
    img[:, Y0:Y1, X0:X1] = reg
    return img

# ------------------------------------------------------------------ 后期
_dither = None
GRAIN = float(os.environ.get("EP_GRAIN", "0"))   # 颗粒默认关闭，画面保持干净；想要一点胶片感可设 0.006
def post(img, t, frame, bloom=0.35, bloom_th=0.75, vig=0.25, grain=0.0, lift=(0.0, 0.0, 0.0), sat=1.0, expo=1.0):
    global _dither
    img = img * expo
    # 泛光：亮部提取，两级柔光
    br = (img - bloom_th).clamp(min=0)
    b = haze(br, 10, 4) * 0.6 + haze(br, 36, 8) * 0.4
    img = img + b * bloom
    # 饱和度
    if sat != 1.0:
        l = (img * C((0.299, 0.587, 0.114))).sum(0, keepdim=True)
        img = l + (img - l) * sat
    # 胶片肩部（柔和高光滚降）
    img = img.clamp(min=0)
    img = torch.where(img < 0.8, img, 0.8 + 0.2 * (1 - torch.exp(-(img - 0.8) / 0.2)))
    # 暗部抬升
    img = img + C(lift) * (1 - img).clamp(0, 1)
    # 暗角
    d = ((X - DW / 2) / (DW * 0.62)) ** 2 + ((Y - DH / 2) / (DH * 0.62)) ** 2
    img = img * (1 - vig * d.clamp(0, 1.6))[None]
    # 颗粒（默认关闭）
    g_amt = GRAIN
    if g_amt > 0:
        g = torch.Generator(device="cpu"); g.manual_seed(1000 + frame)
        n = torch.randn((1, H // 2, W // 2), generator=g).to(DEV)
        n = F.interpolate(n[None], size=(H, W), mode="bilinear", align_corners=False)[0]
        img = img + n * g_amt
    # 静态抖动（±0.5/255，固定图案、不随帧闪烁）：只为消除暗部渐变色带，肉眼不可见
    if _dither is None:
        g = torch.Generator(device="cpu"); g.manual_seed(7)
        _dither = ((torch.rand((1, H, W), generator=g) + torch.rand((1, H, W), generator=g) - 1.0) / 255.0 * 0.5).to(DEV)
    img = img + _dither
    return img.clamp(0, 1)

def letterbox(img, bar=64):
    if bar <= 0: return img
    m = ((Y > bar) & (Y < DH - bar)).float()
    return img * m[None]

def to_u8(img):
    return (img.clamp(0, 1) * 255 + 0.5).byte().permute(1, 2, 0).contiguous().cpu().numpy()

# ------------------------------------------------------------------ 编码
def pick_encoder(threads=None):
    try: enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    except FileNotFoundError: raise SystemExit("找不到 ffmpeg，请先安装并加入 PATH")
    if DEV is not None and DEV.type == "cuda" and "h264_nvenc" in enc:
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "18", "-b:v", "0", "-profile:v", "high", "-g", "60"]
    if DEV is not None and DEV.type == "mps" and "h264_videotoolbox" in enc:
        return ["-c:v", "h264_videotoolbox", "-b:v", "24M"]
    return ["-c:v", "libx264", "-preset", "fast", "-crf", "17", "-g", "60"] + (["-threads", str(threads)] if threads else [])

_enc_ok = None
def open_writer(path, enc=None, threads=None):
    """NVENC 可用就用硬件编码（不占 CPU）；失败自动退回 libx264"""
    global _enc_ok
    if enc is None:
        enc = pick_encoder(threads)
        if enc[1] == "h264_nvenc" and _enc_ok is None:
            t = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"color=s={W}x{H}:d=0.1", *enc, "-f", "null", "-"], capture_output=True)
            _enc_ok = t.returncode == 0
        if enc[1] == "h264_nvenc" and not _enc_ok:
            enc = ["-c:v", "libx264", "-preset", "fast", "-crf", "17", "-g", "60"] + (["-threads", str(threads)] if threads else [])
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-", *enc, "-pix_fmt", "yuv420p", path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)

# ------------------------------------------------------------------ 音频工具
SR = 48000
def load_audio(path, sr=SR):
    p = subprocess.run(["ffmpeg", "-v", "quiet", "-i", path, "-f", "f32le", "-ac", "2", "-ar", str(sr), "-"], capture_output=True)
    a = np.frombuffer(p.stdout, np.float32)
    return a.reshape(-1, 2).copy() if len(a) else np.zeros((0, 2), np.float32)

def save_wav(path, a, sr=SR):
    a = np.clip(a, -1, 1)
    p = subprocess.Popen(["ffmpeg", "-y", "-v", "quiet", "-f", "f32le", "-ac", "2", "-ar", str(sr), "-i", "-", path], stdin=subprocess.PIPE)
    p.communicate(a.astype(np.float32).tobytes())

def portal(src, x0, y0, x1, y1):
    """连续镜头转场用的“窗口”：屏幕矩形里显示 src。矩形小时显示整幅（等比裁切，不变形），
    矩形放大到盖满屏幕时与 src 像素一一对齐，可以无缝接管。返回整帧图（只在矩形内有意义）"""
    rw, rh = max(x1 - x0, 1e-3), max(y1 - y0, 1e-3)
    s = min(1.0, max(rw / DW, rh / DH))
    sw, sh = rw / s, rh / s
    cx = lerp(DW / 2, (x0 + x1) / 2, s); cy = lerp(DH / 2, (y0 + y1) / 2, s)
    sx0, sy0 = cx - sw / 2, cy - sh / 2
    u = (sx0 + (X - x0) / rw * sw) / DW * 2 - 1
    v = (sy0 + (Y - y0) / rh * sh) / DH * 2 - 1
    g = torch.stack([u, v], -1)[None]
    return F.grid_sample(src[None], g, mode="bilinear", padding_mode="border", align_corners=False)[0]
