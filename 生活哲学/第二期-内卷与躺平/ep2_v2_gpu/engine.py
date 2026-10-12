# 点阵线描 + 粒子变形引擎（《给空心的人》系列 v2 风格）
import math, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1440, 1080, 30
GY = 720
FG = np.array([236, 232, 224], np.float32); WARM = np.array([244, 196, 118], np.float32)
FONTS = {"L": "font.otf", "R": "font.otf", "B": "font.otf"}
_fc = {}
def font(w, sz):
    k = (w, sz)
    if k not in _fc: _fc[k] = ImageFont.truetype(FONTS[w], sz)
    return _fc[k]

def cl(x): return np.clip(x, 0.0, 1.0)
def ease(x): x = cl(x); return x * x * (3 - 2 * x)
def eio(x):
    x = cl(x); return np.where(x < .5, 4 * x ** 3, 1 - (-2 * x + 2) ** 3 / 2)
def eout(x): x = cl(x); return 1 - (1 - x) ** 3

# ------------------------------------------------------------ sampling
def sample_polys(polys, n, spacing=6.0, jitter=0.2, rng=None):
    polys = [np.asarray(p, np.float32) for p in polys if len(p) >= 2]
    segs = []; L = 0
    for p in polys:
        d = np.hypot(*np.diff(p, axis=0).T); segs.append((p, d)); L += d.sum()
    if L <= 0: return np.zeros((n, 3), np.float32)
    m = int(max(1, min(n, L / spacing)))
    ds = (np.arange(m) + 0.5) * L / m
    pts = np.zeros((m, 2), np.float32); acc = 0; i = 0
    for p, d in segs:
        cum = np.concatenate([[0], np.cumsum(d)]); tot = cum[-1]
        sel = (ds >= acc) & (ds < acc + tot + 1e-6)
        if sel.any():
            loc = ds[sel] - acc
            pts[sel, 0] = np.interp(loc, cum, p[:, 0]); pts[sel, 1] = np.interp(loc, cum, p[:, 1])
        acc += tot
    return spread(pts, n, jitter)

def spread(pts, n, jitter=0.2):
    m = len(pts)
    if m == 0: return np.zeros((n, 3), np.float32)
    if m > n:
        # 输入按轮廓顺序排列；均匀缩减避免随机丢点在笔画上留下缺口。
        idx = np.floor((np.arange(n) + 0.5) * m / n).astype(np.int64); pts = pts[idx]; m = n
    out = np.zeros((n, 3), np.float32)
    j = np.arange(n) % m
    out[:, :2] = pts[j]
    if n > m:
        r = np.random.default_rng(n + m).normal(0, jitter, (n, 2)); r[:m] = 0; out[:, :2] += r
    out[:, 2] = m / n
    return out

def circle(cx, cy, r, a0=0, a1=2 * math.pi, k=None):
    k = k or max(24, int(abs(a1 - a0) * r / 3))
    a = np.linspace(a0, a1, k); return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1)

_tc = {}
def text_pts_local(text, w, sz, spacing):
    k = (text, w, sz, spacing)
    if k in _tc: return _tc[k]
    f = font(w, sz); d0 = ImageDraw.Draw(Image.new("L", (8, 8)))
    bb = d0.textbbox((0, 0), text, font=f); tw, th = bb[2] - bb[0] + 20, bb[3] - bb[1] + 20
    im = Image.new("L", (tw, th), 0); ImageDraw.Draw(im).text((10 - bb[0], 10 - bb[1]), text, font=f, fill=255)
    a = np.array(im)
    # 外边缘和字内孔洞分别沿弧长采样，细笔画两侧不再被同一网格合并。
    contours, _ = cv2.findContours((a > 128).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=lambda c: (int(c[:, 0, 1].min()), int(c[:, 0, 0].min()), len(c)))
    s = max(1.5, min(float(spacing), 2.5, sz / 40.0))
    parts = []
    for contour in contours:
        p = contour[:, 0, :].astype(np.float32)
        if len(p) == 1:
            parts.append(p); continue
        # 固定起点使缓存和粒子匹配顺序稳定。
        top = np.lexsort((p[:, 0], p[:, 1]))[0]
        p = np.roll(p, -top, axis=0); p = np.vstack([p, p[:1]])
        d = np.hypot(*np.diff(p, axis=0).T)
        cum = np.concatenate([[0.0], np.cumsum(d)])
        total = cum[-1]
        if total <= 0:
            parts.append(p[:1]); continue
        count = max(3, int(math.ceil(total / s)))
        ds = np.arange(count) * total / count
        parts.append(np.stack([np.interp(ds, cum, p[:, 0]), np.interp(ds, cum, p[:, 1])], 1).astype(np.float32))
    pts = np.concatenate(parts) if parts else np.empty((0, 2), np.float32)
    pts -= np.array([tw / 2, th / 2], np.float32)
    _tc[k] = pts; return pts

def text_shape(text, cx, cy, sz, n, w="R", spacing=5, rot=0.0, pivot=None):
    p = text_pts_local(text, w, sz, spacing).copy()
    if rot:
        c, s = math.cos(rot), math.sin(rot); px, py = pivot if pivot else (0, 0)
        p = p - [px, py]; p = np.stack([p[:, 0] * c - p[:, 1] * s, p[:, 0] * s + p[:, 1] * c], 1) + [px, py]
    p += [cx, cy]
    return spread(p, n, 0.15)

# ------------------------------------------------------------ person (outlined figure)
POSES = {}
def P(lean=0, head=0, a1=(8, 6), a2=(-8, -6), l1=(4, 0), l2=(-4, 0)):
    return dict(lean=lean, head=head, a1=a1, a2=a2, l1=l1, l2=l2)
def pose(name, ph=0.0):
    s = math.sin(ph); c = math.cos(ph)
    if name == "stand": return P()
    if name == "run":
        return P(14, -8, (-50 * s, 85), (50 * s, 85), (42 * s, -25 - 45 * max(0, c)), (-42 * s, -25 - 45 * max(0, -c)))
    if name == "walk":
        return P(4, 0, (-22 * s, 12), (22 * s, 12), (24 * s, -8 - 22 * max(0, c)), (-24 * s, -8 - 22 * max(0, -c)))
    if name == "push":
        return P(40, -30, (98, -8), (88, -4), (28 + 14 * s, -38), (-34 - 14 * s, -4))
    if name == "tpose": return P(0, 0, (96, 0), (-96, 0), (18, 0), (-18, 0))
    if name == "lookup": return P(-4, -40, (12, 4), (-12, -4), (5, 0), (-5, 0))
    if name == "sit": return P(4, 6, (60, 30), (52, 30), (86, -84), (82, -80))
    if name == "sad": return P(16, 34, (16, 10), (8, 8), (4, 0), (-4, 0))
    if name == "reach": return P(14, -6, (82, 6), (74, 8), (18, -10), (-14, 0))
    return P()

def fk(p):
    rad = math.radians; lean = rad(p["lean"])
    hip = np.array([0.0, 0.0]); sh = hip + 0.30 * np.array([math.sin(lean), -math.cos(lean)])
    hd = rad(p["lean"] + p["head"]); head = sh + 0.13 * np.array([math.sin(hd), -math.cos(hd)])
    segs = [(hip, sh, 1.7)]
    feet = []
    for (a, b) in (p["l1"], p["l2"]):
        a, b = rad(a), rad(b)
        k = hip + 0.25 * np.array([math.sin(a), math.cos(a)]); f = k + 0.245 * np.array([math.sin(a + b), math.cos(a + b)])
        segs += [(hip, k, 1.15), (k, f, 1.0)]; feet.append(f)
    for (a, b) in (p["a1"], p["a2"]):
        a, b = rad(a), rad(b)
        e = sh + 0.15 * np.array([math.sin(a), math.cos(a)]); hnd = e + 0.14 * np.array([math.sin(a + b), math.cos(a + b)])
        segs += [(sh, e, 0.9), (e, hnd, 0.8)]
    return segs, head

_pc = {}
def person_contour(name, ph, h, rot, flip):
    q = round((ph % (2 * math.pi)) / (2 * math.pi) * 24) % 24
    k = (name, q, int(h), round(rot, 2), flip)
    if k in _pc: return _pc[k]
    segs, head = fk(pose(name, q / 24 * 2 * math.pi))
    S = int(h * 2.4); O = np.array([S / 2, S * 0.62]); wid = 0.095 * h
    im = Image.new("L", (S, S), 0); d = ImageDraw.Draw(im)
    for a, b, wf in segs:
        pa, pb = O + a * h, O + b * h; ww = wid * wf
        d.line([tuple(pa), tuple(pb)], fill=255, width=int(max(2, ww)))
        for pp in (pa, pb): d.ellipse([pp[0] - ww / 2, pp[1] - ww / 2, pp[0] + ww / 2, pp[1] + ww / 2], fill=255)
    hc = O + head * h; r = 0.095 * h
    d.ellipse([hc[0] - r, hc[1] - r, hc[0] + r, hc[1] + r], fill=255)
    if rot: im = im.rotate(math.degrees(rot), resample=Image.BICUBIC, center=tuple(O - [0, 0.45 * h]))
    a = np.array(im); a = cv2.GaussianBlur(a, (0, 0), 1.0)
    cs, _ = cv2.findContours((a > 110).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    ys, xs = np.nonzero(a > 110)
    bottom = ys.max() if len(ys) else O[1]; midx = (xs.min() + xs.max()) / 2 if len(xs) else O[0]
    polys = []
    for c in cs:
        c = c[:, 0, :].astype(np.float32)
        if len(c) < 6: continue
        top = np.argmin(c[:, 1]); c = np.roll(c, -top, 0); c = np.vstack([c, c[:1]])
        c[:, 0] -= midx if rot else O[0]; c[:, 1] -= bottom
        if flip: c[:, 0] *= -1
        polys.append(c)
    _pc[k] = polys; return polys

def person(name, x, y, h, n, ph=0.0, rot=0.0, flip=False, spacing=2.8):
    polys = [p + [x, y] for p in person_contour(name, ph, h, rot, flip)]
    return sample_polys(polys, n, spacing)

# ------------------------------------------------------------ hilbert matching
def hilbert(x, y, n=1024):
    x = x.copy(); y = y.copy(); d = np.zeros_like(x); s = n // 2
    while s > 0:
        rx = ((x & s) > 0).astype(np.int64); ry = ((y & s) > 0).astype(np.int64)
        d += s * s * ((3 * rx) ^ ry)
        m = ry == 0; f = m & (rx == 1)
        x[f] = n - 1 - x[f]; y[f] = n - 1 - y[f]
        xs = x[m].copy(); x[m] = y[m]; y[m] = xs
        s //= 2
    return d
def horder(p):
    mn = p.min(0); sz = max((p.max(0) - mn).max(), 1e-3)
    q = ((p - mn) / sz * 1023).astype(np.int64)
    return np.argsort(hilbert(q[:, 0], q[:, 1]), kind="stable")

# ------------------------------------------------------------ actors
class Shape:
    def __init__(self, fn, anim=False): self.fn, self.anim, self.cache = fn, anim, None
    def get(self, n, t, prev=None):
        if self.anim: return fix(self.fn(n, t), n)
        if self.cache is None: self.cache = fix(self.fn(n, t, prev) if self.fn.__code__.co_argcount == 3 else self.fn(n, t), n)
        return self.cache
def fix(a, n):
    if len(a) == n: return a
    w = a[:, 2].mean() * len(a) if a.shape[1] == 3 else len(a)
    k = max(1, min(len(a), int(round(w))))
    b = a[np.linspace(0, len(a) - 1, k).astype(int), :2]
    return spread(b, n, 0.2)
def S(fn): return Shape(fn)
def A(fn): return Shape(fn, True)

class Actor:
    def __init__(self, name, n, seed):
        self.name, self.n = name, n; r = np.random.default_rng(seed)
        self.delay = r.uniform(0, 1, n).astype(np.float32); self.sw = r.normal(0, 1, n).astype(np.float32)
        self.tw = r.uniform(0, 6.28, n).astype(np.float32); self.br = r.uniform(0.75, 1.15, n).astype(np.float32)
        dx = r.uniform(-300, W + 300, n); dy = r.uniform(-200, H + 200, n)
        self.dust = np.stack([dx, dy, np.full(n, 1.0)], 1).astype(np.float32)
        self.keys = []
    def key(self, t, shape, dur=1.0, a=1.0, col=FG, stag=0.45, swirl=60.0):
        self.keys.append(dict(t=t, shape=shape, dur=max(dur, 1e-3), a=a, col=np.asarray(col, np.float32), stag=stag, swirl=swirl))
    def prep(self):
        self.keys.sort(key=lambda k: k["t"])
        prevP = self.dust.copy(); prevA = 0.0; prevC = FG
        for i, k in enumerate(self.keys):
            if i > 0:
                st = self._eval(k["t"], i - 1)
                prevP, prevA, prevC = st
            k["prevP"], k["prevA"], k["prevC"] = prevP, prevA, prevC
            tgt = k["shape"].get(self.n, k["t"] + k["dur"], prevP)
            perm = np.empty(self.n, np.int64); perm[horder(prevP[:, :2])] = horder(tgt[:, :2])
            k["perm"] = perm
    def _eval(self, t, i):
        k = self.keys[i]
        cur = k["shape"].get(self.n, t, k["prevP"])[k["perm"]]
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
    def state(self, t):
        i = -1
        for j, k in enumerate(self.keys):
            if k["t"] <= t: i = j
        if i < 0: return None
        return self._eval(t, i)

# ------------------------------------------------------------ render
def splat(P, Wt, C):
    x, y = P[:, 0], P[:, 1]
    m = (x > 1) & (x < W - 2) & (y > 1) & (y < H - 2) & (Wt > 0.004)
    x, y, Wt, C = x[m], y[m], Wt[m], C[m]
    x0 = np.floor(x).astype(np.int64); y0 = np.floor(y).astype(np.int64); fx = x - x0; fy = y - y0
    idx = np.concatenate([y0 * W + x0, y0 * W + x0 + 1, (y0 + 1) * W + x0, (y0 + 1) * W + x0 + 1])
    w = np.concatenate([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy]) * np.tile(Wt, 4)
    out = np.empty((H, W, 3), np.float32)
    for c in range(3):
        out[..., c] = np.bincount(idx, weights=w * np.tile(C[:, c], 4), minlength=H * W).reshape(H, W)
    return out

# ============================================================ GPU 加速（PyTorch）
# 自动选择设备：NVIDIA 用 CUDA，Mac 用 MPS，都没有就退回 CPU。
# 可用环境变量 EP_DEVICE=cuda / mps / cpu 强制指定。
_TORCH = None; DEV = None
def gpu_init(prefer=None):
    global _TORCH, DEV
    import os, torch
    _TORCH = torch
    want = prefer or os.environ.get("EP_DEVICE")
    if want: DEV = torch.device(want)
    elif torch.cuda.is_available(): DEV = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available(): DEV = torch.device("mps")
    else: DEV = torch.device("cpu")
    torch.set_grad_enabled(False)
    return DEV

def T(a):
    """numpy -> 设备上的 float32 张量"""
    return _TORCH.as_tensor(np.ascontiguousarray(a), dtype=_TORCH.float32, device=DEV)

def splat_t(P, Wt, C):
    """GPU 版双线性点溅射，返回 (3,H,W) 张量。逻辑与 splat() 相同。"""
    torch = _TORCH
    x, y = P[:, 0], P[:, 1]
    m = (x > 1) & (x < W - 2) & (y > 1) & (y < H - 2) & (Wt > 0.004)
    out = torch.zeros(3, H * W, device=DEV)
    if not m.any(): return out.view(3, H, W)
    x = T(x[m]); y = T(y[m]); Wt = T(Wt[m]); C = T(C[m])
    x0 = torch.floor(x); y0 = torch.floor(y); fx = x - x0; fy = y - y0
    x0 = x0.long(); y0 = y0.long(); base = y0 * W + x0
    idx = torch.cat([base, base + 1, base + W, base + W + 1])
    w = torch.cat([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy]) * Wt.repeat(4)
    vals = w[None, :] * C.t().repeat(1, 4)
    out.index_add_(1, idx, vals)
    return out.view(3, H, W)

_gk = {}
def gblur_t(img, sigma):
    """可分离高斯模糊（等价 cv2.GaussianBlur(…, (0,0), sigma)），img 为 (C,H,W)。"""
    torch = _TORCH; F = torch.nn.functional
    if sigma <= 0: return img
    key = round(float(sigma), 4)
    if key not in _gk:
        r = max(1, int(math.ceil(sigma * 4)))
        xs = torch.arange(-r, r + 1, dtype=torch.float32, device=DEV)
        k = torch.exp(-xs ** 2 / (2 * sigma * sigma)); k = k / k.sum(); _gk[key] = (k, r)
    k, r = _gk[key]; c = img.shape[0]
    x = img[None]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)
    return x[0]

def pick_encoder():
    """按设备挑选 ffmpeg 硬件编码器，失败则用 libx264。"""
    import subprocess
    try: enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    except FileNotFoundError: raise SystemExit("找不到 ffmpeg，请先安装并加入 PATH")
    if DEV is not None and DEV.type == "cuda" and "h264_nvenc" in enc:
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "19", "-b:v", "0"]
    if DEV is not None and DEV.type == "mps" and "h264_videotoolbox" in enc:
        return ["-c:v", "h264_videotoolbox", "-b:v", "20M"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"]
