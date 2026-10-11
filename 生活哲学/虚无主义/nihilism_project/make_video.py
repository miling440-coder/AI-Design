import subprocess, math, wave, random, sys, json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

random.seed(11); np.random.seed(11)
W, H, FPS = 1080, 1920, 30
DUR = 160.0
BG = np.array([8, 8, 10]); FG = (236, 232, 224); DIM = (132, 130, 126); WARM = (232, 192, 140)
FONT = "font.otf"
_fc = {}
def font(sz):
    if sz not in _fc: _fc[sz] = ImageFont.truetype(FONT, sz)
    return _fc[sz]
CX = W / 2; TY = 880          # text centre (above Douyin bottom UI)
BEAT = 0.5315
def snap(t): return round(t / BEAT) * BEAT

def cl(x): return max(0.0, min(1.0, x))
def ease(x): x = cl(x); return x * x * (3 - 2 * x)
def eout(x): x = cl(x); return 1 - (1 - x) ** 3
def ein(x): x = cl(x); return x ** 3
def col(c, a): a = cl(a); return tuple(int(BG[i] + (c[i] - BG[i]) * a) for i in range(3))

STY = {"main": (84, FG), "big": (136, FG), "sub": (44, DIM), "q": (70, FG), "acc": (62, WARM), "small": (38, DIM)}

# ---------------------------------------------------------------- script
# each block: (start, end, lines, opts)   lines: list of (text, style)
E = []
def B(t0, t1, lines, **o):
    E.append(dict(kind="block", t0=snap(t0), t1=t1, lines=lines, **o))
def K(kind, t0, t1, **o):
    E.append(dict(kind=kind, t0=t0, t1=t1, **o))

ACTS = [(0, 49.6, "第一幕 · 钩子与共鸣"), (49.6, 93.4, "第二幕 · 问题深化"), (93.4, 160, "第三幕 · 启发与落点")]

# ---- ACT 1 : hook & resonance (0 - 49.6)
K("dot", 0, 3.2, mode="breathe", y=TY - 230)
B(0.2, 3.7, [("你有多久，", "main"), ("没有真正期待过明天了？", "main")], per=0.06, line_gap=0.55, uline=(1, 1.9))
B(3.9, 7.7, [("明明每天都很忙，", "main"), ("却越来越不知道", "main"), ("自己在忙什么。", "main")], per=0.05, line_gap=0.45)
B(7.9, 11.7, [("你不是想不开。", "main"), ("你只是突然发现——", "sub")], per=0.07, line_gap=0.9)
B(11.9, 16.3, [("这一切，", "main"), ("好像都没有意义。", "main")], per=0.09, line_gap=0.7, fall_at=15.2)
K("title", 16.5, 20.6)                                       # music lifts here
K("words", 20.7, 27.1, words=["07:30  闹钟", "地铁", "工位", "绩效", "外卖", "加班", "刷手机", "凌晨两点"], per=0.8)
B(27.1, 30.2, [("日 复 一 日。", "big")], per=0.2, pop=True)
K("press", 30.2, 45.0)                                       # frame closes in: pressure
B(30.3, 33.6, [("想躺平，", "main"), ("又不敢。", "main")], per=0.07, line_gap=0.8)
B(33.7, 37.6, [("想奋斗，", "main"), ("又觉得没什么意义。", "main")], per=0.07, line_gap=0.8)
B(37.7, 41.2, [("你不是没有欲望。", "main")], per=0.08)
B(41.3, 45.2, [("你只是怕，", "main"), ("所有努力都没有回报。", "main")], per=0.07, line_gap=0.6, uline=(1, 43.2))
B(45.3, 49.6, [("你还活着，", "main"), ("却像只是——没停下。", "main")], per=0.08, line_gap=0.9, type_=True)

# ---- ACT 2 : deepening (49.6 - 93.4)
B(49.8, 53.7, [("选择越来越多，", "main"), ("确定越来越少。", "main")], per=0.06, line_gap=0.8, drift=1)
B(53.8, 57.6, [("这种空，有个名字——", "sub"), ("虚无主义。", "big")], per=0.12, line_gap=0.9, pop=True)
B(57.8, 62.0, [("它不是矫情。", "main"), ("是意义感，崩塌了。", "main")], per=0.07, line_gap=0.8, fall_at=60.6)
B(62.1, 66.4, [("1882 年，尼采写下：", "sub"), ("上帝已死。", "big")], per=0.12, line_gap=0.7, type_=True, glitch_at=65.0)
B(66.6, 70.6, [("他说的不是神，", "main"), ("是那个替你回答", "main"), ("「为什么活」的东西。", "main")], per=0.05, line_gap=0.5, uline=(2, 68.6))
B(70.8, 74.6, [("得不到，痛苦。", "main"), ("得到了，无聊。", "main"), ("—— 叔本华", "sub")], per=0.07, line_gap=0.8)
B(74.8, 77.0, [("从前，人生有标准答案。", "main")], per=0.06)
K("list", snap(77.1), 82.5, items=["好好读书", "找份稳定工作", "买房", "结婚生子"], strikes=[snap(78.7), snap(79.8), snap(80.8), snap(81.3)], fall_at=81.7)
K("stars", 82.4, 93.6, warp_at=91.4)
B(82.8, 86.6, [("宇宙不回答任何人。", "main"), ("—— 加缪：世界，是沉默的", "sub")], per=0.1, line_gap=1.0, track=8)
B(86.8, 90.2, [("你感到空，", "main"), ("不是因为你失败了。", "main")], per=0.08, line_gap=0.8, track=6)
B(90.3, 93.2, [("而是你开始，", "main"), ("看见真实。", "main")], per=0.1, line_gap=0.6, track=6)
K("collapse", 92.4, 93.5)

# ---- ACT 3 : turn & landing (93.4 - 160)
K("ring", 93.5, 98.0)
B(93.5, 97.6, [("不。", "big"), ("虚无不是终点。", "sub")], per=0.1, line_gap=1.2, pop=True)
B(97.8, 101.9, [("它只是把旧答案，", "main"), ("全部擦干净了。", "main")], per=0.07, line_gap=0.8, wipe_at=100.4)
B(102.0, 106.5, [("西西弗推石头上山，", "main"), ("石头，总会滚下来。", "main")], per=0.06, line_gap=1.0, drift=-1)
B(106.6, 111.0, [("加缪却说：", "sub"), ("「我们必须想象", "q"), ("西西弗是幸福的。」", "q")], per=0.08, line_gap=0.7, uline=(2, 109.2))
B(111.1, 115.5, [("不是因为石头有意义，", "main")], per=0.07)
B(115.6, 119.7, [("而是推的人，", "main"), ("决定了怎么推。", "main")], per=0.07, line_gap=0.8)
B(119.8, 124.0, [("意义，不是被找到的。", "main")], per=0.08, strike=(0, 122.3))
K("particles", snap(124.1), 130.0, text="是被创造的。", size=96)
B(130.0, 134.1, [("它不必宏大。", "main"), ("可以很小，很具体。", "sub")], per=0.08, line_gap=0.9)
K("acts", snap(134.2), 144.8, items=["一顿认真吃完的饭", "一次下班后的散步", "一句说出口的「不」", "一件只为自己做的小事"], gap=2.13)
B(144.9, 147.6, [("世界没有标准答案。", "main")], per=0.08)
K("particles", snap(147.7), 153.4, text="那就活出你的回答。", size=80)
B(153.5, 157.6, [("今天，", "main"), ("你想为自己做哪件小事？", "main"), ("评论区，留下你的回答", "small")], per=0.07, line_gap=0.7)
K("dot", 157.4, 160.0, mode="end", y=TY)

# -------------------------------------------------------------- precompute
NST = 650
stars = np.stack([np.random.uniform(-1, 1, NST), np.random.uniform(-1, 1, NST) * 1.8, np.random.uniform(0.05, 1, NST)], 1)

def text_points(text, sz, n):
    im = Image.new("L", (W, H), 0); dd = ImageDraw.Draw(im)
    f = font(sz)
    while dd.textlength(text, font=f) > W * 0.88: sz -= 4; f = font(sz)
    dd.text((CX, TY), text, font=f, fill=255, anchor="mm")
    ys, xs = np.nonzero(np.array(im) > 128)
    idx = np.random.choice(len(xs), n, replace=len(xs) < n)
    return xs[idx].astype(float), ys[idx].astype(float), f
for e in E:
    if e["kind"] == "particles":
        n = 4500
        e["PX"], e["PY"], e["font"] = text_points(e["text"], e["size"], n)
        a = np.random.uniform(0, 2 * np.pi, n); r = np.random.uniform(500, 1500, n)
        e["SX"], e["SY"] = CX + np.cos(a) * r, TY + np.sin(a) * r
        e["PD"] = np.random.uniform(0, 1.2, n); e["UV"] = np.random.uniform(40, 160, n); e["UX"] = np.random.normal(0, 30, n)
    if e["kind"] == "block":
        e["rot"] = [random.uniform(-80, 80) for _ in range(60)]
        e["fd"] = [random.uniform(0, 0.7) for _ in range(60)]

grain = [np.random.normal(0, 3.2, (H // 3, W // 3, 1)).repeat(3, 0).repeat(3, 1) for _ in range(6)]
yy, xx = np.mgrid[0:H, 0:W]
VIG = (1 - 0.32 * (((xx - CX) / CX) ** 2 + ((yy - H / 2) / (H / 2)) ** 2) / 2)[..., None].astype(np.float32)

def sprite(c, f, rgb, a):
    s = int(f.size * 1.6); im = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((s / 2, s / 2), c, font=f, fill=rgb + (int(255 * cl(a)),), anchor="mm"); return im

# -------------------------------------------------------------- renderers
def layout_block(d, e):
    rows = []; y_total = 0
    for txt, st in e["lines"]:
        sz, rgb = STY[st]; f = font(sz); tr = e.get("track", 0)
        while d.textlength(txt, font=f) + tr * len(txt) > W * 0.86: sz -= 3; f = font(sz)
        h = sz * 1.55; rows.append([txt, f, rgb, h]); y_total += h
    y = e.get("y", TY) - y_total / 2; out = []
    for txt, f, rgb, h in rows:
        ws = [d.textlength(c, font=f) for c in txt]; tr = e.get("track", 0)
        tw = sum(ws) + tr * (len(txt) - 1); x = CX - tw / 2; xs = []
        for w in ws: xs.append(x + w / 2); x += w + tr
        out.append((txt, f, rgb, y + h / 2, xs, CX - tw / 2, CX + tw / 2)); y += h
    return out

def draw_block(im, d, t, e):
    t0, t1 = e["t0"], e["t1"]; per = e.get("per", 0.07); lg = e.get("line_gap", 0.6)
    L = layout_block(d, e); k = 0
    out_a = 1 - ein((t - (t1 - 0.55)) / 0.55)
    gz = 0
    for li, (txt, f, rgb, y, xs, xl, xr) in enumerate(L):
        ls = t0 + li * lg + (0.05 if li else 0)
        if e.get("type_"):
            n = int(cl((t - ls) / max(0.3, len(txt) * per)) * len(txt) + 1e-6)
            for i in range(n):
                d.text((xs[i], y), txt[i], font=f, fill=col(rgb, out_a), anchor="mm")
            if 0 < t - ls and n < len(txt) and int(t * 4) % 2 == 0:
                d.line([xs[min(n, len(xs) - 1)] - f.size * 0.5, y - f.size * 0.5, xs[min(n, len(xs) - 1)] - f.size * 0.5, y + f.size * 0.5], fill=col(FG, 0.7 * out_a), width=2)
            k += len(txt); continue
        for i, c in enumerate(txt):
            cs = ls + i * per; p = eout((t - cs) / 0.85)
            if p <= 0.003: k += 1; continue
            yo = (1 - p) * 26 - (1 - out_a) * 22; a = p * out_a
            if e.get("pop") and STY_of(e, li) == "big":
                sc = 1 + 0.35 * (1 - eout((t - cs) / 0.5))
                f2 = font(int(f.size * sc)); fx = f2
            else: fx = f
            if e.get("wipe_at") and t > e["wipe_at"]:
                sweep = xl - 60 + (xr - xl + 120) * eout((t - e["wipe_at"]) / 1.1)
                if xs[i] < sweep: a *= cl(1 - (sweep - xs[i]) / 120)
            fa = e.get("fall_at")
            if fa and t > fa + e["fd"][k % 60]:
                ft = t - fa - e["fd"][k % 60]
                sp = sprite(c, f, rgb, a * (1 - cl(ft / 1.5))).rotate(e["rot"][k % 60] * ft * 1.4, resample=Image.BICUBIC)
                yy_ = y + 0.5 * 2600 * ft * ft
                if yy_ < H + 100: im.paste(sp, (int(xs[i] - sp.width / 2), int(yy_ - sp.height / 2)), sp)
            else:
                d.text((xs[i], y + yo), c, font=fx, fill=col(rgb, a), anchor="mm")
            k += 1
        if e.get("uline") and e["uline"][0] == li and t > e["uline"][1]:
            p = eout((t - e["uline"][1]) / 0.7)
            d.line([xl, y + f.size * 0.72, xl + (xr - xl) * p, y + f.size * 0.72], fill=col(WARM, 0.9 * out_a), width=3)
        if e.get("strike") and e["strike"][0] == li and t > e["strike"][1]:
            p = eout((t - e["strike"][1]) / 0.45)
            d.line([xl - 10, y + 4, xl - 10 + (xr - xl + 20) * p, y + 4], fill=col(FG, out_a), width=4)
        if e.get("wipe_at") and e["wipe_at"] < t < e["wipe_at"] + 1.2:
            sweep = xl - 60 + (xr - xl + 120) * eout((t - e["wipe_at"]) / 1.1)
            d.line([sweep, y - f.size, sweep, y + f.size], fill=col(FG, 0.9), width=2)
    return L

def STY_of(e, li): return e["lines"][li][1]

def draw(t):
    im = Image.new("RGB", (W, H), tuple(BG)); d = ImageDraw.Draw(im)
    zoom = 1.0 + 0.012 * math.sin(t * 0.21) + 0.012; dx = 8 * math.sin(t * 0.13); post = []
    for e in E:
        if not (e["t0"] - 0.01 <= t < e["t1"] + 0.01): continue
        k = e["kind"]; lt = t - e["t0"]; span = e["t1"] - e["t0"]
        if k == "block":
            draw_block(im, d, t, e)
            if e.get("glitch_at") and t > e["glitch_at"]: post.append(("glitch", ein((t - e["glitch_at"]) / (e["t1"] - e["glitch_at"]))))
            if e.get("drift"): dx += e["drift"] * 30 * lt / span
        elif k == "dot":
            y = e["y"]
            if e["mode"] == "breathe":
                a = ease(lt / 0.4) * (1 - ease((lt - 2.6) / 0.6)); r = 6 + 2 * math.sin(t * 4)
                d.ellipse([CX - r, y - r, CX + r, y + r], fill=col(FG, a))
            else:
                p = lt / span; a = ease(p / 0.2) * (1 - ease((p - 0.8) / 0.2)); r = 7
                d.ellipse([CX - r, y - r, CX + r, y + r], fill=col(FG, a))
                rp = eout((p - 0.1) / 0.7); rr = 10 + rp * 160
                if p > 0.1: d.ellipse([CX - rr, y - rr, CX + rr, y + rr], outline=col(FG, (1 - rp) * 0.6 * a), width=2)
        elif k == "title":
            # line sweeps open then title
            p = eout(lt / 0.9); a_out = 1 - ein((t - (e["t1"] - 0.6)) / 0.6)
            for yy_ in (TY - 150, TY + 150):
                lw = 380 * p; d.line([CX - lw, yy_, CX + lw, yy_], fill=col(DIM, 0.8 * a_out), width=1)
            sub = dict(t0=e["t0"] + 0.15, t1=e["t1"], lines=[("虚无主义", "big"), ("N I H I L I S M", "small")], per=0.16, line_gap=0.9, pop=True, track=30 - 20 * eout(lt / 3))
            draw_block(im, d, t, sub); zoom += 0.05 * (1 - eout(lt / 1.2))
        elif k == "words":
            ws = e["words"]; per = e["per"]; i = int(lt / per)
            if i < len(ws):
                q = (lt - i * per) / per; a = ease(q / 0.12) * (1 - ease((q - 0.8) / 0.2))
                sc = 1.08 - 0.08 * eout(q)
                d.text((CX, TY), ws[i], font=font(int(84 * sc)), fill=col(FG, a), anchor="mm")
                # progress ticks
                for j in range(len(ws)):
                    xj = CX + (j - (len(ws) - 1) / 2) * 30
                    d.ellipse([xj - 3, TY + 160 - 3, xj + 3, TY + 160 + 3], fill=col(FG if j <= i else DIM, 0.8 if j <= i else 0.3))
        elif k == "press":
            p = ease(lt / (span * 0.85)); a = 1 - ease((t - (e["t1"] - 0.8)) / 0.8)
            m = 30 + p * 230; my = 60 + p * 520
            d.rectangle([m, my, W - m, H - my], outline=col(DIM, 0.5 * a * ease(lt / 1.2)), width=2)
        elif k == "list":
            items = e["items"]; n = len(items); fa = e["fall_at"]
            for i, it in enumerate(items):
                y = TY - (n - 1) * 70 + i * 140; s0 = e["t0"] + i * 0.35
                p = eout((t - s0) / 0.7); f = font(66)
                if t < fa + i * 0.12:
                    d.text((CX, y + (1 - p) * 20), it, font=f, fill=col(FG, p), anchor="mm")
                    st = e["strikes"][i]
                    if t > st:
                        q = eout((t - st) / 0.35); tw = d.textlength(it, font=f)
                        d.line([CX - tw / 2 - 14, y + 4, CX - tw / 2 - 14 + (tw + 28) * q, y + 4], fill=col(WARM, 1), width=4)
                else:
                    ft = t - fa - i * 0.12
                    sp = Image.new("RGBA", (900, 140), (0, 0, 0, 0)); sd = ImageDraw.Draw(sp)
                    sd.text((450, 70), it, font=f, fill=FG + (int(255 * cl(1 - ft / 1.3)),), anchor="mm")
                    tw = sd.textlength(it, font=f); sd.line([450 - tw / 2 - 14, 74, 450 + tw / 2 + 14, 74], fill=WARM + (int(255 * cl(1 - ft / 1.3)),), width=4)
                    sp = sp.rotate((i % 2 * 2 - 1) * 25 * ft, resample=Image.BICUBIC)
                    im.paste(sp, (int(CX - 450), int(y - 70 + 0.5 * 2400 * ft * ft)), sp)
        elif k == "stars":
            warp = ein((t - e["warp_at"]) / (e["t1"] - e["warp_at"])) if t > e["warp_at"] else 0
            z = (stars[:, 2] - (lt * 0.045 + warp ** 2 * 1.3)) % 1.0 + 0.02
            sx = CX + stars[:, 0] / z * 200; sy = TY + stars[:, 1] / z * 200
            vis = ease(lt / 1.5) * (1 - ease((t - (e["t1"] - 0.6)) / 0.6))
            for i in range(NST):
                b = cl((1 - z[i]) * 1.2) * vis
                if b < 0.03 or not (-50 < sx[i] < W + 50 and -50 < sy[i] < H + 50): continue
                if warp > 0.02:
                    z2 = z[i] + 0.12 * warp
                    d.line([sx[i], sy[i], CX + stars[i, 0] / z2 * 200, TY + stars[i, 1] / z2 * 200], fill=col(FG, b), width=1)
                else:
                    s = 1 + (1 - z[i]) * 2.2; d.ellipse([sx[i] - s / 2, sy[i] - s / 2, sx[i] + s / 2, sy[i] + s / 2], fill=col(FG, b))
        elif k == "collapse":
            post.append(("collapse", ein(lt / span)))
        elif k == "ring":
            for off in (0, 0.35, 0.8):
                p = eout((lt - off) / 2.6); r = 20 + p * 1000
                if lt > off: d.ellipse([CX - r, TY - r, CX + r, TY + r], outline=col(FG, (1 - p) * 0.6), width=2)
        elif k == "particles":
            n = len(e["PX"]); p = 1 - (1 - np.clip((t - e["t0"] - e["PD"]) / 2.4, 0, 1)) ** 3
            x = e["SX"] + (e["PX"] - e["SX"]) * p; y = e["SY"] + (e["PY"] - e["SY"]) * p
            rel = e["t1"] - 1.6; up = np.clip(t - rel - e["PD"] * 0.4, 0, None)
            x = x + e["UX"] * up; y = y - e["UV"] * up * up
            a = np.clip(lt / 0.8, 0, 1) * np.clip(1 - up / 1.0, 0, 1)
            arr = np.array(im); xi = x.astype(int); yi = y.astype(int)
            ok = (xi >= 0) & (xi < W - 1) & (yi >= 0) & (yi < H - 1) & (a > 0.01)
            cc = (BG + (np.array(FG) - BG) * a[ok, None]).astype(np.uint8)
            for ddx in (0, 1):
                for ddy in (0, 1): arr[yi[ok] + ddy, xi[ok] + ddx] = cc
            im = Image.fromarray(arr); d = ImageDraw.Draw(im)
            ta = ease((lt - 2.8) / 0.8) * (1 - ease((t - (rel - 0.4)) / 0.4))
            if ta > 0.01: d.text((CX, TY), e["text"], font=e["font"], fill=col(FG, ta), anchor="mm")
        elif k == "acts":
            items = e["items"]; g = e["gap"]
            a_out = 1 - ein((t - (e["t1"] - 0.6)) / 0.6)
            for i, it in enumerate(items):
                s0 = e["t0"] + i * g; y = TY - 1.5 * 130 + i * 130
                if t < s0: continue
                p = eout((t - s0) / 0.9)
                # small warm dot leads the line
                d.ellipse([CX - 300 - 5, y - 5, CX - 300 + 5, y + 5], fill=col(WARM, p * a_out))
                f = font(60); tw = d.textlength(it, font=f)
                d.text((CX - 270 + (1 - p) * 30, y), it, font=f, fill=col(FG, p * a_out), anchor="lm")
    # ---- post
    if zoom != 1.0 or dx:
        z = max(zoom, 1.0); w2, h2 = W / z, H / z; cx = min(max(CX + dx, w2 / 2), W - w2 / 2)
        im = im.resize((W, H), Image.BILINEAR, box=(cx - w2 / 2, H / 2 - h2 / 2, cx + w2 / 2, H / 2 + h2 / 2))
    arr = np.array(im).astype(np.float32)
    for kind, g in post:
        if kind == "glitch":
            rng = np.random.default_rng(int(t * 30))
            for _ in range(int(4 + 22 * g)):
                y0 = int(rng.integers(TY - 220, TY + 220)); hh = int(rng.integers(3, 20))
                arr[y0:y0 + hh] = np.roll(arr[y0:y0 + hh], int(rng.normal(0, 180 * g)), axis=1)
            arr = BG + (arr - BG) * (1 - g)
        if kind == "collapse":
            s = max(0.004, 1 - g)
            sm = Image.fromarray(arr.astype(np.uint8)).resize((max(2, int(W * s)), max(2, int(H * s))), Image.BILINEAR)
            cv = Image.new("RGB", (W, H), tuple(BG)); cv.paste(sm, (int(CX - sm.width / 2), int(TY - sm.height * TY / H)))
            if g > 0.55:
                r = 7; ImageDraw.Draw(cv).ellipse([CX - r, TY - r, CX + r, TY + r], fill=col(FG, ease((g - 0.55) / 0.3)))
            arr = np.array(cv).astype(np.float32)
    arr = arr * VIG + grain[int(t * 12) % 6]
    return np.clip(arr, 0, 255).astype(np.uint8)

# -------------------------------------------------------------- SFX + mix
def build_audio():
    SR = 44100; n = int(SR * DUR); tt = np.arange(n) / SR; fx = np.zeros(n)
    def tone(s, f, amp, dec):
        m = tt >= s; x = tt[m] - s
        fx[m] += amp * np.sin(2 * np.pi * f * x) * np.exp(-x * dec) * np.clip(x / 0.008, 0, 1)
    def whoosh(s, dur, amp, rev=False, fc=900):
        i0, i1 = int(s * SR), min(n, int((s + dur) * SR)); L = i1 - i0
        z = np.random.normal(0, 1, L); sp = np.fft.rfft(z); fr = np.fft.rfftfreq(L, 1 / SR)
        sp *= np.exp(-((fr - fc) / fc) ** 2); z = np.fft.irfft(sp, L); z /= np.abs(z).max()
        x = np.linspace(0, 1, L); en = (x ** 2) * np.clip((1 - x) / 0.15, 0, 1) if not rev else ((1 - x) ** 2) * np.clip(x / 0.08, 0, 1)
        fx[i0:i1] += amp * z * en
    def tick(s, amp=0.05):
        i0 = int(s * SR); L = int(0.03 * SR); x = np.arange(L) / SR
        fx[i0:i0 + L] += amp * np.sin(2 * np.pi * 2600 * x) * np.exp(-x * 240)
    for e in E:
        k = e["kind"]; t0 = e["t0"]
        if k == "block":
            tone(t0, 880, 0.035, 3)
            if e.get("type_"):
                for li, (txt, st) in enumerate(e["lines"]):
                    for i in range(len(txt)): tick(t0 + li * e.get("line_gap", .6) + i * max(0.3, len(txt) * e.get("per", .07)) / len(txt))
            if e.get("fall_at"): whoosh(e["fall_at"] - 0.1, 1.0, 0.18, rev=True, fc=500); tone(e["fall_at"] + 0.6, 55, 0.35, 3)
            if e.get("glitch_at"): whoosh(e["glitch_at"], 1.0, 0.18, fc=3000)
            if e.get("uline"): tone(e["uline"][1], 1318, 0.04, 4)
            if e.get("strike"): whoosh(e["strike"][1] - 0.05, 0.5, 0.12, rev=True, fc=2500)
            if e.get("wipe_at"): whoosh(e["wipe_at"], 1.2, 0.14, fc=1500)
        elif k == "title": tone(t0, 55, 0.4, 1.4); whoosh(t0 - 1.0, 1.0, 0.2)
        elif k == "words":
            for i in range(len(e["words"])): tick(t0 + i * e["per"], 0.06)
        elif k == "list":
            for s in e["strikes"]: whoosh(s - 0.05, 0.4, 0.12, rev=True, fc=2500)
            whoosh(e["fall_at"], 1.2, 0.18, rev=True, fc=500); tone(e["fall_at"] + 0.8, 50, 0.3, 3)
        elif k == "stars": whoosh(e["warp_at"], e["t1"] - e["warp_at"], 0.3)
        elif k == "ring": tone(t0, 45, 0.5, 1.2); tone(t0, 440, 0.05, 1.0)
        elif k == "particles": whoosh(t0, 2.4, 0.16, fc=4000); tone(t0 + 2.6, 1760, 0.03, 2); whoosh(e["t1"] - 1.6, 1.4, 0.1, rev=True, fc=4000)
        elif k == "acts":
            for i in range(len(e["items"])): tone(t0 + i * e["gap"], 1046, 0.03, 3)
        elif k == "collapse": whoosh(t0 - 0.4, 1.4, 0.22)
    fx = fx / max(1e-6, np.abs(fx).max()) * 0.5
    st = np.stack([fx, np.roll(fx, 220)], 1)
    with wave.open("sfx3.wav", "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((st * 32767).astype(np.int16).tobytes())

def storyboard():
    rows = []
    for e in sorted(E, key=lambda e: e["t0"]):
        if e["kind"] == "block": txt = " / ".join(l[0] for l in e["lines"])
        elif e["kind"] in ("words", "list", "acts"): txt = " / ".join(e.get("words") or e.get("items"))
        elif e["kind"] == "particles": txt = e["text"]
        elif e["kind"] == "title": txt = "虚无主义 NIHILISM"
        else: txt = ""
        fxs = [k for k in ("type_", "fall_at", "glitch_at", "uline", "strike", "wipe_at", "pop") if e.get(k)]
        rows.append(dict(t0=round(e["t0"], 1), t1=round(e["t1"], 1), kind=e["kind"], text=txt, fx=fxs))
    json.dump(rows, open("storyboard3.json", "w"), ensure_ascii=False, indent=0)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "pv":
        for s in sys.argv[2:]:
            Image.fromarray(draw(float(s))).resize((360, 640)).save(f"p3_{s}.png")
        sys.exit()
    storyboard(); build_audio()
    p = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                          "-i", "bgm.mp3", "-i", "sfx3.wav",
                          "-filter_complex", "[1:a]volume=1.0[m];[2:a]volume=0.45[s];[m][s]amix=inputs=2:normalize=0,afade=t=out:st=157.5:d=2.5,loudnorm=I=-14:TP=-1:LRA=11[a]",
                          "-map", "0:v", "-map", "[a]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", "-tune", "grain",
                          "-maxrate", "4M", "-bufsize", "8M", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-t", str(DUR),
                          "-movflags", "+faststart", "nihilism_douyin.mp4"], stdin=subprocess.PIPE)
    for i in range(int(FPS * DUR)):
        p.stdin.write(draw(i / FPS).tobytes())
    p.stdin.close(); p.wait(); print("done", p.returncode)
