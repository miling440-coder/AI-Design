import subprocess, math, wave, random, sys, json
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops

random.seed(11); np.random.seed(11)
W, H, FPS = 1440, 1080, 30
DUR = 160.0
BG = np.array([8, 8, 10]); FG = (236, 232, 224); DIM = (132, 130, 126); WARM = (232, 192, 140)
FONT = "font.otf"
_fc = {}
def font(sz):
    if sz not in _fc: _fc[sz] = ImageFont.truetype(FONT, sz)
    return _fc[sz]
CX = W / 2; TY = 540          # text centre (above Douyin bottom UI)
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
B(0.2, 3.7, [("白天拼命卷，", "main"), ("晚上想躺平。", "main")], per=0.07, line_gap=0.8, uline=(1, 2.2))
B(3.9, 7.7, [("卷，怕自己白卷。", "main"), ("躺，怕自己被落下。", "main")], per=0.06, line_gap=0.8)
B(7.9, 11.7, [("你被夹在中间，", "main"), ("哪边都不敢松手。", "main")], per=0.07, line_gap=0.8, lie_at=None)
K("spiral", 11.9, 16.6, words=["绩效", "KPI", "排名", "考核", "学历", "房贷", "35 岁", "年终", "内推", "考编", "相亲", "比较"])
B(12.3, 16.3, [("越转越快，越转越小。", "sub")], per=0.08, y=TY + 360)
K("iris", 16.5, 17.7)
K("ring", 16.5, 19.5)
K("title", 16.5, 20.6, lines=[("内卷与躺平", "big"), ("第二期 · 给空心的人", "small")])
K("grid", 20.7, 27.1, ch="跑", fall_at=24.9)
B(27.2, 30.2, [("没人知道，终点在哪。", "main")], per=0.08)
K("counter", 30.3, 33.6)
K("chart", 33.7, 41.2, flat_at=36.6)
B(33.8, 37.4, [("你以为在往上爬，", "main")], per=0.07, y=TY - 200)
B(37.5, 41.2, [("其实只是原地加速。", "main")], per=0.07, y=TY - 200, uline=(0, 39.3))
B(41.3, 45.2, [("于是有人说：", "sub"), ("躺平吧。", "big")], per=0.12, line_gap=0.6, lie_at=(1, 43.4))
B(45.3, 49.6, [("可躺下之后，", "main"), ("心并没有放松。", "main")], per=0.08, line_gap=0.9, type_=True)

# ---- ACT 2 : deepening (49.6 - 93.4)
B(49.8, 53.7, [("因为卷和躺，", "main"), ("问的是同一个问题：", "main")], per=0.06, line_gap=0.8, drift=1)
B(53.8, 57.6, [("「我这样做，", "q"), ("到底值不值？」", "big")], per=0.1, line_gap=0.8, pop=True, glitch_at=56.6)
K("stone", 57.8, 77.0, keys=[(57.8, 0.03), (68.6, 0.93), (69.3, 0.93), (70.5, 0.02), (71.8, 0.02), (76.0, 0.55)],
  pkeys=[(57.8, -0.04), (68.6, 0.86), (70.5, 0.86), (71.6, -0.05), (76.0, 0.48)], warm=False)
B(57.9, 62.0, [("两千多年前，", "main"), ("有个人也问过。", "main")], per=0.07, line_gap=0.8, y=TY - 230)
B(62.1, 66.4, [("西西弗被诸神惩罚，", "main"), ("推一块巨石上山。", "main")], per=0.06, line_gap=0.8, y=TY - 230)
B(66.5, 70.6, [("石头到了山顶，", "main"), ("就会滚下来。", "main")], per=0.07, line_gap=0.8, y=TY - 230, fall_at=69.6)
B(70.8, 74.6, [("日复一日，", "main"), ("永无止境。", "main")], per=0.1, line_gap=0.8, y=TY - 230)
B(74.8, 77.0, [("诸神以为，这是最可怕的刑罚。", "sub")], per=0.05, y=TY - 230)
K("stars", 77.2, 93.6, warp_at=91.4)
B(77.4, 81.2, [("徒劳，", "big"), ("且无望。", "main")], per=0.14, line_gap=0.9, track=10)
B(81.4, 85.6, [("加缪说，这就是「荒诞」：", "sub"), ("人渴望意义，世界沉默。", "main")], per=0.07, line_gap=0.9)
B(85.8, 89.6, [("你的工位，", "main"), ("就是你的山。", "main")], per=0.09, line_gap=0.8, uline=(1, 88.0))
B(89.8, 93.2, [("你的 KPI，", "main"), ("就是你的石头。", "main")], per=0.09, line_gap=0.7, uline=(1, 91.6))
K("collapse", 92.4, 93.5)

# ---- ACT 3 : turn & landing (93.4 - 160)
K("ring", 93.5, 98.0)
B(93.5, 97.6, [("但加缪写下了", "main"), ("整本书最重要的一句：", "sub")], per=0.08, line_gap=0.9, pop=True)
K("morph", snap(97.8), 108.0, a="西西弗是幸福的。", b="因为他看清了石头。", tm=snap(102.6), size=88)
B(108.2, 112.4, [("他不再等诸神给答案，", "main"), ("不再问「值不值」。", "main")], per=0.06, line_gap=0.8)
B(112.5, 116.6, [("推，是他自己的选择。", "main"), ("这一刻，石头属于他。", "main")], per=0.06, line_gap=0.9, uline=(1, 114.8))
K("stone", 116.7, 129.2, keys=[(116.7, 0.03), (127.4, 0.97)], pkeys=[(116.7, -0.04), (127.4, 0.9)], warm=True)
B(116.8, 120.8, [("内卷，是被推着跑。", "main")], per=0.07, y=TY - 230)
B(120.9, 125.0, [("躺平，是被石头压住。", "main")], per=0.07, y=TY - 230)
B(125.1, 129.0, [("而你，可以选择怎么推。", "main")], per=0.07, y=TY - 230, uline=(0, 126.8))
B(129.2, 133.4, [("不必赢过所有人，", "main"), ("也不必假装不在乎。", "main")], per=0.06, line_gap=0.8, wipe_at=132.0)
K("acts", snap(133.5), 144.8, items=["给努力，设一个下班时间", "做一件不为 KPI 的事", "允许自己慢一点", "把「值不值」换成「想不想」"], gap=2.66)
B(144.9, 147.6, [("石头，还会滚下来。", "main")], per=0.08, fall_at=146.6)
K("particles", snap(147.7), 153.4, text="但山，是你的。", size=96)
B(153.5, 157.6, [("你现在推的石头，", "main"), ("是你自己选的吗？", "main"), ("评论区聊聊 · 下期：焦虑与自由", "small")], per=0.07, line_gap=0.7)
K("dot", 157.4, 160.0, mode="end", y=TY)

# -------------------------------------------------------------- precompute
NST = 650
stars = np.stack([np.random.uniform(-1, 1, NST) * 1.5, np.random.uniform(-1, 1, NST), np.random.uniform(0.05, 1, NST)], 1)

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

for e in E:
    if e["kind"] == "grid":
        nn = 13 * 9; e["order"] = list(np.random.permutation(nn)); e["fd"] = list(np.random.uniform(0, 0.9, nn)); e["fx"] = list(np.random.normal(0, 40, nn))
    if e["kind"] == "morph":
        n = 5000
        e["AX"], e["AY"], e["fa"] = text_points(e["a"], e["size"], n)
        e["BX"], e["BY"], e["fb"] = text_points(e["b"], e["size"], n)
        o = np.argsort(e["AX"]); e["AX"], e["AY"] = e["AX"][o], e["AY"][o]
        o = np.argsort(e["BX"]); e["BX"], e["BY"] = e["BX"][o], e["BY"][o]
        a_ = np.random.uniform(0, 2 * np.pi, n); r_ = np.random.uniform(500, 1300, n)
        e["SX"], e["SY"] = CX + np.cos(a_) * r_, TY + np.sin(a_) * r_
        e["PD"] = np.random.uniform(0, 1.2, n); e["ARC"] = np.random.normal(0, 120, n)
        e["UV"] = np.random.uniform(40, 160, n); e["UX"] = np.random.normal(0, 30, n)
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
            la = e.get("lie_at")
            if la and la[0] == li and t > la[1]:
                q = eout((t - la[1] - i * 0.08) / 0.7)
                sp = sprite(c, fx, rgb, a).rotate(-90 * q, resample=Image.BICUBIC)
                im.paste(sp, (int(xs[i] - sp.width / 2), int(y + q * fx.size * 0.25 - sp.height / 2)), sp); k += 1; continue
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
            r_in = 9 * (1 - eout(lt / 0.8)); r_out = 9 * ein((t - (e["t1"] - 0.55)) / 0.55); rad = max(r_in, r_out)
            if rad > 0.6:
                lay = Image.new("RGB", (W, H), tuple(BG)); draw_block(lay, ImageDraw.Draw(lay), t, e)
                im = ImageChops.lighter(im, lay.filter(ImageFilter.GaussianBlur(rad))); d = ImageDraw.Draw(im)
            else:
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
            sub = dict(t0=e["t0"] + 0.15, t1=e["t1"], lines=e.get("lines", [("虚无主义", "big"), ("N I H I L I S M", "small")]), per=0.16, line_gap=0.9, pop=True, track=30 - 20 * eout(lt / 3))
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
            m = 40 + p * 380; my = 30 + p * 250
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
        elif k == "spiral":
            ws = e["words"]; n = len(ws); prog = ein(lt / span)
            a_all = ease(lt / 0.6) * (1 - ease((lt - span + 0.5) / 0.5))
            for i, w in enumerate(ws):
                base = i / n * 2 * math.pi
                r = (520 - 440 * prog) * (0.55 + 0.45 * ((i * 7) % n) / n) * (1 - 0.15 * math.sin(lt * 2 + i))
                ang = base + lt * (0.4 + 3.5 * prog * prog) + r * 0.004
                x = CX + math.cos(ang) * r * 1.15; y = TY + math.sin(ang) * r * 0.8
                sz = max(14, int(46 * (0.4 + 0.6 * r / 520)))
                d.text((x, y), w, font=font(sz), fill=col(FG if i % 3 else WARM, a_all * cl(r / 140)), anchor="mm")
            rr = 6 + 4 * prog; d.ellipse([CX - rr, TY - rr, CX + rr, TY + rr], fill=col(FG, a_all))
            zoom += 0.06 * prog
        elif k == "grid":
            f = font(50); cols, rows = 13, 9; gx, gy = 100, 104
            a_out = 1 - ease((t - (e["t1"] - 0.5)) / 0.5)
            fa = e["fall_at"]; order = e["order"]
            for idx in range(cols * rows):
                cxi, ryi = idx % cols, idx // cols
                x = CX + (cxi - (cols - 1) / 2) * gx; y = TY + (ryi - (rows - 1) / 2) * gy
                s0 = e["t0"] + order[idx] * (fa - e["t0"] - 0.6) / (cols * rows)
                if t < s0: continue
                p = eout((t - s0) / 0.3)
                center = (cxi == cols // 2 and ryi == rows // 2)
                if center:
                    w = ease((t - fa) / 0.8); c = tuple(int(FG[j] + (WARM[j] - FG[j]) * w) for j in range(3))
                    ff = font(int(50 + 40 * w)); d.text((x, y), "你" if t > fa + 0.4 else e["ch"], font=ff, fill=col(c, p * a_out), anchor="mm")
                elif t > fa + e["fd"][idx]:
                    ft = t - fa - e["fd"][idx]; yy_ = y + 0.5 * 2200 * ft * ft
                    if yy_ < H + 60: d.text((x + e["fx"][idx] * ft, yy_), e["ch"], font=f, fill=col(DIM, (1 - cl(ft / 1.2)) * a_out), anchor="mm")
                else:
                    d.text((x, y), e["ch"], font=f, fill=col(DIM, 0.75 * p * a_out), anchor="mm")
        elif k == "counter":
            p = cl((lt - 0.2) / 1.6); mins = int(9 * 60 + p * 12 * 60)
            a = ease(lt / 0.3) * (1 - ein((t - (e["t1"] - 0.5)) / 0.5))
            s = f"{mins // 60:02d}:{mins % 60:02d}"
            d.text((CX, TY - 60), s, font=font(150), fill=col(FG, a), anchor="mm")
            if lt > 1.9:
                q = eout((lt - 1.9) / 0.5)
                d.text((CX + 330, TY - 60), "× 6", font=font(int(70 + 30 * (1 - q))), fill=col(WARM, q * a), anchor="mm")
            if lt > 2.2:
                draw_block(im, d, t, dict(t0=e["t0"] + 2.2, t1=e["t1"], lines=[("一周六天，一年三百多天。", "sub")], per=0.05, y=TY + 90))
            # progress bar
            d.line([CX - 300, TY + 30, CX - 300 + 600 * p, TY + 30], fill=col(DIM, a), width=2)
        elif k == "chart":
            x0, x1, yb, yt = CX - 480, CX + 480, TY + 330, TY + 60
            a = ease(lt / 0.5) * (1 - ein((t - (e["t1"] - 0.6)) / 0.6))
            d.line([x0, yb, x1, yb], fill=col(DIM, 0.5 * a), width=1); d.line([x0, yb, x0, yt - 20], fill=col(DIM, 0.5 * a), width=1)
            fl = e["flat_at"] - e["t0"]; pts = []
            hx = 0
            for i in range(121):
                u = i / 120
                tu = u * (span * 0.9)
                if tu > lt: break
                if tu < fl: xx_ = x0 + (u * span * 0.9 / fl) * 420; yy_ = yb - (tu / fl) ** 0.8 * 230
                else: xx_ = x0 + 420 + (tu - fl) / (span * 0.9 - fl) * 520; yy_ = yb - 230 - 6 * math.sin((tu - fl) * 9)
                pts.append((xx_, yy_))
            if len(pts) > 1: d.line(pts, fill=col(FG, a), width=3)
            if pts:
                hx, hy = pts[-1]; rr = 7 + (4 * abs(math.sin(t * 14)) if lt > fl else 0)
                d.ellipse([hx - rr, hy - rr, hx + rr, hy + rr], fill=col(WARM if lt > fl else FG, a))
        elif k == "stone":
            a = ease(lt / 0.8) * (1 - ease((t - (e["t1"] - 0.8)) / 0.8))
            sx0, sy0, sx1, sy1 = CX - 520, TY + 360, CX + 470, TY - 10
            L = math.hypot(sx1 - sx0, sy1 - sy0); ux, uy = (sx1 - sx0) / L, (sy1 - sy0) / L; nx, ny = uy, -ux
            p = eout(lt / 1.2)
            d.line([sx0, sy0, sx0 + (sx1 - sx0) * p, sy0 + (sy1 - sy0) * p], fill=col(DIM, 0.9 * a), width=2)
            d.line([sx1, sy1, sx1 + 160 * p, sy1], fill=col(DIM, 0.9 * a), width=2)
            def kf(keys):
                if t <= keys[0][0]: return keys[0][1]
                for (ta, va), (tb, vb) in zip(keys, keys[1:]):
                    if ta <= t <= tb:
                        u = (t - ta) / (tb - ta)
                        u = ein(u) if vb < va else (u * 0.85 + 0.15 * ease(u))
                        return va + (vb - va) * u
                return keys[-1][1]
            s = kf(e["keys"]); ps = kf(e["pkeys"]); R = 40
            cx_ = sx0 + ux * L * s + nx * (R + 2); cy_ = sy0 + uy * L * s + ny * (R + 2)
            sc = WARM if e["warm"] else FG
            d.ellipse([cx_ - R, cy_ - R, cx_ + R, cy_ + R], outline=col(sc, a), width=3)
            rot = s * L / R
            d.line([cx_, cy_, cx_ + math.cos(rot) * R * 0.7, cy_ + math.sin(rot) * R * 0.7], fill=col(sc, 0.6 * a), width=2)
            if e["warm"] and s > 0.9:
                g = ease((t - 127.4) / 0.8); gr = R + 10 + 30 * g
                d.ellipse([cx_ - gr, cy_ - gr, cx_ + gr, cy_ + gr], outline=col(WARM, 0.35 * g * a), width=2)
            px = sx0 + ux * L * ps; py = sy0 + uy * L * ps
            pushing = s - ps < 0.12
            lean = 26 if pushing else 4
            hx_, hy_ = px + nx * 64 + ux * lean, py + ny * 64 + uy * lean
            d.line([px, py, hx_, hy_], fill=col(FG, a), width=3)
            hcx, hcy = hx_ + nx * 12 + ux * 4, hy_ + ny * 12 + uy * 4
            d.ellipse([hcx - 10, hcy - 10, hcx + 10, hcy + 10], fill=col(FG, a))
            mx, my_ = px + nx * 44 + ux * lean * 0.7, py + ny * 44 + uy * lean * 0.7
            if pushing:
                d.line([mx, my_, cx_ - ux * R * 0.95, cy_ - uy * R * 0.95], fill=col(FG, 0.85 * a), width=2)
            else:
                d.line([mx, my_, mx - nx * 22 + ux * 8, my_ - ny * 22 + uy * 8], fill=col(FG, 0.85 * a), width=2)
        elif k == "iris":
            post.append(("iris", eout(lt / span)))
        elif k == "morph":
            n = len(e["AX"]); tm = e["tm"]
            p1 = 1 - (1 - np.clip((t - e["t0"] - e["PD"]) / 2.4, 0, 1)) ** 3
            x = e["SX"] + (e["AX"] - e["SX"]) * p1; y = e["SY"] + (e["AY"] - e["SY"]) * p1
            q = np.clip((t - tm - e["PD"] * 0.6) / 1.6, 0, 1); q = q * q * (3 - 2 * q)
            arc = np.sin(q * np.pi) * e["ARC"]
            x = x + (e["BX"] - e["AX"]) * q; y = y + (e["BY"] - e["AY"]) * q + arc
            rel = e["t1"] - 1.6; up = np.clip(t - rel - e["PD"] * 0.4, 0, None)
            x = x + e["UX"] * up; y = y - e["UV"] * up * up
            aa = np.clip(lt / 0.8, 0, 1) * np.clip(1 - up / 1.0, 0, 1)
            arr = np.array(im); xi = x.astype(int); yi = y.astype(int)
            ok = (xi >= 0) & (xi < W - 1) & (yi >= 0) & (yi < H - 1) & (aa > 0.01)
            warmq = q[ok, None]
            tgt = np.array(FG) + (np.array(WARM) - np.array(FG)) * warmq
            cc = (BG + (tgt - BG) * aa[ok, None]).astype(np.uint8)
            for ddx in (0, 1):
                for ddy in (0, 1): arr[yi[ok] + ddy, xi[ok] + ddx] = cc
            im = Image.fromarray(arr); d = ImageDraw.Draw(im)
            ta = ease((lt - 2.8) / 0.8) * (1 - ease((t - (tm - 0.3)) / 0.3))
            if ta > 0.01: d.text((CX, TY), e["a"], font=e["fa"], fill=col(FG, ta), anchor="mm")
            tb = ease((t - tm - 2.2) / 0.6) * (1 - ease((t - (rel - 0.4)) / 0.4))
            if tb > 0.01: d.text((CX, TY), e["b"], font=e["fb"], fill=col(WARM, tb), anchor="mm")
            d.text((CX, TY - 120), "「", font=font(60), fill=col(DIM, ease((lt - 1.5) / 0.8) * (1 - ease((t - tm) / 0.5))), anchor="mm")
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
    for kind, g in post:
        if kind == "iris":
            rr = 10 + g * math.hypot(W, H) / 2
            m = ((xx - CX) ** 2 + (yy - TY) ** 2) <= rr * rr
            arr = np.where(m[..., None], arr, BG.astype(np.float32))
            ring = np.abs(np.sqrt((xx - CX) ** 2 + (yy - TY) ** 2) - rr) < 1.5
            arr[ring] = arr[ring] * 0 + np.array(FG) * (1 - g)
    # arr = arr * VIG + grain[int(t * 12) % 6]
    arr = arr * VIG

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
    for e in E:
        k = e["kind"]; t0 = e["t0"]
        if k == "spiral": whoosh(t0, e["t1"] - t0, 0.28, fc=700)
        if k == "iris": whoosh(t0 - 0.2, 1.0, 0.2, rev=True, fc=1200)
        if k == "grid":
            for i in range(0, 40): tick(t0 + i * (e["fall_at"] - t0 - 0.6) / 40, 0.03)
            whoosh(e["fall_at"], 1.2, 0.2, rev=True, fc=500); tone(e["fall_at"] + 0.4, 523, 0.06, 2); tone(e["fall_at"] + 0.9, 50, 0.3, 3)
        if k == "counter":
            for i in range(12): tick(t0 + 0.2 + i * 0.135, 0.05)
            tone(t0 + 1.9, 196, 0.12, 2)
        if k == "chart": tone(t0, 330, 0.03, 0.5); tone(e["flat_at"], 1760, 0.03, 1)
        if k == "block" and e.get("lie_at"): tone(e["lie_at"][1] + 0.5, 60, 0.3, 4)
        if k == "stone":
            for (ta, va), (tb, vb) in zip(e["keys"], e["keys"][1:]):
                if vb < va: whoosh(ta, tb - ta + 0.3, 0.3, fc=250); tone(tb, 45, 0.4, 3)
            if e["warm"]: tone(127.4, 659, 0.08, 1.2); tone(127.4, 988, 0.05, 1.2)
        if k == "morph": whoosh(t0, 2.4, 0.16, fc=4000); whoosh(e["tm"], 1.8, 0.16, fc=3000); tone(e["tm"] + 2.0, 1318, 0.04, 2)
    fx = fx / max(1e-6, np.abs(fx).max()) * 0.5
    st = np.stack([fx, np.roll(fx, 220)], 1)
    with wave.open("sfx_ep2.wav", "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((st * 32767).astype(np.int16).tobytes())

def storyboard():
    rows = []
    for e in sorted(E, key=lambda e: e["t0"]):
        if e["kind"] == "block": txt = " / ".join(l[0] for l in e["lines"])
        elif e["kind"] in ("words", "list", "acts"): txt = " / ".join(e.get("words") or e.get("items"))
        elif e["kind"] == "particles": txt = e["text"]
        elif e["kind"] == "morph": txt = e["a"] + " → " + e["b"]
        elif e["kind"] == "spiral": txt = " / ".join(e["words"])
        elif e["kind"] == "title": txt = "虚无主义 NIHILISM"
        else: txt = ""
        fxs = [k for k in ("type_", "fall_at", "glitch_at", "uline", "strike", "wipe_at", "pop") if e.get(k)]
        rows.append(dict(t0=round(e["t0"], 1), t1=round(e["t1"], 1), kind=e["kind"], text=txt, fx=fxs))
    json.dump(rows, open("storyboard_ep2.json", "w"), ensure_ascii=False, indent=0)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "pv":
        for s in sys.argv[2:]:
            Image.fromarray(draw(float(s))).resize((480, 360)).save(f"e2_{s}.png")
        sys.exit()
    storyboard(); build_audio()
    p = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                          "-i", "bgm.mp3", "-i", "sfx_ep2.wav",
                          "-filter_complex", "[1:a]volume=1.0[m];[2:a]volume=0.45[s];[m][s]amix=inputs=2:normalize=0,afade=t=out:st=157.5:d=2.5,loudnorm=I=-14:TP=-1:LRA=11[a]",
                          "-map", "0:v", "-map", "[a]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", "-tune", "grain",
                          "-maxrate", "4M", "-bufsize", "8M", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-t", str(DUR),
                          "-movflags", "+faststart", "ep2_neijuan_4x3.mp4"], stdin=subprocess.PIPE)
    for i in range(int(FPS * DUR)):
        p.stdin.write(draw(i / FPS).tobytes())
    p.stdin.close(); p.wait(); print("done", p.returncode)
