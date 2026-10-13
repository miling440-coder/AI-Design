# -*- coding: utf-8 -*-
# 《露营是最小成本的出逃》 · 轻动画版 v2 · 治愈系 × 3D 自然场景 × 2D 叙事 × 连续镜头转场 × 小猫陪伴
# 无旁白、无字幕，只有画面、音效与音乐
# 用法：
#   python scene_camp.py bench                     测速
#   python scene_camp.py pv 26 30 34.6 37 70       导出这些秒数的预览图 pv_*.png（gpu_pv 同义）
#   python scene_camp.py seg 0 900 test.mp4        只渲染第 0~900 帧（无声）
#   python scene_camp.py full                      一键：渲染 → 音效 → 混音 → 响度标准化
#   python scene_camp.py audio video_tmp.mp4       只重做声音（画面已渲染好时）
# 低分辨率快速预览：先设环境变量 EP_SCALE=0.5
import os, sys, math, json, time, subprocess, numpy as np
import engine as E
from engine import (DW, DH, FPS, cl, sm, ease, eio, eout, ein, lerp, win, kf, kf_dict, Mask, over, add,
                    gblur, haze, mix, draw_text, splat, glow_points, hash1)

NS = 5.9                          # 新增“回家”段后，自然段及之后整体后移的秒数（自然段内部仍用旧时间轴 tn = t - NS）
DUR = 116.0 + NS
OUT_NAME = "camp_escape_4x3.mp4"
BAR = int(os.environ.get("EP_BAR", "0"))   # 轻动画版默认不加电影遮幅（无字幕）；想要遮幅可设 EP_BAR=60
BGM_FILE = "bgm.mp3"              # Windswept - Kevin MacLeod (incompetech.com) · CC BY 4.0

# ======================================================================= 角色（2D 剪影，正向运动学）
def pose(name, ph=0.0):
    s, c = math.sin(ph), math.cos(ph)
    P = dict(lean=2, head=0, ua1=8, la1=4, ua2=-6, la2=-4, th1=3, sh1=0, th2=-3, sh2=0)
    if name == "walk":
        P.update(lean=5, head=6, ua1=-20 * s, la1=-20 * s + 18, ua2=20 * s, la2=20 * s + 18,
                 th1=24 * s, sh1=24 * s - 38 * max(0, c), th2=-24 * s, sh2=-24 * s - 38 * max(0, -c))
    elif name == "trudge":
        P.update(lean=10, head=22, ua1=-10 * s + 4, la1=-10 * s + 10, ua2=10 * s + 4, la2=10 * s + 10,
                 th1=20 * s, sh1=20 * s - 30 * max(0, c), th2=-20 * s, sh2=-20 * s - 30 * max(0, -c))
    elif name == "phone":
        P.update(lean=3, head=24, ua1=14, la1=150, ua2=-4, la2=-2)
    elif name == "stand":
        P.update(lean=1, head=-4)
    elif name == "sit":
        P.update(lean=-6, head=-2, ua1=62, la1=104, ua2=56, la2=98, th1=128, sh1=6, th2=122, sh2=2)
    elif name == "situp":   # 抬头看天
        P.update(lean=-14, head=-34, ua1=-40, la1=-30, ua2=-46, la2=-36, th1=128, sh1=6, th2=118, sh2=0)
    elif name == "stretch":
        P.update(lean=-4, head=-26, ua1=168, la1=176, ua2=160, la2=172)
    elif name == "sofa":    # 坐在沙发上低头看手机
        P.update(lean=-8 + 2 * s, head=26, ua1=24, la1=158, ua2=14, la2=150, th1=96, sh1=4, th2=90, sh2=-4)
    return P

def _dir(deg): r = math.radians(deg); return np.array([math.sin(r), math.cos(r)])

def figure(m, x, yg, h, P, flip=False, pack=True, hood=False, phone=None):
    """在 Mask m 上画一个人。x,yg 为脚底（最低点）位置，h 为身高（设计单位）。返回关键点"""
    hip = np.zeros(2); lean = P["lean"]
    sh = hip + 0.31 * _dir(180 - lean)
    head = sh + 0.115 * _dir(180 - lean - P["head"])
    k1 = hip + 0.245 * _dir(P["th1"]); f1 = k1 + 0.235 * _dir(P["sh1"])
    k2 = hip + 0.245 * _dir(P["th2"]); f2 = k2 + 0.235 * _dir(P["sh2"])
    e1 = sh + 0.17 * _dir(P["ua1"]); h1 = e1 + 0.16 * _dir(P["la1"])
    e2 = sh + 0.17 * _dir(P["ua2"]); h2 = e2 + 0.16 * _dir(P["la2"])
    pts = [hip, sh, head, k1, f1, k2, f2, e1, h1, e2, h2]
    low = max(p[1] for p in pts if p is not head)
    sgn = -1 if flip else 1
    def T(p): return (x + sgn * p[0] * h, yg + (p[1] - low) * h)
    lw = 0.062 * h
    # 背包（身后）
    tv = sh - hip; tl = np.linalg.norm(tv); tn = tv / tl; nb = np.array([-tn[1], tn[0]])  # 背向（朝 -x）
    if pack:
        c0 = hip + tn * 0.07 - nb * 0.0; bp = [c0 - nb * 0.0 + nb * 0.0, c0 + tn * 0.24, c0 + tn * 0.24 + nb * 0.11, c0 + nb * 0.12]
        bp = [hip + tn * 0.06 - nb * 0.035, hip + tn * 0.29 - nb * 0.035, hip + tn * 0.29 - nb * 0.125, hip + tn * 0.08 - nb * 0.13]
        m.poly([T(p) for p in bp])
    # 躯干
    tw0, tw1 = 0.055, 0.068
    torso = [hip + nb * tw0, sh + nb * tw1, sh - nb * tw1, hip - nb * tw0]
    m.poly([T(p) for p in torso])
    for a, b, w in ((hip, k1, 1.25), (k1, f1, 1.0), (hip, k2, 1.25), (k2, f2, 1.0), (sh, e1, 0.95), (e1, h1, 0.85), (sh, e2, 0.95), (e2, h2, 0.85)):
        m.line([T(a), T(b)], lw * w)
    m.circle(T(sh), lw * 1.05)
    m.circle(T(hip), lw * 0.95)
    hr = 0.072 * h * (1.12 if hood else 1.0)
    m.circle(T(head), hr)
    m.line([T(sh), T(head)], lw * 0.8)
    if phone is not None:
        hp = T(h1); m.rrect(hp[0] - 0.02 * h, hp[1] - 0.05 * h, hp[0] + 0.02 * h, hp[1] + 0.012 * h, 0.006 * h, phone)
    return dict(head=T(head), hand=T(h1), sh=T(sh), hip=T(hip), hr=hr)

def bust(m, x, y, s, lean=0.0, arm=None, v=255):
    """人群胸像：头 + 颈 + 肩。y 为头心，s 为尺度"""
    hx = x + lean
    m.ellipse((hx, y), (0.36 * s, 0.44 * s), 0, v)
    m.poly([(hx - 0.14 * s, y + 0.3 * s), (hx + 0.14 * s, y + 0.3 * s), (hx + 0.2 * s, y + 0.62 * s), (hx - 0.2 * s, y + 0.62 * s)], v)
    sh = [(x - 0.95 * s, y + 3.2 * s), (x - 0.92 * s, y + 0.95 * s), (x - 0.55 * s + lean * 0.6, y + 0.62 * s), (x + 0.55 * s + lean * 0.6, y + 0.62 * s),
          (x + 0.92 * s, y + 0.95 * s), (x + 0.95 * s, y + 3.2 * s)]
    m.poly(sh, v)
    if arm is not None:   # 举手拉吊环
        ax, ay = arm
        m.line([(x + 0.7 * s, y + 0.9 * s), (ax, ay)], 0.26 * s, v)

def rim_light(mask, dx, dy):
    """剪影描边光：mask 减去朝光源方向平移后的 mask"""
    sh = E.shift(mask[None], -dx, -dy)[0]
    return (mask - sh).clamp(0, 1)

# ======================================================================= 小猫（2D 剪影 + 平涂上色）
def cat(m, x, yg, s, pose_name="sit", ph=0.0, flip=False, t=0.0, look=0.0):
    """在 Mask m 上画一只小猫。x,yg 为脚底中心，s 为肩高（设计单位）。返回头部位置 (hx, hy, r)
    pose: walk / sit / loaf / lookup / paw / stretch / jump"""
    sg = -1 if flip else 1
    def T(px, py): return (x + sg * px * s, yg + py * s)
    tail_sw = math.sin(t * 2.2 + ph * 0.1)
    if pose_name == "walk" or pose_name == "jump":
        bob = 0.03 * math.sin(ph * 2) if pose_name == "walk" else 0.0
        m.ellipse(T(0, -0.62 + bob), (0.56 * s, 0.25 * s), 0, 255)
        hx, hy, r = 0.62, -0.92 + bob, 0.25
        if pose_name == "walk":
            for k, (lx, p0) in enumerate(((0.36, 0.0), (0.26, math.pi), (-0.36, math.pi * 0.5), (-0.26, math.pi * 1.5))):
                sw = 0.16 * math.sin(ph + p0)
                m.line([T(lx, -0.55 + bob), T(lx + sw, -0.02)], 0.11 * s)
                m.circle(T(lx + sw, -0.03), 0.06 * s)
        else:   # 起跳/落下：前腿前伸、后腿后蹬
            m.line([T(0.35, -0.6), T(0.75, -0.35)], 0.11 * s); m.line([T(-0.35, -0.6), T(-0.75, -0.4)], 0.12 * s)
        tail = [T(-0.5 - 0.25 * u, -0.66 - 0.55 * u + 0.12 * math.sin(u * 3 + t * 3) * u) for u in np.linspace(0, 1, 9)]
        m.line(tail, 0.09 * s)
    elif pose_name in ("sit", "lookup", "paw"):
        m.circle(T(-0.12, -0.3), 0.32 * s)                              # 后腿/臀
        m.ellipse(T(0.02, -0.58), (0.24 * s, 0.42 * s), sg * 12, 255)   # 身体
        m.line([T(0.16, -0.55), T(0.2, -0.03)], 0.11 * s)                # 前腿
        m.circle(T(0.21, -0.04), 0.07 * s)
        if pose_name == "paw":
            m.line([T(0.12, -0.6), T(0.42, -0.86)], 0.1 * s); m.circle(T(0.44, -0.88), 0.07 * s)
        else:
            m.line([T(0.06, -0.55), T(0.08, -0.03)], 0.11 * s)
        if pose_name == "lookup": hx, hy, r = 0.08, -1.08, 0.25
        else: hx, hy, r = 0.14 + 0.03 * look, -1.02, 0.25
        tail = [T(-0.32 + 0.7 * u, -0.04 - 0.05 * u - (0.22 * (u - 0.7) / 0.3 * (1 + 0.6 * tail_sw) if u > 0.7 else 0)) for u in np.linspace(0, 1, 10)]
        m.line(tail, 0.09 * s)
    elif pose_name == "loaf":
        breath = 0.015 * math.sin(t * 1.6)
        m.ellipse(T(-0.02, -0.3), ((0.55 + breath) * s, (0.3 + breath) * s), 0, 255)
        hx, hy, r = 0.42, -0.4, 0.23
        tail = [T(-0.5 + 0.9 * u, -0.05 + 0.04 * math.sin(u * 3)) for u in np.linspace(0, 1, 9)]
        m.line(tail, 0.09 * s)
    else:   # stretch：前低后高
        m.ellipse(T(0.0, -0.5), (0.6 * s, 0.22 * s), sg * 14, 255)
        m.line([T(0.4, -0.3), T(0.85, -0.03)], 0.11 * s); m.line([T(0.3, -0.3), T(0.75, -0.03)], 0.11 * s)
        m.line([T(-0.42, -0.62), T(-0.45, -0.02)], 0.12 * s); m.line([T(-0.3, -0.6), T(-0.32, -0.02)], 0.12 * s)
        hx, hy, r = 0.68, -0.42, 0.24
        tail = [T(-0.55 - 0.15 * u, -0.7 - 0.5 * u) for u in np.linspace(0, 1, 7)]
        m.line(tail, 0.09 * s)
    hc = T(hx, hy); R = r * s
    m.circle(hc, R)
    m.ellipse((hc[0] + sg * 0.16 * R, hc[1] + 0.35 * R), (0.62 * R, 0.5 * R), 0, 255)   # 腮
    for ex in (-0.62, 0.42):
        ex0 = ex * sg
        m.poly([(hc[0] + (ex0 - 0.25) * R, hc[1] - 0.55 * R), (hc[0] + (ex0 + 0.05 * sg) * R, hc[1] - 1.45 * R), (hc[0] + (ex0 + 0.42) * R, hc[1] - 0.6 * R)])
    return dict(head=hc, r=R, flip=flip)

CAT_FUR = (0.96, 0.62, 0.30)      # 橘猫
def paint_cat(img, a, amb, key=(1.0, 0.85, 0.6), key_k=0.5, rim_col=None, rim_k=0.0, rim_dir=(3.0, -1.5), extra=0.0, eye=None, eye_k=0.0):
    """平涂上色：底色 × 环境光 + 顶部受光渐变 + 描边光；eye=(x,y,r,flip) 时在夜里画一点眼睛反光"""
    if not torch.is_tensor(amb): amb = E.C(amb)
    base = E.C(CAT_FUR) * (amb * 0.85 + 0.12 + extra)
    top = (a - E.shift(a[None], 0, 4)[0]).clamp(0, 1)                 # 上沿高光
    img = over(img, base, a)
    img = add(img, E.C(key), top * key_k * 0.35)
    if rim_col is not None and rim_k > 0:
        img = add(img, E.C(rim_col), rim_light(a, *rim_dir) * rim_k)
    if eye is not None and eye_k > 0.01:
        ex, ey, er, fl = eye
        sg = -1 if fl else 1
        em = Mask().ellipse((ex + sg * 0.55 * er, ey - 0.05 * er), (0.13 * er, 0.17 * er)).t(0.6)
        img = add(img, E.C((0.85, 1.0, 0.55)), em * eye_k * 1.6)
    return img

# ======================================================================= 自然场景（3D 着色）
HOR = 468.0
PAL = {
    "golden": dict(top=(0.22, 0.38, 0.62), mid=(0.70, 0.62, 0.56), hor=(0.97, 0.72, 0.46), sun=(1.0, 0.80, 0.52), sunx=1110., suny=300., sunr=22.,
                   glow=0.8, amb=1.0, ambc=(1.0, 0.92, 0.78), mtn=(0.40, 0.36, 0.44), haze=0.5, g1=(0.58, 0.52, 0.20), g2=(0.20, 0.24, 0.07),
                   water=(0.16, 0.26, 0.32), stars=0., mw=0., cloud=0.75, cl=(1.0, 0.86, 0.68), cs=(0.62, 0.62, 0.72), mist=0.08, rays=0.55, night=0.),
    "sunset": dict(top=(0.17, 0.23, 0.45), mid=(0.86, 0.52, 0.42), hor=(1.0, 0.57, 0.28), sun=(1.0, 0.62, 0.30), sunx=1150., suny=452., sunr=33.,
                   glow=1.3, amb=0.62, ambc=(1.0, 0.72, 0.55), mtn=(0.30, 0.22, 0.32), haze=0.5, g1=(0.33, 0.28, 0.12), g2=(0.12, 0.10, 0.06),
                   water=(0.12, 0.12, 0.20), stars=0., mw=0., cloud=0.9, cl=(1.0, 0.60, 0.42), cs=(0.45, 0.35, 0.50), mist=0.05, rays=0.7, night=0.),
    "blue":   dict(top=(0.035, 0.055, 0.13), mid=(0.17, 0.16, 0.30), hor=(0.42, 0.30, 0.38), sun=(0.9, 0.4, 0.3), sunx=1150., suny=560., sunr=30.,
                   glow=0.22, amb=0.22, ambc=(0.6, 0.7, 1.0), mtn=(0.10, 0.10, 0.17), haze=0.45, g1=(0.08, 0.10, 0.11), g2=(0.03, 0.04, 0.05),
                   water=(0.05, 0.07, 0.12), stars=0.35, mw=0.2, cloud=0.5, cl=(0.6, 0.4, 0.5), cs=(0.15, 0.14, 0.25), mist=0.1, rays=0., night=0.6),
    "night":  dict(top=(0.008, 0.014, 0.042), mid=(0.025, 0.04, 0.09), hor=(0.07, 0.09, 0.17), sun=(0.9, 0.4, 0.3), sunx=1150., suny=900., sunr=30.,
                   glow=0., amb=0.19, ambc=(0.55, 0.65, 1.0), mtn=(0.06, 0.07, 0.12), haze=0.35, g1=(0.07, 0.09, 0.12), g2=(0.035, 0.045, 0.065),
                   water=(0.02, 0.03, 0.06), stars=1., mw=1., cloud=0.22, cl=(0.25, 0.27, 0.40), cs=(0.04, 0.05, 0.09), mist=0.05, rays=0., night=1.),
    "dawn":   dict(top=(0.20, 0.31, 0.52), mid=(0.72, 0.62, 0.64), hor=(0.97, 0.74, 0.58), sun=(1.0, 0.83, 0.62), sunx=390., suny=440., sunr=28.,
                   glow=1.1, amb=0.78, ambc=(1.0, 0.89, 0.80), mtn=(0.38, 0.38, 0.48), haze=0.66, g1=(0.34, 0.40, 0.19), g2=(0.14, 0.18, 0.08),
                   water=(0.30, 0.34, 0.42), stars=0., mw=0., cloud=0.6, cl=(1.0, 0.82, 0.72), cs=(0.55, 0.55, 0.66), mist=1.0, rays=0.5, night=0.),
}
TOD = [(0, "golden"), (50, "golden"), (61, "sunset"), (66.5, "blue"), (71, "night"), (84.5, "night"), (88.5, "dawn"), (200, "dawn")]
def tod_at(t, force=None):
    if force: return dict(PAL[force])
    return kf_dict(t, [(a, PAL[b]) for a, b in TOD])

# 镜头：CX（物件空间 x）、TILT（仰角像素，正=抬头）、Z
NCAM = [(30.0, (640, 0, 1.0)), (35.0, (700, 0, 1.0)), (45.6, (1740, 0, 1.0)), (50.0, (1880, 10, 1.04)), (56.0, (1960, 24, 1.0)),
        (64.0, (1975, 30, 1.0)), (70.0, (1945, 42, 1.09)), (75.6, (1945, 42, 1.09)), (80.6, (1945, 650, 1.0)), (85.4, (1945, 650, 1.0)),
        (89.2, (1990, 24, 1.0)), (93.8, (2050, 0, 1.32))]
def ncam(t): return kf(t, NCAM)

TENT_X = 2100.0; FIRE_X = 1890.0; STOVE_X = 1845.0
def hero_state(t):
    """返回 (x, pose 名, 相位, 是否背包)"""
    if t < 35.5: return 700.0, "stand", 0.0, True
    if t < 46.0:
        u = (t - 35.5) / 10.5; x = 700 + 1080 * (u - 0.04 * math.sin(u * math.pi))
        return x, "walk", 2 * math.pi * 1.55 * t, True
    if t < 49.6: return 1780.0, "stand", 0.0, t < 47.2
    if t < 77.0: return 1800.0, "sit", 0.0, False
    if t < 86.4: return 1800.0, "situp", 0.0, False
    if t < 88.2: return 1800.0, "stand", 0.0, False
    return 1800.0, "stretch", 0.0, False

def cat_state(t):
    """小猫在自然段的走位（旧时间轴）：返回 (x, pose, 相位, flip, look)"""
    if t < 35.9: return 640.0, "sit", 0.0, False, 0.0
    if t < 46.4:
        hx = hero_state(max(35.5, t - 0.45))[0]
        return hx - 78 + 6 * math.sin(t * 1.3), "walk", 2 * math.pi * 2.4 * t, False, 0.0
    if t < 49.8: return 1712.0 + 4 * ease((t - 46.4) / 0.6), "sit", 0.0, False, math.sin(t * 0.8)
    if t < 52.6:   # 小跑到主人身边
        u = ease((t - 49.8) / 1.0); return lerp(1716, 1742, u), ("walk" if t < 50.8 else "sit"), 2 * math.pi * 2.4 * t, False, 0.0
    if t < 54.6: return 1742.0, ("paw" if math.sin((t - 52.6) * 5.5) > 0.2 else "sit"), 0.0, False, 0.0   # 拍风里飘来的花粉
    if t < 58.5: return 1742.0, "sit", 0.0, False, 0.0
    if t < 66.5: return 1748.0, "loaf", 0.0, False, 0.0          # 日落：靠着主人打盹
    if t < 72.0: return 1742.0, "sit", 0.0, False, 0.0
    if t < 76.0: return 1742.0, ("paw" if math.sin((t - 72) * 2.6) > 0.55 else "lookup"), 0.0, False, 0.0   # 够萤火虫
    if t < 86.4: return 1742.0, "lookup", 0.0, False, 0.0        # 和主人一起看星星
    if t < 88.4: return 1742.0, "sit", 0.0, False, 0.0
    if t < 90.6: return 1742.0, "stretch", 0.0, False, 0.0       # 一起伸懒腰
    return 1742.0, "sit", 0.0, False, 0.0

def gust(t):
    return 0.55 + 0.9 * win(t, 36.4, 42.5, 1.0, 2.0) + 0.5 * win(t, 51.5, 54.5, 0.8, 1.5) + 0.3 * win(t, 86, 93, 1, 1)

def wind(u, t):
    g = gust(t)
    return g * (0.55 * torch.sin(t * 2.1 + u * 0.012) + 1.6 * (E.fbm1(u * 0.0025 - t * 0.45, 3, 91) - 0.5)) + 0.22 * torch.sin(t * 1.4 + u * 0.035)

torch = None
def _sky(Xq, Yq, hor, P, CX, t):
    s = ((hor - Yq) / 620.0).clamp(0, 1.4)
    c = mix(E.C(P["hor"]), E.C(P["mid"]), sm(0.0, 0.3, s)[None])
    c = mix(c, E.C(P["top"]), sm(0.22, 1.05, s)[None])
    sx = P["sunx"] - (CX - 1500) * 0.03; sy = P["suny"] + (hor - HOR)
    d = torch.sqrt((Xq - sx) ** 2 + (Yq - sy) ** 2)
    g = P["glow"]
    c = c + E.C(P["sun"]) * ((torch.exp(-d / 70) * 0.85 + torch.exp(-d / 300) * 0.32 + torch.exp(-d / 900) * 0.10) * g)[None]
    c = c + E.C((1.0, 0.97, 0.9)) * (sm(P["sunr"] + 1.5, P["sunr"] - 1.5, d) * 1.4 * min(1.0, g))[None]
    # 云（拉长的卷云）
    cxn = Xq * 0.0016 + CX * 0.0004 + t * 0.006; cyn = (Yq - hor) * 0.0068
    n = E.fbm(cxn, cyn, 5, 11)
    band = sm(0.04, 0.22, s) * sm(1.15, 0.5, s)
    ca = sm(0.52, 0.80, n) * band * P["cloud"]
    lit = (torch.exp(-d / 520) * 1.25 + 0.2).clamp(0, 1)
    c = mix(c, mix(E.C(P["cs"]), E.C(P["cl"]), lit[None]), (ca * 0.85)[None])
    # 银河 + 星
    if P["stars"] > 0.01:
        a = math.radians(-33)
        px = Xq - 720 + CX * 0.02; py = (Yq - hor) + 560
        along = px * math.cos(a) + py * math.sin(a); dist = -px * math.sin(a) + py * math.cos(a)
        mwm = torch.exp(-(dist / 190) ** 2)
        if P["mw"] > 0.01:
            st = E.fbm(along * 0.0045, dist * 0.011, 5, 23)
            dust = E.fbm(along * 0.006 + 4, dist * 0.02, 4, 29) * torch.exp(-(dist / 60) ** 2)
            core = torch.exp(-((along - 80) / 420) ** 2)
            mwv = (mwm * (0.06 + 0.30 * st ** 2) - dust * 0.16).clamp(min=0) * P["mw"] * sm(0.05, 0.25, s)
            mcol = mix(E.C((0.62, 0.68, 0.92)), E.C((1.0, 0.84, 0.70)), (core * 0.6)[None])
            c = c + mcol * (mwv * (1 - ca))[None]
        cs = 5.0
        qx = (Xq + CX * 0.02) / cs; qy = (Yq - hor) / cs
        ix = torch.floor(qx).long(); iy = torch.floor(qy).long()
        h = E.ihash(ix, iy, 77)
        dens = 0.014 + 0.055 * mwm
        pxs = E.ihash(ix, iy, 78) * 0.8 + 0.1; pys = E.ihash(ix, iy, 79) * 0.8 + 0.1
        dd = ((qx - ix - pxs) ** 2 + (qy - iy - pys) ** 2) * cs * cs
        sig = max(0.85, 1.05 / E.PX)
        b = E.ihash(ix, iy, 80) ** 4 * 2.6 + 0.12
        tw = 0.85 + 0.15 * torch.sin(t * (0.5 + 1.2 * E.ihash(ix, iy, 81)) + E.ihash(ix, iy, 82) * 6.28)
        star = (h < dens).float() * torch.exp(-dd / (2 * sig * sig)) * b * tw * P["stars"] * sm(0.02, 0.12, s) * (1 - ca)
        scol = mix(E.C((0.80, 0.86, 1.0)), E.C((1.0, 0.88, 0.72)), (E.ihash(ix, iy, 83) > 0.7).float()[None])
        c = c + scol * star[None]
    return c, d

def _ridge(Xq, layer, CX, z, t, city=0.0):
    r, h0, amp, fq, seed, trees = layer
    zl = 1 + (z - 1) * r
    u = (Xq - 720) / zl + CX * r + seed * 1000
    n = E.fbm1(u * fq, 5, seed)
    rid = 1 - (2 * n - 1).abs()
    hgt = h0 + amp * (0.55 * n + 0.45 * rid ** 1.5)
    if trees:
        tu = u / 13.0; ti = torch.floor(tu); f = tu - ti; ii = ti.long()
        zone = sm(0.35, 0.6, E.fbm1(u * 0.004, 3, seed + 5))
        th = (E.ihash(ii, ii * 0 + 3, seed) > 0.22).float() * (12 + 26 * E.ihash(ii, ii * 0 + 4, seed)) * zone
        hgt = hgt + th * (1 - (f - 0.5).abs() * 2.3).clamp(min=0)
    if city > 0:
        bu = u / 46.0; bi = torch.floor(bu).long(); bf = bu - torch.floor(bu)
        bh = 30 + amp * 1.25 * E.ihash(bi, bi * 0 + 9, seed) ** 1.6
        bh = torch.where((bf > 0.06) & (bf < 0.94), bh, bh * 0.3)
        hgt = hgt * (1 - city) + bh * city
    return hgt * zl

MTN = [(0.08, 70, 240, 0.0021, 3, False), (0.18, 28, 150, 0.0033, 5, False), (0.32, 4, 58, 0.0052, 7, True)]

def _upper(Xq, Yq, hor, P, CX, z, t, city=0.0):
    c, d = _sky(Xq, Yq, hor, P, CX, t)
    rimk = P["glow"] * (0.25 + 0.75 * torch.exp(-(Xq - (P["sunx"] - (CX - 1500) * 0.03)).abs() / 500))
    hz = [P["haze"], P["haze"] * 0.6, P["haze"] * 0.25]
    dark = [0.85, 0.62, 0.38]
    occ = torch.zeros_like(Xq)
    for k, L in enumerate(MTN):
        ry = hor - _ridge(Xq, L, CX, z, t, city * (1.0 if k < 2 else 0.6))
        m = sm(ry - 0.9, ry + 0.9, Yq) * (Yq <= hor + 2).float()
        base = E.C(P["mtn"]) * dark[k] * (1.0 + 0.25 * (2 - k) * P["night"])
        if city > 0: base = mix(base, E.C((0.20, 0.23, 0.29)) * (P["amb"] * 0.9 + 0.1), city)
        tex = E.fbm(Xq * 0.012 + k * 9, Yq * 0.02, 3, 40 + k)
        col = base * (0.94 + 0.12 * tex)[None]
        valley = torch.exp(-(hor - Yq).clamp(min=0) / (35 + 20 * k))
        col = mix(col, E.C(P["hor"]) * (0.75 + 0.25 * P["amb"]), (hz[k] * 0.7 + valley * (0.35 + 0.4 * P["mist"]) * (1 - 0.3 * k)).clamp(0, 1)[None])
        rim = sm(ry + 4.5, ry, Yq) * rimk * (0.35 if k == 0 else 0.6)
        col = col + E.C(P["sun"]) * (rim * m)[None] * 0.8
        if city > 0 and k < 2:   # 天际线亮窗
            wx = torch.frac(Xq * 0.11); wy = torch.frac(Yq * 0.09)
            lit = (E.ihash(torch.floor(Xq * 0.11).long(), torch.floor(Yq * 0.09).long(), 5 + k) < 0.18).float()
            col = col + E.C((1.0, 0.86, 0.6)) * (lit * (wx > 0.35).float() * (wy > 0.4).float() * city * 0.35 * (Yq > ry + 6).float())[None]
        c = mix(c, col, m[None]); occ = torch.maximum(occ, m)
    # 地平线薄雾
    if P["mist"] > 0.02:
        mn = E.fbm(Xq * 0.004 + t * 0.03, (Yq - hor) * 0.025, 4, 61)
        mm = torch.exp(-(hor - Yq).clamp(min=0) / 55) * (0.4 + 0.6 * mn) * P["mist"] * 0.75
        c = mix(c, E.C(P["hor"]) * 0.95 + 0.05, mm.clamp(0, 1)[None])
    return c, occ, d

REFL_DOWN = int(os.environ.get("EP_REFL", "2"))   # 湖面倒影降采样倍数，1 = 全分辨率

def nature(t, tod=None, cam=None, fire=None, show_hero=True, city=0.0, extra_tilt=0.0, hero=None, catst=None):
    """渲染自然场景一帧，返回 (img, info)"""
    P = tod if tod is not None else tod_at(t)
    CX, TILT, Z = cam if cam is not None else ncam(t)
    TILT += extra_tilt
    X, Y = E.X, E.Y
    hor = 540 + (HOR - 540) * Z + TILT
    shore = hor + 112 * Z
    img, occ, dsun = _upper(X, Y, hor, P, CX, Z, t, city)
    amb = E.C(P["ambc"]) * P["amb"]
    sxs = P["sunx"] - (CX - 1500) * 0.03
    # ---------------- 湖面（倒影重算上半场景）
    r0 = int(max(0, min(E.H, math.floor(hor * E.PX)))); r1 = int(max(0, min(E.H, math.ceil((shore + 14) * E.PX))))
    if r1 > r0:
        Xc = X[r0:r1]; Yc = Y[r0:r1]
        k = ((Yc - hor) / (shore - hor + 1e-3)).clamp(0, 1)
        rip = (E.vnoise(Xc * 0.02 + t * 0.3, (Yc - hor) * 0.28 - t * 0.9, 51) - 0.5)
        Xr = Xc + torch.sin((Yc - hor) * 0.9 + t * 2.2) * 1.2 * k + rip * 10 * k
        Ym = 2 * hor - Yc + rip * 7 * k
        if REFL_DOWN > 1 and (r1 - r0) >= 2 * REFL_DOWN:
            # 倒影半分辨率重算再双线性放大：水面本身就有波纹和模糊，肉眼看不出差别，算量降到约 1/4
            q = REFL_DOWN
            rc, _, _ = _upper(Xr[::q, ::q], Ym[::q, ::q], hor, P, CX, Z, t, city)
            rc = torch.nn.functional.interpolate(rc[None], size=(r1 - r0, Xc.shape[1]), mode="bilinear", align_corners=False)[0]
        else:
            rc, _, _ = _upper(Xr, Ym, hor, P, CX, Z, t, city)
        fres = 0.82 - 0.38 * k
        wc = E.C(P["water"]) * (0.6 + 0.4 * P["amb"])
        col = wc + (rc - wc) * fres[None]
        lines = (E.vnoise(Xc * 0.006, (Yc - hor) * 0.6 - t * 0.25, 53) - 0.5) * 0.12 * k
        col = col * (1 + lines)[None]
        glit = sm(0.70, 0.95, E.vnoise(Xc * 0.035 + t * 0.35, (Yc - hor) * 0.45 + t * 0.8, 55)) * torch.exp(-(Xc - sxs).abs() / (30 + (Yc - hor) * 2.2))
        sunup = sm(hor + 20, hor - 40, P["suny"] + (hor - HOR))
        col = col + E.C(P["sun"]) * (glit * 1.0 * P["glow"] * sunup)[None]
        img[:, r0:r1] = mix(img[:, r0:r1], col, sm(hor - 0.5, hor + 0.8, Yc)[None])
    # ---------------- 草地底色
    gd = ((Y - shore) / (DH + 80 - shore)).clamp(0, 1)
    gtex = E.fbm(X * 0.01 + CX * 0.01, Y * 0.03, 4, 71)
    patch = E.fbm(X * 0.0025 + CX * 0.0025, Y * 0.008, 3, 77)
    gcol = mix(E.C(P["g1"]), E.C(P["g2"]), gd[None] ** 0.7) * (0.95 + 0.10 * gtex)[None]
    gcol = mix(gcol, gcol * E.C((1.12, 0.92, 0.62)), sm(0.45, 0.7, patch)[None])
    gcol = gcol * (amb * 0.6 + 0.4 * P["amb"]) + E.C(P["sun"]) * ((1 - gd) * 0.10 * P["glow"])[None]
    shore_w = shore + 5 * (E.fbm1((X - 720) / Z * 0.01 + CX * 0.9 * 0.01, 3, 73) - 0.5) * 2
    gm = sm(shore_w - 1, shore_w + 1.5, Y)
    img = mix(img, gcol, gm[None])
    foam = torch.exp(-((Y - shore_w + 1.5) / 1.6) ** 2) * 0.12 * (0.4 + P["amb"])
    img = img + E.C(P["hor"]) * foam[None]
    # ---------------- 草带 + 物件（画家算法）
    yg = shore + (DH + 70 - shore) * 0.40
    strips = [0.07, 0.17, 0.29, 0.41, 0.56, 0.73, 0.93]
    info = dict(hor=hor, shore=shore, yg=yg, Z=Z, CX=CX)
    def OX(wx): return 720 + (wx - CX) * Z
    drawn_objects = False
    for kk, f in enumerate(strips):
        if f > 0.40 and not drawn_objects:
            img = _objects(img, t, P, CX, Z, yg, fire, show_hero, info, hero, catst); drawn_objects = True
        img = _grass_strip(img, t, P, CX, Z, shore, f, kk)
    if not drawn_objects: img = _objects(img, t, P, CX, Z, yg, fire, show_hero, info, hero, catst)
    # ---------------- 体积光（向太阳的径向模糊）
    if P["rays"] > 0.02 and P["glow"] > 0.05:
        img = _godrays(img, occ, dsun, sxs, P["suny"] + (hor - HOR), P, hor)
    # ---------------- 粒子：花粉 / 萤火 / 火星
    img = _particles(img, t, P, CX, Z, yg, info)
    # ---------------- 地面薄雾（黎明）
    if P["mist"] > 0.3:
        mn = E.fbm(X * 0.003 - t * 0.05, (Y - shore) * 0.02, 4, 63)
        mm = torch.exp(-((Y - shore - 10) / 60) ** 2) * (0.3 + 0.7 * mn) * (P["mist"] - 0.3) * 0.6
        img = mix(img, E.C(P["hor"]) * 0.9 + 0.08, mm.clamp(0, 0.8)[None])
    return img, info

def _grass_strip(img, t, P, CX, Z, shore, f, kk):
    X, Y = E.X, E.Y
    base = shore + (DH + 70 - shore) * f
    r = 0.62 + 0.85 * f; zk = Z * (0.85 + 0.35 * f)
    sp = 3.0 + 9.0 * f; hb = 10 + 78 * f; bw = 0.7 + 2.3 * f
    r0 = int(max(0, (base - hb * 1.25) * E.PX)); 
    if r0 >= E.H: return img
    Xc = X[r0:]; Yc = Y[r0:]
    u0 = (Xc - 720) / zk + CX * r + kk * 137.0
    yy = ((base - Yc) / hb)
    lightk = E.C(P["ambc"]) * P["amb"]
    dark = mix(E.C(P["g2"]), E.C(P["g1"]), 0.25 + 0.45 * (1 - f)) * (lightk * 0.75 + 0.25 * P["amb"])
    acc_a = torch.zeros_like(Xc); acc_tip = torch.zeros_like(Xc)
    w_ = wind(u0, t) * (3 + 15 * f)
    for j in range(2):
        u = u0 + j * sp * 0.5
        cell = torch.floor(u / sp); fr = u / sp - cell; ci = cell.long()
        hr = E.ihash(ci, ci * 0 + kk * 7 + j, 101)
        bh = hb * (0.45 + 0.55 * hr)
        q = ((base - Yc) / bh)
        jit = (E.ihash(ci, ci * 0 + kk * 7 + j, 102) - 0.5) * 0.7
        lean = (E.ihash(ci, ci * 0 + kk * 7 + j, 103) - 0.5) * 0.5
        qq = q.clamp(0, 1)
        off = (fr - 0.5 - jit * 0.5) * sp - w_ * qq ** 2 - lean * bh * 0.45 * qq
        wid = bw * (1 - qq) + 0.25
        a = sm(wid + 1.1 / E.PX, wid - 0.45 / E.PX, off.abs()) * ((q >= 0) & (q <= 1)).float() * (0.35 + 0.65 * sm(0.05, 0.35, f))
        acc_a = torch.maximum(acc_a, a)
        acc_tip = torch.maximum(acc_tip, a * qq ** 2.5)
        if 2 <= kk <= 5 and j == 0:   # 小野花
            fl = (E.ihash(ci, ci * 0 + kk, 104) < 0.05).float()
            fx = (0.5 + jit * 0.5) * sp + w_ * 1.0 + lean * bh * 0.45
            dx = (fr * sp - fx); dy = (Yc - (base - bh))
            fm = fl * sm(2.4 + 1.6 * f, 1.0 + f, torch.sqrt(dx * dx + dy * dy))
            fc = mix(E.C((1.0, 0.95, 0.85)), E.C((1.0, 0.82, 0.35)), (E.ihash(ci, ci * 0, 105) > 0.5).float()[None])
            acc_tip = acc_tip
            img_fl = fc * (lightk * 0.8 + 0.2 * P["amb"] + P["night"] * 0.05)
            img[:, r0:] = mix(img[:, r0:], img_fl, (fm * 0.9)[None])
    below = sm(base - 0.5, base + 1.5, Yc)
    tipc = mix(E.C(P["g1"]) * (lightk * 0.9 + 0.1), E.C(P["sun"]) * 0.9, 0.55 * P["glow"])
    gc = dark + (tipc - dark) * (acc_tip * (0.6 + 0.4 * P["glow"]))[None]
    gc = gc + E.C(P["sun"]) * (acc_tip * 0.35 * P["glow"] * P["rays"])[None]
    # 草带底色（让前排盖住后排）
    tex = E.fbm(Xc * 0.02 + kk, Yc * 0.05, 3, 107 + kk)
    patch = E.fbm(Xc * 0.0022 + CX * r * 0.0022, Yc * 0.006 + kk, 3, 79)
    bcol = dark * (0.93 + 0.12 * tex)[None]
    pw = sm(0.42, 0.72, patch)[None]
    bcol = mix(bcol, bcol * E.C((1.15, 0.95, 0.62)), pw)
    gc = mix(gc, gc * E.C((1.1, 0.97, 0.7)), pw)
    img[:, r0:] = mix(img[:, r0:], bcol, below[None])
    img[:, r0:] = mix(img[:, r0:], gc, (acc_a * (1 - below))[None])
    return img

def _godrays(img, occ, dsun, sx, sy, P, hor):
    X, Y = E.X, E.Y
    src = torch.exp(-dsun / 110) * (1 - occ) * (Y < hor).float()
    s = E.F.avg_pool2d(src[None, None], 4)[0, 0]
    hh, ww = s.shape
    gy, gx = torch.meshgrid(torch.linspace(-1, 1, hh, device=E.DEV), torch.linspace(-1, 1, ww, device=E.DEV), indexing="ij")
    cx = sx / DW * 2 - 1; cy = sy / DH * 2 - 1
    acc = torch.zeros_like(s); n = 22; dec = 1.0
    for i in range(n):
        k = i / n * 0.85
        g = torch.stack([gx + (cx - gx) * k, gy + (cy - gy) * k], -1)[None]
        acc += E.F.grid_sample(s[None, None], g, align_corners=True)[0, 0] * dec; dec *= 0.94
    acc = acc / n
    acc = E.F.interpolate(acc[None, None], size=img.shape[1:], mode="bilinear", align_corners=False)[0, 0]
    return img + E.C(P["sun"]) * (acc * P["rays"] * 1.1 * P["glow"])[None]

def _tent(m_front, m_side, m_door, m_lines, tx, yg, Z, CX, u=1.0, fold=0.0):
    def O(x, y): return (720 + (x - CX) * Z, yg + (y - 0) * Z)
    sq = 1 - fold
    A = (tx - 40, -146 * sq); B = (tx + 82, -136 * sq)
    FL = (tx - 126, 0); FR = (tx + 40, 5); BR = (tx + 152, -6)
    pole_u = cl(u / 0.45); fab = sm(0.35, 0.95, u)
    if pole_u > 0:
        for p0, p1 in ((FL, A), (FR, A), (A, B), (B, BR)):
            pe = (p0[0] + (p1[0] - p0[0]) * pole_u, p0[1] + (p1[1] - p0[1]) * pole_u)
            m_lines.line([O(*p0), O(*pe)], 2.2 * Z, int(255 * (1 - fab * 0.6)))
    if fab > 0:
        v = int(255 * fab)
        m_front.poly([O(*A), O(*FL), O(*FR)], v)
        m_side.poly([O(*A), O(*FR), O(*BR), O(*B)], v)
        m_door.poly([O(tx - 38, -128 * sq), O(tx - 70, 0), O(tx - 6, 2)], v)
        for gp, peg in ((A, (tx - 190, 6)), (B, (tx + 215, -2))):
            m_lines.line([O(*gp), O(*peg)], 1.0 * Z, int(110 * fab))
    return O(*A), O(*B)

def _objects(img, t, P, CX, Z, yg, fire, show_hero, info, hero=None, catst=None):
    X, Y = E.X, E.Y
    amb = E.C(P["ambc"]) * P["amb"]
    def OX(wx): return 720 + (wx - CX) * Z
    fire_l = fire if fire is not None else (ease((t - 66.0) / 1.6) * (1 - ease((t - 86.0) / 2.0)))
    flick = 0.82 + 0.18 * math.sin(t * 13.1) * math.sin(t * 7.3 + 1) + 0.08 * math.sin(t * 23.7)
    tentglow = cl((P["night"] - 0.2) / 0.6) * (1 - ease((t - 86.5) / 1.5) if fire is None else 1.0)
    # ---- 帐篷
    tent_u = ease((t - 46.2) / 3.0) if fire is None else 1.0
    fold = ease((t - 88.6) / 1.4) if fire is None else 0.0
    if tent_u > 0 and fold < 1:
        mf, ms, md, ml = Mask(), Mask(), Mask(), Mask()
        _tent(mf, ms, md, ml, TENT_X, yg, Z, CX, tent_u, fold)
        af, as_, ad, al = mf.t(), ms.t(), md.t(), ml.t()
        sidec = E.C((0.93, 0.55, 0.27)) * (amb * 0.9 + 0.08) + E.C(P["sun"]) * 0.18 * P["glow"]
        frontc = E.C((0.60, 0.31, 0.17)) * (amb * 0.75 + 0.06)
        doorc = E.C((0.12, 0.07, 0.05)) * (amb + 0.2)
        glowc = E.C((1.0, 0.62, 0.30))
        sidec = sidec + glowc * 0.55 * tentglow; frontc = frontc + glowc * 0.75 * tentglow
        doorc = doorc + E.C((1.0, 0.80, 0.50)) * 1.25 * tentglow
        img = over(img, E.C((0.35, 0.33, 0.30)) * (amb + 0.1), al * 0.9)
        img = over(img, sidec, as_); img = over(img, frontc, af); img = over(img, doorc, ad * 0.95)
        rim = rim_light(as_ + af, 2.5, -2.0)
        img = add(img, E.C(P["sun"]), rim * 0.7 * P["glow"])
        if tentglow > 0.01:
            pool = torch.exp(-(((X - OX(TENT_X - 40)) / (260 * Z)) ** 2 + ((Y - yg - 6) / (40 * Z)) ** 2)) * (Y > info["shore"]).float()
            img = add(img, glowc, pool * 0.28 * tentglow)
        # 帐篷上的暖色串灯（夜里亮起）
        lights_on = cl((P["night"] - 0.25) / 0.5) * (1.0 if fire is not None else (1 - ease((t - 88.0) / 1.5)))
        if lights_on > 0.01 and tent_u > 0.95 and fold < 0.05:
            A_ = (TENT_X - 40, -146); peg = (TENT_X - 190, 6); B_ = (TENT_X + 82, -136)
            pts = []
            for (p0, p1, n_) in ((peg, A_, 9), (A_, B_, 6)):
                for i in range(n_):
                    u = (i + 0.5) / n_
                    pts.append((720 + (lerp(p0[0], p1[0], u) - CX) * Z, yg + (lerp(p0[1], p1[1], u) + 10 * math.sin(u * math.pi)) * Z))
            pts = np.array(pts); nb = len(pts)
            tw = 0.85 + 0.15 * np.sin(t * 1.3 + np.arange(nb) * 1.7)
            img = img + glow_points(pts, tw * 1.6 * lights_on * min(1.5, Z), np.broadcast_to(np.array([1.0, 0.78, 0.45]), (nb, 3)), sigma=3.0, core=0.5)
    # ---- 马克杯（主人身边的一杯热可可）
    mug = cl((t - 50.2) / 0.4) if fire is None else 1.0
    if mug > 0.01 and t < 88.6 or fire is not None:
        mx = OX(1838); mm_ = Mask()
        mm_.rrect(mx - 6 * Z, yg - 14 * Z, mx + 6 * Z, yg + 1 * Z, 2 * Z); mm_.circle((mx + 7 * Z, yg - 7 * Z), 4 * Z, w=1.6 * Z)
        img = over(img, E.C((0.92, 0.88, 0.80)) * (amb * 0.8 + 0.15), mm_.t() * mug)
    # ---- 炉子 + 水壶（烧一壶水）
    stove = win(t, 47.6, 62.0, 0.6, 1.5) if fire is None else 0.0
    if stove > 0.01:
        ms = Mask(); x0 = OX(STOVE_X)
        ms.rect(x0 - 10 * Z, yg - 10 * Z, x0 + 10 * Z, yg)
        ms.rrect(x0 - 14 * Z, yg - 30 * Z, x0 + 14 * Z, yg - 12 * Z, 6 * Z)
        ms.line([(x0 + 10 * Z, yg - 26 * Z), (x0 + 20 * Z, yg - 34 * Z)], 2.5 * Z)
        img = over(img, E.C((0.30, 0.30, 0.32)) * (amb + 0.1), ms.t() * stove)
        bf = Mask().ellipse((x0, yg - 11 * Z), (8 * Z, 2.5 * Z)).t(1.5)
        img = add(img, E.C((0.35, 0.55, 1.0)), bf * 1.4 * stove)
    # ---- 篝火
    if fire_l > 0.01:
        fx = OX(FIRE_X); fy = yg + 4 * Z
        ml = Mask()
        ml.line([(fx - 30 * Z, fy + 2 * Z), (fx + 26 * Z, fy - 8 * Z)], 7 * Z)
        ml.line([(fx - 24 * Z, fy - 8 * Z), (fx + 30 * Z, fy + 3 * Z)], 7 * Z)
        for a in range(9):
            ang = a / 9 * 6.283; ml.circle((fx + math.cos(ang) * 38 * Z, fy + math.sin(ang) * 9 * Z + 2), 5.5 * Z)
        img = over(img, E.C((0.10, 0.07, 0.06)), ml.t())
        # 地面光池
        pool = torch.exp(-(((X - fx) / (330 * Z)) ** 2 + ((Y - fy) / (70 * Z)) ** 2)) * (Y > info["shore"]).float()
        glow = torch.exp(-(((X - fx) ** 2 + (Y - fy + 40 * Z) ** 2) / (240 * Z) ** 2))
        img = img * (1 + (pool * 1.6 + glow * 0.5) * fire_l * flick)[None] + E.C((1.0, 0.50, 0.20)) * ((pool * 0.30 + glow * 0.07) * fire_l * flick)[None]
        # 火焰（局部着色）
        x0, x1 = int(max(0, (fx - 60 * Z) * E.PX)), int(min(E.W, (fx + 60 * Z) * E.PX))
        y0, y1 = int(max(0, (fy - 140 * Z) * E.PX)), int(min(E.H, (fy + 6 * Z) * E.PX))
        if x1 > x0 and y1 > y0:
            Xc = X[y0:y1, x0:x1]; Yc = Y[y0:y1, x0:x1]
            lx = (Xc - fx) / Z; ly = (fy - Yc) / Z
            fh = (78 + 22 * math.sin(t * 5.3) + 12 * math.sin(t * 11.7)) * (0.4 + 0.6 * fire_l)
            q = (ly / fh).clamp(0, 1.2)
            warp = (E.vnoise(lx * 0.06, ly * 0.05 - t * 4.0, 201) - 0.5) * 30 * q + (E.vnoise(lx * 0.12, ly * 0.1 - t * 7, 202) - 0.5) * 10 * q
            wdt = 26 * (1 - q).clamp(0, 1) ** 0.65 * (1 + 0.6 * q)
            fl = sm(wdt, wdt * 0.15, (lx + warp).abs()) * (ly > -4).float() * (1 - sm(0.75, 1.1, q))
            tongue = sm(0.5, 0.85, E.vnoise(lx * 0.08, ly * 0.04 - t * 5.5, 203)) * sm(1.3, 0.6, q) * sm(30, 6, (lx + warp).abs())
            I = (fl + tongue * 0.6 * (q > 0.4).float()).clamp(0, 1)
            core = sm(0.55, 0.95, I) * sm(0.7, 0.1, q)
            col = mix(E.C((0.95, 0.25, 0.05)), E.C((1.0, 0.62, 0.15)), sm(0.2, 0.7, I)[None])
            col = mix(col, E.C((1.0, 0.95, 0.75)), core[None])
            img[:, y0:y1, x0:x1] = img[:, y0:y1, x0:x1] + col * (I * 1.8 * fire_l)[None]
        info["fire"] = (fx, fy, fire_l * flick)
    # ---- 小猫（画在主角前一层：它在主人身后/身边）
    if show_hero:
        cx_, cp, cph, cfl, clook = cat_state(t) if catst is None else catst
        mc = Mask(); kc = cat(mc, OX(cx_), yg + 3 * Z, 33 * Z, cp, cph, cfl, t, clook)
        ac = mc.t()
        firek = info["fire"][2] if "fire" in info else 0.0
        img = paint_cat(img, ac, amb, key=P["sun"], key_k=0.4 + 0.6 * P["glow"], rim_col=P["sun"], rim_k=0.9 * P["glow"],
                        extra=0.12 * firek, eye=(kc["head"][0], kc["head"][1], kc["r"], cfl), eye_k=cl((P["night"] - 0.3) / 0.5) * 0.8)
        if firek > 0: img = add(img, E.C((1.0, 0.55, 0.22)), rim_light(ac, 2.0 * (1 if info['fire'][0] > OX(cx_) else -1), 0.5) * 1.2 * firek)
    # ---- 主角
    if show_hero:
        hx, pname, ph, pack = hero_state(t) if hero is None else hero
        mh = Mask()
        pp = pose(pname, ph)
        k = figure(mh, OX(hx), yg + 3 * Z, 150 * Z, pp, flip=False, pack=pack)
        a = mh.t()
        body = E.C((0.075, 0.07, 0.075)) * (amb * 0.7 + 0.25) + E.C((0.03, 0.03, 0.04))
        img = over(img, body, a)
        rim = rim_light(a, 3.0, -1.5)
        img = add(img, E.C(P["sun"]), rim * 1.1 * P["glow"] + rim * 0.15)
        if "fire" in info:
            fx, fy, fl = info["fire"]
            rimf = rim_light(a, 2.5 * (1 if fx > OX(hx) else -1), 0.5)
            img = add(img, E.C((1.0, 0.55, 0.22)), rimf * 1.4 * fl)
            img = add(img, E.C((1.0, 0.5, 0.2)), a * 0.10 * fl)
        info["hero"] = k
    return img

_PG = None
def _particles(img, t, P, CX, Z, yg, info):
    global _PG
    if _PG is None:
        g = np.random.default_rng(5)
        _PG = dict(px=g.uniform(-200, 3400, 520), py=g.uniform(0, 1, 520), pv=g.uniform(0.3, 1, 520), pp=g.uniform(0, 6.28, 520),
                   fx=g.uniform(1400, 2500, 90), fy=g.uniform(0, 1, 90), ff=g.uniform(0.3, 1.0, 90), fp=g.uniform(0, 6.28, 90),
                   sp=g.uniform(0, 1, 70), sr=g.uniform(0.35, 0.8, 70), sd=g.uniform(-1, 1, 70),
                   st=g.uniform(0, 1, 40), sx=g.uniform(-14, 14, 40))
    G = _PG; layers = []
    # 花粉（金色时刻，被风吹向右）
    pol = P["glow"] * (1 - P["night"]) * 0.9
    if pol > 0.02:
        x = (G["px"] + t * (40 + 60 * gust(t)) * G["pv"] + 30 * np.sin(t * 0.7 + G["pp"])) % 3600 - 200
        sx = 720 + (x - CX) * Z
        sx = (sx % 1700) - 130
        sy = info["hor"] - 120 + G["py"] * 600 + 20 * np.sin(t * 0.9 + G["pp"] * 3)
        dsun = np.hypot(sx - P["sunx"], sy - P["suny"])
        b = (0.25 + 1.4 * np.exp(-dsun / 420)) * pol * (0.6 + 0.4 * np.sin(t * 0.8 + G["pp"])) * 1.5
        b = b * (np.arange(len(b)) < 170)          # 花粉减量、放大、慢闪：柔和光斑而不是噪点
        layers.append((np.stack([sx, sy], 1), b, np.broadcast_to(np.array([1.0, 0.85, 0.55]), (len(sx), 3)), 2.6))
    # 萤火虫
    ff = cl((P["night"] - 0.3) / 0.5)
    if ff > 0.02:
        x = G["fx"] + 50 * np.sin(t * 0.37 * (1 + G["ff"]) + G["fp"])
        y = yg - 18 - G["fy"] * 210 + 22 * np.sin(t * 0.5 * (1 + G["ff"]) + G["fp"] * 2)
        sx = 720 + (x - CX) * Z; sy = info["hor"] - HOR + (y - info["hor"] + HOR) * 1.0 + 0 * Z
        blink = np.clip(np.sin(t * (1.2 + G["ff"]) + G["fp"] * 5) * 1.6 - 0.2, 0, 1) ** 2
        layers.append((np.stack([sx, sy], 1), blink * 2.6 * ff, np.broadcast_to(np.array([0.78, 1.0, 0.45]), (len(sx), 3)), 3.2))
    # 火星
    if "fire" in info and info["fire"][2] > 0.02:
        fx, fy, fl = info["fire"]
        age = (t * G["sr"] + G["sp"]) % 1.0
        x = fx + (G["sd"] * 25 + np.sin(age * 9 + G["sp"] * 6) * 18 + age * 40) * Z
        y = fy - (30 + age * 260) * Z
        b = (1 - age) ** 2 * 3.0 * fl
        layers.append((np.stack([x, y], 1), b, np.broadcast_to(np.array([1.0, 0.62, 0.25]), (len(x), 3)), 1.3))
    # 壶口蒸汽
    st = win(t, 50.0, 60.5, 1.0, 1.5)
    if st > 0.02:
        age = (t * 0.45 + G["st"]) % 1.0
        x0 = 720 + (STOVE_X + 22 - CX) * Z
        x = x0 + (G["sx"] + age * 30 + np.sin(age * 7 + G["st"] * 9) * 10) * Z
        y = yg - (36 + age * 110) * Z
        b = np.sin(age * np.pi) * 0.5 * st
        layers.append((np.stack([x, y], 1), b, np.broadcast_to(np.array([0.95, 0.92, 0.9]), (len(x), 3)), 5.0))
    for Pp, b, c, sg in layers:
        img = img + glow_points(Pp, b, c, sigma=sg, core=0.45)
    return img

def _wind_lines(img, t, strength, col=(1.0, 0.95, 0.85)):
    """2D 风线：动画式的风的笔触"""
    if strength < 0.01: return img
    m = Mask()
    for i in range(7):
        ph = hash1(i, 3) * 6.28; y0 = 180 + hash1(i, 4) * 560; spd = 900 + hash1(i, 5) * 500
        per = 2.6 + hash1(i, 6) * 1.6
        u = ((t + hash1(i, 7) * per) % per) / per
        head = -400 + u * (DW + 900); L = 260 + 180 * hash1(i, 8)
        xs = np.linspace(head - L, head, 24)
        ys = y0 + 26 * np.sin(xs * 0.006 + ph + t * 0.8) + 40 * np.sin(xs * 0.0021 + ph)
        for s0 in range(3):
            a = int(255 * (0.25 + 0.3 * s0) * math.sin(u * math.pi))
            seg = slice(s0 * 8, s0 * 8 + 9)
            m.line(np.stack([xs[seg], ys[seg]], 1), 1.6 + s0 * 0.5, a)
    return add(img, E.C(col), m.t(0.6) * strength * 0.55)

def _birds(img, t, P):
    a = win(t, 37.0, 49.0, 1.0, 1.5) * (1 - P["night"])
    if a < 0.01: return img
    m = Mask()
    for i in range(5):
        x = -100 + (t - 37) * (70 + 15 * i) + i * 60; y = 200 + i * 22 + 18 * math.sin(t * 0.6 + i)
        fl = math.sin(t * (7 + i) + i) * 7; s = 9 - i * 0.8
        m.line([(x - s, y - fl), (x, y), (x + s, y - fl)], 1.8)
    return over(img, E.C((0.15, 0.12, 0.14)), m.t() * a)

def _meteor(img, t, specs):
    for (t0, x0, y0, dx, dy, d) in specs:
        u = (t - t0) / d
        if not (0 <= u < 1): continue
        k = np.linspace(0, 1, 140); hu = eout(u); uu = np.clip(hu - k * 0.25, 0, 1)
        Pp = np.stack([x0 + dx * uu, y0 + dy * uu], 1)
        b = (1 - k) ** 2 * 3.5 * (1 - u) ** 0.6 * min(1.0, u * 8)
        img = img + glow_points(Pp, b, np.array([1.0, 0.96, 0.9]), sigma=1.4, core=0.7)
    return img

# ======================================================================= 城市：楼群着色器
def buildings(img, Xw, Yw, base, sw, hmin, hmax, seed, col, wcol, lit_p, warm_p, fog, fogc, win_w=9, win_h=12, wa=1.0, dxp=None):
    bu = Xw / sw; bi = torch.floor(bu).long(); bf = bu - torch.floor(bu)
    g0 = 0.04 + E.ihash(bi, bi * 0 + 1, seed) * 0.12; g1 = 0.88 + E.ihash(bi, bi * 0 + 2, seed) * 0.1
    hgt = hmin + (hmax - hmin) * E.ihash(bi, bi * 0 + 3, seed) ** 1.4
    top = base - hgt
    inside = ((bf > g0) & (bf < g1) & (Yw > top)).float()
    lx = (bf - g0) * sw; ly = Yw - top
    wx = torch.frac(lx / win_w); wy = torch.frac(ly / win_h)
    wxi = torch.floor(lx / win_w).long(); wyi = torch.floor(ly / win_h).long()
    wmask = ((wx > 0.22) & (wx < 0.78) & (wy > 0.28) & (wy < 0.78) & (ly > 10) & (lx > 3) & (lx < (g1 - g0) * sw - 3)).float()
    hh = E.ihash(bi * 97 + wxi, wyi, seed + 11)
    lit = (hh < lit_p).float(); warm = (E.ihash(bi * 97 + wxi, wyi, seed + 12) < warm_p).float()
    inten = 0.55 + 0.45 * E.ihash(bi * 97 + wxi, wyi, seed + 13)
    grad = (1 - (ly / (hgt + 1)).clamp(0, 1) * 0.35)
    c = col * grad[None]
    wc = mix(wcol, E.C((1.0, 0.78, 0.48)), warm[None])
    # 窗格在屏幕上太小时会变成闪烁的噪点：按像素尺寸平滑过渡到平均亮度
    if dxp is None:
        # 未提供步长时保留原逻辑；场景调用处已使用解析步长以消除 GPU→CPU 同步
        dxp = float((Xw[0, -1] - Xw[0, 0]).abs()) / max(1, Xw.shape[1] - 1) + 1e-6
    aa = cl((win_w / dxp - 3.0) / 3.0)
    wl = wmask * lit * inten * aa + (1 - aa) * (0.56 * 0.5 * lit_p * 0.78)
    wo = wmask * lit * aa + (1 - aa) * (0.56 * 0.5 * lit_p)
    wc = wc if aa > 0.02 else mix(wcol, E.C((1.0, 0.78, 0.48)), warm_p)
    c = c + (wc * (wl * wa)[None] - c * (wo * wa)[None] * 0.5)
    c = mix(c, fogc, fog)
    edge = sm(top + 1.5, top - 0.5, Yw) * inside   # 屋顶细高光
    out = mix(img, c, inside[None])
    return out + fogc * (edge * 0.15)[None]

def sky_grad(top, bot, y0=0, y1=DH):
    u = ((E.Y - y0) / (y1 - y0)).clamp(0, 1)
    return mix(E.C(top), E.C(bot), u[None])

def ui_card(img, x, y, text, sz, a, dot=True, w=None, light=True):
    if a < 0.01: return img
    tw = len(text) * sz * 0.92 + sz * (2.2 if dot else 1.2) if w is None else w
    m = Mask().rrect(x - tw / 2, y - sz * 0.95, x + tw / 2, y + sz * 0.95, sz * 0.5)
    img = over(img, E.C((0.88, 0.91, 0.95)) if light else E.C((0.16, 0.19, 0.24)), m.t() * a * 0.9)
    tx = x + (sz * 0.55 if dot else 0)
    if dot:
        d = Mask().circle((x - tw / 2 + sz * 0.9, y), sz * 0.32).t()
        img = over(img, E.C((0.93, 0.26, 0.24)), d * a)
    return draw_text(img, text, tx, y, sz, E.C((0.12, 0.14, 0.18)) if light else E.C((0.85, 0.88, 0.92)), a)

# ---------------------------------------------------------------- A：清晨城市航拍，推进到列车车窗
def train_x(t): return 732 - 1500 * (1 - eout(t / 4.9)) ** 1.0   # 第 k 节车窗最后停在屏幕中央
TRAIN_Y = 752; WIN_REL = (0, -6, 44, 26)  # 目标车窗相对位置（中心 x 偏移、y 偏移、宽、高）
def a_cam(t):
    if t < 5.6: return 1.0 + 0.32 * ease(t / 5.6), 720 + 0, 520 + 230 * ease(t / 5.6)
    z = 1.32 * (48.0 / 1.32) ** ein((t - 5.6) / 1.6) if t < 7.2 else 48.0
    return z, 720, TRAIN_Y + WIN_REL[1]

def scene_a(t, cam=None):
    z, cx, cy = cam if cam is not None else a_cam(t)
    Xw = (E.X - 720) / z + cx; Yw = (E.Y - 540) / z + cy
    img = sky_grad((0.30, 0.36, 0.45), (0.58, 0.62, 0.67), 0, 700)
    img = img + E.C((0.9, 0.75, 0.6)) * (torch.exp(-((Xw - 1100) ** 2 + (Yw - 470) ** 2) / 400 ** 2) * 0.12)[None]
    fogc = E.C((0.50, 0.56, 0.63))
    rows = [(600, 26, 160, 330, 21, 0.70, 3.0), (640, 34, 120, 300, 22, 0.55, 1.8), (690, 44, 90, 260, 23, 0.42, 1.0),
            (740, 58, 80, 250, 24, 0.28, 0.4), (800, 76, 60, 230, 25, 0.16, 0.0), (900, 96, 70, 260, 26, 0.06, -0.6)]
    for base, sw, hmin, hmax, sd, fog, par in rows:
        Xp = Xw + par * t * 6 + sd * 50
        img = buildings(img, Xp, Yw, base, sw, hmin, hmax, sd, E.C((0.16, 0.19, 0.24)) * (1 + fog * 0.3), E.C((0.70, 0.80, 0.90)),
                        0.22 + 0.1 * (1 - fog), 0.04, fog, fogc, 6 + 6 * (1 - fog), 8 + 7 * (1 - fog), 0.8, dxp=1.0 / (z * E.PX) + 1e-6)
    # 高架轨道 + 列车
    m = Mask(); ty = TRAIN_Y
    def S(x, y): return ((x - cx) * z + 720, (y - cy) * z + 540)
    m.rect(*S(-200, ty + 32), *S(1640, ty + 46))
    for px in range(-100, 1700, 230): m.rect(*S(px, ty + 46), *S(px + 26, 1100))
    img = over(img, E.C((0.11, 0.13, 0.16)), m.t())
    tx = train_x(t); car = 210; ncar = 8
    head = tx + 105 + car * 2
    mt = Mask(); mw = Mask(); mh = Mask()
    for k in range(ncar):
        x1 = head - k * car; x0 = x1 - car + 8
        r = 18 if k == 0 else 4
        mt.rrect(*S(x0, ty - 26), *S(x1, ty + 30), r * z)
        for j in range(4):
            wx = x0 + 16 + j * 50
            mw.rrect(*S(wx, ty - 19), *S(wx + 38, ty + 7), 3 * z)
            for q in range(3):
                hx = wx + 6 + q * 13 + 2 * math.sin(t * 2 + k + j + q)
                if hash1(k * 13 + j * 3 + q, 7) < 0.85:
                    p_ = S(hx, ty - 6 + 4 * hash1(k + j + q, 9)); bust(mh, p_[0], p_[1], 4.6 * z)
    img = over(img, E.C((0.62, 0.66, 0.72)), mt.t())
    img = over(img, E.C((0.62, 0.72, 0.80)), mw.t())
    img = over(img, E.C((0.16, 0.19, 0.24)), mh.t() * mw.t())
    stripe = Mask().rect(*S(head - ncar * car, ty + 14), *S(head, ty + 18)).t()
    img = over(img, E.C((0.25, 0.55, 0.75)), stripe * mt.t())
    # 街道车流光点
    g = np.random.default_rng(3); n = 260
    lane = g.integers(0, 3, n); sp = g.uniform(60, 140, n) * np.where(lane == 1, -1, 1)
    x = (g.uniform(0, 2000, n) + sp * t) % 2000 - 280; y = 1000 + lane * 18 + g.uniform(-3, 3, n)
    pts = np.stack([(x - cx) * z + 720, (y - cy) * z + 540], 1)
    colp = np.where((sp > 0)[:, None], np.array([1.0, 0.92, 0.8]), np.array([1.0, 0.25, 0.2]))
    img = img + glow_points(pts, np.full(n, 1.2), colp, sigma=2.2 * min(z, 4), core=0.5)
    # 雾
    mist = E.fbm(Xw * 0.003 + t * 0.02, Yw * 0.01, 3, 9) * torch.exp(-((Yw - 820) / 200) ** 2) * 0.35
    img = mix(img, fogc, mist[None])
    wr = (S(701, ty - 19), S(739, ty + 7))
    return img, wr

# ---------------------------------------------------------------- B / R：地铁车厢（warm=1 为回城后的清晨版本）
def scene_b(t, warm=0.0, cam=None, portal=None, beam=1.0):
    z, cx, cy = cam if cam is not None else (1.0 + 0.04 * math.sin(t * 0.3), 700 + 9 * (t % 100), 540)
    if cam is None: cx = 700 + 10 * (t - 7.2) if warm == 0 else 760 + 6 * (t - 93)
    sway = 7 * math.sin(t * 1.15) + 4 * math.sin(t * 2.7)
    Xw = (E.X - 720) / z + cx; Yw = (E.Y - 540) / z + cy
    def S(x, y): return ((x - cx) * z + 720, (y - cy) * z + 540)
    wall = mix(E.C((0.15, 0.19, 0.25)), E.C((0.36, 0.33, 0.30)), warm)
    img = wall * (0.75 + 0.35 * sm(900, 150, Yw))[None]
    # 车窗内容
    wins = [(160, 560), (640, 1040), (1120, 1520)]
    mw = Mask()
    for a, b in wins: mw.rrect(*S(a, 250), *S(b, 520), 22 * z)
    wm = mw.t()
    if warm < 0.5:
        streak = torch.zeros_like(Xw)
        for k, yy in enumerate((300, 360, 455)):
            u = torch.frac((Xw * 0.35 + t * 1100 + k * 300) / (420 + 80 * k))
            streak = streak + sm(0.0, 0.02, u) * sm(0.35, 0.05, u) * torch.exp(-((Yw - yy) / 5.0) ** 2)
        tun = E.C((0.02, 0.03, 0.05)) + E.C((0.7, 0.85, 1.0)) * streak[None] * 0.9
        flash = 0.25 * max(0.0, math.sin(t * 0.9)) ** 30
        tun = tun + E.C((0.6, 0.7, 0.8)) * flash
    else:
        wy = Yw
        sky = mix(E.C((1.0, 0.86, 0.66)), E.C((0.62, 0.74, 0.88)), sm(520, 250, wy)[None])
        tun = sky
        tun = buildings(tun, Xw + t * 260, Yw, 560, 70, 60, 240, 41, E.C((0.36, 0.30, 0.30)), E.C((1.0, 0.85, 0.6)), 0.1, 0.5, 0.35, E.C((1.0, 0.82, 0.62)), 7, 10, 0.5, dxp=1.0 / (z * E.PX) + 1e-6)
        tun = buildings(tun, Xw + t * 520, Yw, 600, 110, 40, 160, 42, E.C((0.22, 0.18, 0.18)), E.C((1.0, 0.85, 0.6)), 0.05, 0.5, 0.1, E.C((1.0, 0.82, 0.62)), 10, 14, 0.3, dxp=1.0 / (z * E.PX) + 1e-6)
    if portal is not None:
        tun = portal
    img = mix(img, tun, wm[None])
    refl = (1 - warm) * 0.10
    img = img + E.C((0.6, 0.7, 0.8)) * (wm * refl * sm(520, 250, Yw))[None]
    # 广告条与灯带
    ml = Mask().rect(*S(-200, 112), *S(1800, 124)).t()
    img = add(img, mix(E.C((0.85, 0.93, 1.0)), E.C((1.0, 0.92, 0.8)), warm), ml * 1.2)
    img = img + mix(E.C((0.5, 0.6, 0.75)), E.C((0.7, 0.6, 0.5)), warm) * (torch.exp(-((Yw - 118) / 70) ** 2) * 0.25)[None]
    ad = Mask()
    for a, b in wins: ad.rect(*S(a + 40, 160), *S(b - 40, 225))
    img = over(img, mix(E.C((0.30, 0.36, 0.44)), E.C((0.55, 0.48, 0.40)), warm), ad.t() * 0.9)
    for (a, b), txt in zip(wins, ["早高峰 · 请勿倚靠车门", "下一站 · 公司", "前方到站 · 周一"]):
        p = S((a + b) / 2, 192)
        img = draw_text(img, txt, p[0], p[1], 22 * z, mix(E.C((0.80, 0.86, 0.92)), E.C((0.98, 0.92, 0.82)), warm), 0.85)
    # 扶手与吊环、立柱
    m = Mask()
    m.line([S(-200, 236), S(1800, 236)], 5 * z)
    hp = []
    for i in range(-2, 22):
        hx = i * 90 + 30; ang = math.radians(6 * math.sin(t * 1.3 + i * 0.4) + sway * 0.8)
        L = 70; ex, ey = hx + L * math.sin(ang), 236 + L * math.cos(ang)
        m.line([S(hx, 236), S(ex, ey)], 3 * z)
        m.circle(S(ex, ey + 9), 10 * z, w=3 * z)
    for px in (330, 1180):
        m.line([S(px, 100), S(px, 1200)], 9 * z)
    img = over(img, mix(E.C((0.55, 0.60, 0.66)), E.C((0.85, 0.75, 0.62)), warm), m.t())
    # 人群：后排 / 中排（主角）/ 前排
    mb, mm, mf = Mask(), Mask(), Mask()
    for i in range(16):
        x = i * 112 - 80 + 30 * hash1(i, 1); s = 64 + 16 * hash1(i, 2)
        bust(mb, *S(x, 430 + 30 * hash1(i, 3)), s * z, (sway + 4 * math.sin(t + i)) * z,
             arm=S(x + 60, 245) if hash1(i, 4) < 0.35 else None)
    HERO_X = 760
    for i in range(12):
        x = i * 170 - 120 + 40 * hash1(i, 5)
        if abs(x - HERO_X) < 150: continue
        s = 92 + 18 * hash1(i, 6)
        bust(mm, *S(x, 520 + 25 * hash1(i, 7)), s * z, (sway * 1.2 + 5 * math.sin(t * 0.8 + i)) * z,
             arm=S(x + 90, 245) if hash1(i, 8) < 0.4 else None)
    mhero = Mask()
    hlean = sway * 1.2
    hy = 520 + (0 if warm < 0.5 else -18)
    bust(mhero, *S(HERO_X, hy), 100 * z, hlean * z)
    for i in range(6):
        x = [-60, 260, 520, 1020, 1280, 1560][i]
        bust(mf, *S(x, 700), 190 * z, (sway * 1.6) * z)
    backc = mix(E.C((0.20, 0.23, 0.28)), E.C((0.40, 0.33, 0.28)), warm)
    img = over(img, backc, mb.t(1.2))
    midc = mix(E.C((0.085, 0.10, 0.13)), E.C((0.22, 0.17, 0.14)), warm)
    img = over(img, midc, mm.t())
    ha = mhero.t()
    heroc = mix(E.C((0.10, 0.12, 0.15)), E.C((0.26, 0.19, 0.15)), warm)
    img = over(img, heroc, ha)
    if warm < 0.5:
        # 手机屏幕冷光照脸
        hp = S(HERO_X + hlean + 10, hy + 70)
        face = torch.exp(-(((E.X - hp[0]) / (60 * z)) ** 2 + ((E.Y - hp[1] + 40 * z) / (70 * z)) ** 2))
        img = img + E.C((0.35, 0.55, 0.85)) * (face * ha * 0.7)[None]
        ph = Mask().rrect(hp[0] - 16 * z, hp[1] - 6 * z, hp[0] + 16 * z, hp[1] + 18 * z, 4 * z).t()
        img = over(img, E.C((0.75, 0.88, 1.0)), ph)
        img = img + E.C((0.4, 0.6, 1.0)) * (torch.exp(-((E.X - hp[0]) ** 2 + (E.Y - hp[1]) ** 2) / (60 * z) ** 2) * 0.35)[None]
        img = add(img, E.C((0.70, 0.82, 1.0)), rim_light(ha, -2.5, -3.0) * 0.85)
    else:
        # 晨光：窗外斜射的光柱
        beams = Mask()
        for a, b in wins:
            beams.poly([S(a + 60, 250), S(b - 40, 250), S(b + 260, 1100), S(a + 300, 1100)])
        bm = beams.t(18)
        img = img + E.C((1.0, 0.80, 0.55)) * (bm * 0.22 * beam)[None]
        rim = rim_light(ha, -3, -2)
        img = add(img, E.C((1.0, 0.82, 0.58)), rim * 0.9)
        # 背包带上别着的一小朵野花
        fp = S(HERO_X + hlean + 50, hy + 120)
        fm = Mask()
        for k in range(5):
            ang = k / 5 * 6.283 + 0.3; fm.circle((fp[0] + math.cos(ang) * 5 * z, fp[1] + math.sin(ang) * 5 * z), 3.6 * z)
        img = over(img, E.C((1.0, 0.86, 0.35)), fm.t())
        img = over(img, E.C((0.95, 0.55, 0.2)), Mask().circle(fp, 2.6 * z).t())
        # 手里一杯热咖啡（周一的早晨也可以慢一点）
        cp_ = S(HERO_X + hlean - 70, hy + 150)
        cm_ = Mask().poly([(cp_[0] - 15 * z, cp_[1] - 26 * z), (cp_[0] + 15 * z, cp_[1] - 26 * z), (cp_[0] + 11 * z, cp_[1] + 22 * z), (cp_[0] - 11 * z, cp_[1] + 22 * z)]).t()
        img = over(img, E.C((0.95, 0.90, 0.82)), cm_)
        sl_ = Mask().rect(cp_[0] - 14 * z, cp_[1] - 8 * z, cp_[0] + 14 * z, cp_[1] + 6 * z).t()
        img = over(img, E.C((0.62, 0.42, 0.28)), sl_ * cm_)
        lid = Mask().rrect(cp_[0] - 17 * z, cp_[1] - 32 * z, cp_[0] + 17 * z, cp_[1] - 24 * z, 3 * z).t()
        img = over(img, E.C((0.85, 0.80, 0.74)), lid)
        ag = (t * 0.3 + hash1(np.arange(10), 95)) % 1.0
        stp = np.stack([cp_[0] + (np.sin(ag * 6 + np.arange(10)) * 6 + ag * 10) * z, cp_[1] - (34 + ag * 70) * z], 1)
        img = img + glow_points(stp, np.sin(ag * np.pi) * 0.25, np.array([1.0, 0.96, 0.9]), sigma=5 * z, core=0.3)
    img = over(img, mix(E.C((0.02, 0.025, 0.035)), E.C((0.10, 0.07, 0.06)), warm), mf.t(2.5))
    return img

# 通知卡片（2D 情绪层）
NOTES_B = [(8.4, "99+", 560, 380), (9.2, "@全体成员 今天务必", 1010, 330), (9.9, "日报未提交", 470, 300), (10.5, "会议提醒 09:30", 1050, 420),
           (11.1, "[未读] 37", 640, 250), (11.6, "您有 5 个待办", 900, 260), (12.0, "收到请回复", 420, 450), (12.4, "在吗？", 1120, 300)]
NOTES_R = [(94.6, "99+", 560, 360), (95.6, "会议提醒 09:30", 1050, 410), (96.6, "日报未提交", 460, 300)]
def notes_layer(img, t, notes, fade=1.0):
    for (t0, txt, x, y) in notes:
        a = ease((t - t0) / 0.18) * fade
        if a < 0.01: continue
        pop = 1 + 0.18 * math.exp(-(t - t0) * 9) * math.sin((t - t0) * 30)
        rise = -(t - t0) * 6
        img = ui_card(img, x, y + rise, txt, 21 * pop, a * 0.95)
    return img

# ---------------------------------------------------------------- C：深夜办公室
MSGS = [(14.8, "在吗？"), (15.5, "方案今晚能发我吗"), (16.2, "辛苦再改一版～"), (16.8, "明早九点过一下"), (17.3, "收到请回复"),
        (17.7, "@你 群里看一下"), (18.0, "在吗？？"), (18.25, "客户又提了新需求"), (18.5, "今天能给吗"), (18.7, "再对齐一下"),
        (18.88, "加个班"), (19.05, "在吗"), (19.2, "进度？"), (19.33, "@你"), (19.45, "辛苦"), (19.56, "在？"), (19.66, "……")]
def scene_c(t):
    z = 1.0 + 0.10 * ease((t - 14.0) / 7.0); cx, cy = 720 + 30 * ease((t - 14) / 7), 540
    Xw = (E.X - 720) / z + cx; Yw = (E.Y - 540) / z + cy
    def S(x, y): return ((x - cx) * z + 720, (y - cy) * z + 540)
    img = E.C((0.025, 0.032, 0.045)).expand(3, E.H, E.W).clone()
    # 背窗夜景
    mwin = Mask().rect(*S(-100, 90), *S(1600, 430)).t()
    city = sky_grad((0.03, 0.05, 0.09), (0.10, 0.13, 0.20), 90, 430)
    city = buildings(city, Xw * 1.0 + 40, Yw, 440, 38, 80, 300, 61, E.C((0.06, 0.08, 0.11)), E.C((0.6, 0.75, 0.9)), 0.35, 0.05, 0.25, E.C((0.12, 0.16, 0.24)), 5, 7, 0.9, dxp=1.0 / (z * E.PX) + 1e-6)
    city = buildings(city, Xw * 1.0 + 300, Yw, 450, 55, 50, 220, 62, E.C((0.04, 0.05, 0.08)), E.C((0.6, 0.75, 0.9)), 0.30, 0.05, 0.05, E.C((0.12, 0.16, 0.24)), 7, 9, 0.9, dxp=1.0 / (z * E.PX) + 1e-6)
    img = mix(img, city, mwin[None])
    mul = Mask()
    for x in range(-100, 1700, 180): mul.line([S(x, 90), S(x, 430)], 6 * z)
    mul.line([S(-100, 430), S(1600, 430)], 10 * z)
    img = over(img, E.C((0.05, 0.06, 0.08)), mul.t())
    # 天花板灯（远处先熄）
    VP = (720, 380)
    off_t = [20.4, 19.4, 18.6, 17.8, 17.0]
    for k, f in enumerate([0.15, 0.3, 0.48, 0.7, 0.95]):
        on = 1 - ease((t - off_t[k]) / 0.15)
        y = VP[1] - (VP[1] - 40) * f
        for j in range(-3, 4):
            xc = VP[0] + j * 260 * f
            lm = Mask().poly([S(xc - 70 * f, y - 6 * f), S(xc + 70 * f, y - 6 * f), S(xc + 80 * f, y + 6 * f), S(xc - 80 * f, y + 6 * f)]).t()
            img = add(img, E.C((0.75, 0.85, 1.0)), lm * on * 0.9)
        img = img + E.C((0.25, 0.32, 0.42)) * (torch.exp(-((Yw - y) / (40 + 120 * f)) ** 2) * 0.10 * on)[None]
    # 工位排：显示器冷光
    mon = Mask(); mon2 = Mask(); ppl = Mask()
    for k, f in enumerate([0.22, 0.34, 0.5, 0.7]):
        y = VP[1] + (760 - VP[1]) * f
        dm = Mask().rect(*S(-200, y + 22 * f), *S(1700, y + 30 * f))
        img = over(img, E.C((0.06, 0.07, 0.09)), dm.t())
        for j in range(-6, 7):
            xc = VP[0] + j * 230 * f + 60 * f
            if abs(xc - 720) < 40: continue
            on = hash1(k * 31 + j, 3) < 0.75
            r = [S(xc - 40 * f, y - 40 * f), S(xc + 40 * f, y)]
            (mon if on else mon2).rect(r[0][0], r[0][1], r[1][0], r[1][1])
            leave = 15.0 + hash1(k * 31 + j, 5) * 6
            if hash1(k * 31 + j, 4) < 0.45 and t < leave:
                bust(ppl, *S(xc + 10 * f, y - 30 * f), 40 * f * z, 0)
    img = add(img, E.C((0.30, 0.52, 0.85)), mon.t() * 0.85)
    img = img + E.C((0.2, 0.35, 0.6)) * haze(mon.t()[None], 20)[0][None] * 0.6
    img = over(img, E.C((0.05, 0.06, 0.08)), mon2.t())
    img = over(img, E.C((0.02, 0.025, 0.035)), ppl.t())
    # 墙钟：时间飞转
    ccx, ccy = S(1240, 180); R = 52 * z
    cm = Mask().circle((ccx, ccy), R).t()
    img = over(img, E.C((0.80, 0.84, 0.88)), cm * 0.85)
    hours = 18.5 + 4.2 * ein((t - 14.0) / 6.6)
    hm = Mask()
    for i in range(12):
        a = i / 12 * 6.283; hm.line([(ccx + math.sin(a) * R * 0.82, ccy - math.cos(a) * R * 0.82), (ccx + math.sin(a) * R * 0.92, ccy - math.cos(a) * R * 0.92)], 2 * z)
    ah = (hours % 12) / 12 * 6.283; am = (hours % 1) * 6.283
    hm.line([(ccx, ccy), (ccx + math.sin(ah) * R * 0.5, ccy - math.cos(ah) * R * 0.5)], 4 * z)
    hm.line([(ccx, ccy), (ccx + math.sin(am) * R * 0.78, ccy - math.cos(am) * R * 0.78)], 2.5 * z)
    img = over(img, E.C((0.08, 0.09, 0.11)), hm.t())
    # 前景：主角显示器（聊天窗口刷屏）
    screen_on = 1 - ease((t - 20.35) / 0.25)
    mx0, my0, mx1, my1 = 300, 470, 860, 790
    mm = Mask().rrect(*S(mx0 - 14, my0 - 14), *S(mx1 + 14, my1 + 14), 10 * z).t()
    img = over(img, E.C((0.05, 0.055, 0.065)), mm)
    sm_ = Mask().rect(*S(mx0, my0), *S(mx1, my1)).t()
    scr = E.C((0.70, 0.76, 0.84)) * screen_on + E.C((0.02, 0.02, 0.03)) * (1 - screen_on)
    img = over(img, scr, sm_)
    side = Mask().rect(*S(mx0, my0), *S(mx0 + 120, my1)).t()
    img = over(img, E.C((0.22, 0.26, 0.32)) * screen_on + 0.02, side)
    for i in range(7):
        p = S(mx0 + 60, my0 + 34 + i * 40)
        img = over(img, E.C((0.45, 0.5, 0.58)) * screen_on, Mask().circle(p, 12 * z).t())
        if i < 4: img = over(img, E.C((0.93, 0.27, 0.24)) * screen_on, Mask().circle((p[0] + 10 * z, p[1] - 9 * z), 4.5 * z).t())
    # 消息气泡由底部堆叠上推
    shown = [m for m in MSGS if t >= m[0]]
    ybase = my1 - 30
    clip_y0 = S(0, my0 + 6)[1]
    for idx, (t0, txt) in enumerate(reversed(shown)):
        yy = ybase - idx * 50 + 50 * (1 - ease((t - shown[-1][0]) / 0.18)) * (1 if idx > 0 else 0)
        if yy < my0 + 20: break
        a = ease((t - t0) / 0.15) * screen_on
        sz = 25
        p = S(mx0 + 140, yy)
        tw = len(txt) * sz * 1.0 + 28
        bm = Mask().rrect(p[0], p[1] - 20 * z, p[0] + tw * z, p[1] + 20 * z, 11 * z).t()
        img = over(img, E.C((1.0, 1.0, 1.0)), bm * a)
        img = draw_text(img, txt, p[0] + 14 * z, p[1], sz * z, E.C((0.06, 0.07, 0.09)), a, anchor="l")
    img = img + E.C((0.55, 0.7, 1.0)) * haze(sm_[None], 60)[0][None] * 0.35 * screen_on
    # 主角背影（被屏幕光勾边）
    mh = Mask(); bust(mh, *S(1000, 640), 150 * z, -10 * z)
    ha = mh.t()
    img = over(img, E.C((0.03, 0.035, 0.045)), ha)
    img = add(img, E.C((0.55, 0.72, 1.0)), rim_light(ha, -3, 0) * 0.9 * (0.15 + 0.85 * screen_on))
    # 桌沿 + 咖啡杯
    dk = Mask().poly([S(-100, 800), S(1600, 800), S(1600, 1100), S(-100, 1100)]).t()
    img = over(img, E.C((0.06, 0.065, 0.075)), dk)
    cup = Mask().rrect(*S(1180, 740), *S(1230, 800), 6 * z).t()
    img = over(img, E.C((0.75, 0.75, 0.74)) * 0.5, cup)
    # 日常小物：桌上的小盆栽、显示器边的便利贴、一张小猫照片（家里有只猫在等）
    pot = Mask().poly([S(1262, 760), S(1318, 760), S(1310, 800), S(1270, 800)]).t()
    img = over(img, E.C((0.42, 0.26, 0.20)) * 0.55, pot)
    lv = Mask()
    for k_ in range(6):
        ang = -2.4 + k_ * 0.33; L_ = 46 + 10 * math.sin(k_ * 2.1)
        sway_ = 0.04 * math.sin(t * 1.2 + k_)
        lv.ellipse(S(1290 + math.cos(ang + sway_) * L_ * 0.5, 760 + math.sin(ang + sway_) * L_ * 0.5), (L_ * 0.5 * z, 8 * z), math.degrees(ang + sway_))
    img = over(img, E.C((0.20, 0.36, 0.26)) * (0.45 + 0.25 * screen_on), lv.t())
    for k_, (nx, ny, rot) in enumerate(((mx1 + 6, my0 + 40, 0), (mx1 + 4, my0 + 96, 0), (mx0 - 50, my0 + 60, 0))):
        nm = Mask().rect(*S(nx, ny), *S(nx + 44, ny + 44)).t()
        img = over(img, E.C(((1.0, 0.86, 0.45), (0.65, 0.88, 0.75), (1.0, 0.70, 0.62))[k_]) * (0.35 + 0.35 * screen_on), nm * 0.95)
    fr = Mask().rrect(*S(1090, 712), *S(1160, 800), 4 * z).t()
    img = over(img, E.C((0.85, 0.80, 0.72)) * (0.35 + 0.3 * screen_on), fr)
    ph_ = Mask().rect(*S(1097, 719), *S(1153, 793)).t()
    img = over(img, E.C((0.55, 0.62, 0.70)) * (0.35 + 0.3 * screen_on), ph_)
    mc = Mask(); cat(mc, S(1125, 790)[0], S(1125, 790)[1], 34 * z, "sit", t=t)
    img = paint_cat(img, mc.t() * ph_, E.C((0.55, 0.55, 0.6)) * (0.4 + 0.5 * screen_on))
    return img

# ---------------------------------------------------------------- D：雨夜街道（向右跟拍）→ 回到楼里，镜头推进自家亮灯的窗
LAMP_X = 760.0
DOOR_X = 945.0
HOME_WIN = (1000.0, 440.0, 1080.0, 500.0)    # 自家窗户（世界坐标，4:3）
def hero_d(t):
    """返回 (x, pose, 相位, 不透明度)"""
    u = (t - 21.4) / 4.2
    if u < 1:
        x = 330 + (DOOR_X - 330) * (u - 0.06 * math.sin(u * math.pi))
        return x, "trudge", 2 * math.pi * 1.25 * t, 1.0
    return DOOR_X, "stand", 0.0, 1 - ease((t - 25.6) / 0.4)

D_PUSH = (25.4, 28.4)
def d_cam(t):
    def follow(tt): return 1.0, min(hero_d(tt)[0], DOOR_X) + 40, 540
    if t < D_PUSH[0]: return follow(t)
    u = eio((t - D_PUSH[0]) / (D_PUSH[1] - D_PUSH[0]))
    z0, cx0, cy0 = follow(D_PUSH[0])
    wcx, wcy = (HOME_WIN[0] + HOME_WIN[2]) / 2, (HOME_WIN[1] + HOME_WIN[3]) / 2
    z = z0 * (19.5 / z0) ** u
    # 让窗户中心在屏幕上匀速滑到正中（而不是镜头中心直线移动），推镜才不会“甩”
    p0x = (wcx - cx0) * z0 + 720; p0y = (wcy - cy0) * z0 + 540
    px = lerp(p0x, 720, u); py = lerp(p0y, 540, u)
    return z, wcx - (px - 720) / z, wcy - (py - 540) / z

def scene_d(t, home_img=None):
    YG = 840
    hx, pname, ph, halpha = hero_d(t)
    z, cx, cy = d_cam(t)
    Xw = (E.X - 720) / z + cx; Yw = (E.Y - 540) / z + cy
    def S(x, y): return ((x - cx) * z + 720, (y - cy) * z + 540)
    img = sky_grad((0.03, 0.04, 0.07), (0.09, 0.11, 0.16), 0, 840)
    img = img + E.C((0.15, 0.2, 0.3)) * (torch.exp(-((Yw - 600) / 260) ** 2) * 0.25)[None]
    fogc = E.C((0.10, 0.13, 0.18))
    img = buildings(img, Xw * 0.35 + 900, Yw, 760, 46, 160, 420, 71, E.C((0.05, 0.065, 0.09)), E.C((0.55, 0.70, 0.85)), 0.3, 0.03, 0.45, fogc, 6, 8, 0.8, dxp=0.35 / (z * E.PX) + 1e-6)
    img = buildings(img, Xw * 0.6 + 500, Yw, 800, 70, 140, 380, 72, E.C((0.045, 0.055, 0.075)), E.C((0.6, 0.75, 0.9)), 0.28, 0.04, 0.2, fogc, 8, 11, 0.9, dxp=0.6 / (z * E.PX) + 1e-6)
    img = buildings(img, Xw * 1.0, Yw, 830, 150, 300, 560, 73, E.C((0.035, 0.042, 0.058)), E.C((0.65, 0.8, 0.95)), 0.22, 0.06, 0.0, fogc, 16, 20, 0.85, dxp=1.0 / (z * E.PX) + 1e-6)
    # 店招冷光
    sg = Mask()
    for i in range(-2, 12):
        x0 = i * 300 + 40 + 60 * hash1(i, 3)
        if 820 < x0 < 1320: continue
        if hash1(i, 4) < 0.6: sg.rect(*S(x0, 700 + 40 * hash1(i, 5)), *S(x0 + 120 + 80 * hash1(i, 6), 730 + 40 * hash1(i, 5)))
    sgt = sg.t()
    img = add(img, E.C((0.35, 0.75, 0.85)), sgt * 0.8)
    img = img + E.C((0.2, 0.5, 0.6)) * haze(sgt[None], 30)[0][None] * 0.5
    # 街角便利店（暖光，日常的一点温度）
    shop = Mask().rect(*S(110, 650), *S(430, YG)).t()
    img = over(img, E.C((0.95, 0.82, 0.60)) * 0.55, shop)
    shelf = Mask()
    for k in range(4): shelf.rect(*S(130, 690 + k * 36), *S(410, 694 + k * 36))
    for k in range(9): shelf.rect(*S(135 + k * 31, 700), *S(150 + k * 31, 820), 90)
    img = over(img, E.C((0.55, 0.42, 0.32)) * 0.6, shelf.t() * shop)
    awn = Mask().rect(*S(95, 630), *S(445, 652)).t()
    img = over(img, E.C((0.30, 0.55, 0.45)) * 0.5, awn)
    img = img + E.C((1.0, 0.8, 0.55)) * haze(shop[None], 50)[0][None] * 0.35
    # 自家那栋楼（近景立面）
    fac = Mask().rect(*S(860, 300), *S(1300, YG)).t()
    img = over(img, E.C((0.055, 0.06, 0.075)), fac)
    wl = Mask(); ww = Mask(); frame_m = Mask()
    for c in range(4):
        for r in range(4):
            x0 = 890 + c * 110 if c < 3 else 1210; y0 = 340 + r * 100
            if (x0, y0) == (HOME_WIN[0], HOME_WIN[1]): continue
            if r == 3 and c < 2: continue
            hsh = hash1(c * 7 + r, 41)
            (wl if hsh < 0.3 else ww).rect(*S(x0, y0), *S(x0 + 80, y0 + 60))
            frame_m.rect(*S(x0 - 4, y0 + 60), *S(x0 + 84, y0 + 66))
    img = over(img, E.C((0.95, 0.72, 0.45)) * 0.55, wl.t()); img = over(img, E.C((0.10, 0.13, 0.18)), ww.t())
    img = img + E.C((1.0, 0.7, 0.4)) * haze(wl.t()[None], 16)[0][None] * 0.25
    img = over(img, E.C((0.12, 0.12, 0.14)), frame_m.t())
    # 单元门：主角走到时亮起
    door_on = ease((t - 24.8) / 0.6)
    dm = Mask().rect(*S(DOOR_X - 32, 730), *S(DOOR_X + 32, YG)).t()
    img = over(img, E.C((0.10, 0.10, 0.12)) + E.C((1.0, 0.80, 0.52)) * 0.75 * door_on, dm)
    img = img + E.C((1.0, 0.75, 0.45)) * haze(dm[None], 40)[0][None] * 0.5 * door_on
    # 自家窗户：主角进门后灯亮，窗里就是家（连续镜头直接推进去）
    x0, y0, x1, y1 = S(*HOME_WIN[:2]) + S(*HOME_WIN[2:])
    on = ease((t - 26.0) / 0.35)
    wm_ = Mask().rect(x0, y0, x1, y1).t()
    if home_img is not None:
        inside = E.portal(home_img, x0, y0, x1, y1) * (0.18 + 0.82 * on)
    else:
        inside = E.C((0.08, 0.09, 0.12)).expand(3, E.H, E.W)
    img = mix(img, inside, wm_[None])
    img = img + E.C((1.0, 0.72, 0.42)) * haze(wm_[None], 26)[0][None] * 0.45 * on * (1 - ease((z - 3) / 6))
    sill = Mask().rect(x0 - 4 * z, y1, x1 + 4 * z, y1 + 6 * z).t()
    img = over(img, E.C((0.12, 0.12, 0.14)), sill)
    # 人行道
    sw = Mask().rect(*S(-2000, YG), *S(4000, YG + 12)).t()
    img = over(img, E.C((0.07, 0.08, 0.10)), sw)
    # 路灯
    lm = Mask(); lights = []
    for lx in (LAMP_X - 640, LAMP_X, LAMP_X + 640):
        lm.line([S(lx, YG), S(lx, 420)], 7 * z); lm.line([S(lx, 420), S(lx - 60, 410)], 5 * z)
        lm.rrect(*S(lx - 92, 404), *S(lx - 44, 418), 4 * z)
        lights.append(S(lx - 68, 418))
    img = over(img, E.C((0.05, 0.06, 0.07)), lm.t())
    cone = Mask()
    for (px, py) in lights:
        cone.poly([(px - 18 * z, py), (px + 18 * z, py), (px + 190 * z, py + 422 * z), (px - 190 * z, py + 422 * z)])
    ct = cone.t(14)
    img = img + E.C((0.70, 0.82, 1.0)) * (ct * 0.16)[None]
    for (px, py) in lights:
        img = img + E.C((0.8, 0.9, 1.0)) * (torch.exp(-((E.X - px) ** 2 + (E.Y - py) ** 2) / (16 * z) ** 2) * 1.4 + torch.exp(-((E.X - px) ** 2 + (E.Y - py) ** 2) / (90 * z) ** 2) * 0.2)[None]
    pool = Mask()
    for (px, py) in lights: pool.ellipse((px, S(0, YG)[1] + 4 * z), (210 * z, 16 * z))
    img = add(img, E.C((0.55, 0.65, 0.8)), pool.t(10) * 0.35)
    # 撑伞的路人（反方向走过）
    mp = Mask(); mu = Mask()
    for k, (x0p, spd, hh) in enumerate(((1650, 92, 285), (2050, 78, 270))):
        px_ = x0p - spd * (t - 21.0)
        kk = figure(mp, *S(px_, YG), hh * z, pose("walk", 2 * math.pi * 1.1 * t + k), flip=True, pack=False)
        hd = kk["head"]
        mu.ellipse((hd[0] - 10 * z, hd[1] - 22 * z), (68 * z, 34 * z), 0, 255, 180, 360)
        mu.line([(hd[0] - 10 * z, hd[1] - 22 * z), kk["hand"]], 2.5 * z)
    pa = 1 - ease((t - D_PUSH[0]) / 0.6)
    img = over(img, E.C((0.04, 0.045, 0.055)), mp.t() * pa)
    ua = mu.t() * pa
    img = over(img, E.C((0.16, 0.20, 0.30)), ua)
    img = add(img, E.C((0.7, 0.82, 1.0)), rim_light(ua, 0, 3.0) * 0.6)
    # 主角（帽衫、背包）
    if halpha > 0.01:
        mh = Mask()
        figure(mh, *S(hx, YG), 300 * z, pose(pname, ph), hood=True, pack=True)
        ha = mh.t() * halpha
        img = over(img, E.C((0.035, 0.04, 0.05)), ha)
        lamp_near = math.exp(-((hx - (LAMP_X - 68)) / 260) ** 2)
        img = add(img, E.C((0.7, 0.82, 1.0)), (rim_light(ha, 0, 3.5) * (0.6 + 0.9 * lamp_near) + rim_light(ha, 2.5, 0) * 0.7 + rim_light(ha, -2.5, 0) * 0.5))
        img = add(img, E.C((1.0, 0.75, 0.45)), rim_light(ha, -2.5, 0) * door_on * 0.8)
    # 路面倒影（湿地面）
    r0 = int(max(0, min(E.H, S(0, YG + 12)[1] * E.PX)))
    if r0 < E.H:
        n = E.H - r0
        src_top = max(0, r0 - n)
        flip = torch.flip(img[:, src_top:r0], [1])
        if flip.shape[1] < n: flip = torch.cat([flip, flip[:, -1:].expand(-1, n - flip.shape[1], -1)], 1)
        flip = E.dir_blur(flip.contiguous(), 0, 24, 8)
        ripple = (E.vnoise(E.X[r0:] * 0.03, E.Y[r0:] * 0.18 - t * 2, 81) - 0.5)
        road = E.C((0.02, 0.025, 0.035)) + flip[:, :n] * (0.42 + ripple * 0.2)[None]
        img[:, r0:] = road
    # 车灯
    g = np.random.default_rng(8); n_c = 18
    dirn = np.where(g.uniform(0, 1, n_c) < 0.5, -1, 1); sp = g.uniform(500, 800, n_c) * dirn
    x = (g.uniform(0, 3000, n_c) + sp * t) % 3000 - 800
    y = np.where(dirn > 0, 930, 990)
    pts = []; cols = []
    for xi, yi, di in zip(x, y, dirn):
        for off in (-22, 22):
            pts.append(S(xi + off, yi)); cols.append([1.0, 0.95, 0.85] if di < 0 else [1.0, 0.22, 0.18])
    img = img + glow_points(np.array(pts), np.full(len(pts), 2.5), np.array(cols), sigma=5 * min(z, 3), core=0.3)
    # 雨（更少、更长、更柔；随推镜淡出）
    rain_a = 1 - ease((t - 26.6) / 1.3)
    if rain_a > 0.01:
        rm = Mask(); rm2 = Mask()
        N = 520
        idx = np.arange(N)
        rx = (hash1(idx, 11) * 1700 - 130 + t * 120) % 1700 - 130
        ry = (hash1(idx, 12) * 1300 + t * 1300 * (0.8 + 0.4 * hash1(idx, 13))) % 1300 - 110
        L = 26 + 26 * hash1(idx, 14)
        segs = [np.array([[x0_, y0_], [x0_ + L0 * 0.12, y0_ + L0]]) for x0_, y0_, L0 in zip(rx, ry, L)]
        rm.lines(segs[:360], 1.2, 120); rm2.lines(segs[360:], 2.2, 100)
        rt = rm.t(0.5); rt2 = rm2.t(1.0)
        img = add(img, E.C((0.65, 0.75, 0.88)), (rt * 0.32 + rt2 * 0.22) * (1 + ct * 2.5) * rain_a)
    return img

# ---------------------------------------------------------------- H：到家（暖光小屋 + 小猫）→ 手机里的山野
H0 = 28.4            # 进屋的时刻（窗户推镜完成）
H_POV = (32.6, 33.4) # 宽景 → 主观镜头
TAP_T = 34.6         # 小猫拍了一下手机屏幕
PUSH = (35.2, 38.6)  # 推进手机，画面接管为真实山野（= 自然段旧时间 32.7）
HERO_H = (905.0, 834.0, 330.0)
WARM = (1.0, 0.80, 0.56)

def h_cat(t):
    """返回 (x, yg, s, pose, 相位, flip, 亮度系数)"""
    if t < 29.6: return 240.0, 1150.0, 230.0, "sit", 0.0, False, 0.0
    if t < 30.1:
        u = ein((t - 29.6) / 0.5); return 240 + 80 * u, 1150 + 520 * u, 230.0, "jump", 0.0, False, 0.0
    if t < 31.8:
        u = (t - 30.2) / 1.6; return lerp(330, 690, cl(u)), 872.0, 58.0, ("walk" if t > 30.2 else "sit"), 2 * math.pi * 2.2 * t, False, 1.0
    if t < 32.25:
        u = (t - 31.8) / 0.45; ue = ease(u)
        return lerp(690, 760, ue), lerp(872, 744, ue) - 70 * math.sin(math.pi * cl(u)), 56.0, "jump", 0.0, False, 1.0
    return 760.0, 744.0, 54.0, "loaf", 0.0, False, 1.0

def scene_h(t, cam=None):
    z, cx, cy = cam if cam is not None else (1.0 + 0.06 * ease((t - H0) / 4.2), 760.0, 600.0)
    Xw = (E.X - 720) / z + cx; Yw = (E.Y - 540) / z + cy
    def S(x, y): return ((x - cx) * z + 720, (y - cy) * z + 540)
    LX, LY = 1292.0, 400.0                                    # 落地灯
    lampd = torch.sqrt((Xw - LX) ** 2 + (Yw - LY) ** 2)
    lamp = torch.exp(-lampd / 520) * 0.55 + torch.exp(-lampd / 160) * 0.25
    lamp_mean = lamp.mean()  # 保留 GPU 标量，不调用 float()，避免每本书都阻塞 CUDA
    img = mix(E.C((0.33, 0.24, 0.20)), E.C((0.44, 0.33, 0.26)), sm(120, 800, Yw)[None])
    img = img * (0.62 + 0.6 * lamp)[None]
    # 地板 + 地毯
    fl = sm(818, 822, Yw)
    floor = mix(E.C((0.34, 0.24, 0.18)), E.C((0.22, 0.15, 0.11)), sm(820, 1150, Yw)[None]) * (0.7 + 0.5 * lamp)[None]
    img = mix(img, floor, fl[None])
    plank = Mask()
    for yy in (858, 912, 990, 1090): plank.line([S(-300, yy), S(1800, yy)], 2.0 * z)
    img = over(img, E.C((0.18, 0.12, 0.09)), plank.t() * 0.5)
    base_line = Mask().rect(*S(-300, 806), *S(1800, 820)).t()
    img = over(img, E.C((0.52, 0.42, 0.34)) * (0.7 + 0.4 * lamp), base_line)
    rug = Mask().ellipse(S(860, 930), (430 * z, 74 * z)).t(1.0)
    img = over(img, E.C((0.66, 0.46, 0.36)) * (0.7 + 0.45 * lamp), rug)
    rug2 = Mask().ellipse(S(860, 930), (360 * z, 56 * z), w=5 * z).t()
    img = over(img, E.C((0.85, 0.70, 0.52)) * (0.6 + 0.4 * lamp), rug2 * 0.6)
    # 书架
    bs = Mask().rect(*S(110, 290), *S(410, 812)).t()
    img = over(img, E.C((0.30, 0.20, 0.15)) * 0.9, bs)
    inner = Mask()
    for k in range(5): inner.rect(*S(126, 306 + k * 100), *S(394, 392 + k * 100))
    img = over(img, E.C((0.20, 0.13, 0.10)), inner.t())
    bcols = [(0.75, 0.42, 0.32), (0.36, 0.52, 0.58), (0.85, 0.72, 0.48), (0.45, 0.55, 0.40), (0.82, 0.80, 0.74), (0.55, 0.38, 0.50)]
    # 书本位置/尺寸/颜色完全沿用原逻辑；只将同颜色的不相交矩形合并到一个遮罩
    book_masks = [Mask() for _ in bcols]
    for k in range(5):
        xx = 132.0; j = 0
        while xx < 380:
            wb = 14 + 14 * hash1(k * 50 + j, 61); hb = 52 + 30 * hash1(k * 50 + j, 62)
            if hash1(k * 50 + j, 63) < 0.12: xx += 22; j += 1; continue
            ci = int(hash1(k * 50 + j, 64) * 6) % 6
            book_masks[ci].rect(*S(xx, 392 + k * 100 - hb), *S(min(xx + wb, 392), 392 + k * 100))
            xx += wb + 2; j += 1
    book_brightness = 0.55 + 0.35 * lamp_mean
    for ci, book_mask in enumerate(book_masks):
        img = over(img, E.C(bcols[ci]) * book_brightness, book_mask.t())
    # 墙上的小画（山与湖——那个周末要去的地方）
    pf = Mask().rect(*S(690, 300), *S(900, 450)).t()
    img = over(img, E.C((0.85, 0.78, 0.66)) * (0.75 + 0.3 * lamp), pf)
    pin = Mask().rect(*S(702, 312), *S(888, 438)).t()
    img = over(img, E.C((0.98, 0.80, 0.60)) * 0.85, pin)
    mt_ = Mask().poly([S(702, 400), S(760, 345), S(800, 375), S(850, 330), S(888, 380), S(888, 438), S(702, 438)]).t()
    img = over(img, E.C((0.55, 0.45, 0.55)), mt_ * pin)
    lk = Mask().rect(*S(702, 405), *S(888, 438)).t()
    img = over(img, E.C((0.45, 0.60, 0.70)), lk * pin)
    sun_ = Mask().circle(S(845, 360), 11 * z).t()
    img = over(img, E.C((1.0, 0.92, 0.70)), sun_)
    # 墙上的暖色串灯
    pts = []
    for i in range(16):
        u = i / 15; x_ = lerp(480, 1180, u); y_ = 240 + 60 * math.sin(u * math.pi)
        pts.append(S(x_, y_))
    pts = np.array(pts); tw = 0.8 + 0.2 * np.sin(t * 1.5 + np.arange(16) * 2.3)
    wire = Mask().line([tuple(p) for p in pts], 1.4 * z).t()
    img = over(img, E.C((0.15, 0.10, 0.08)), wire * 0.8)
    img = img + glow_points(pts, tw * 1.4, np.broadcast_to(np.array([1.0, 0.78, 0.45]), (16, 3)), sigma=3.0 * z, core=0.5)
    # 落地灯
    lp = Mask(); lp.line([S(LX, 812), S(LX, 410)], 6 * z); lp.ellipse(S(LX, 812), (40 * z, 7 * z))
    img = over(img, E.C((0.16, 0.12, 0.10)), lp.t())
    shade = Mask().poly([S(LX - 52, 330), S(LX + 52, 330), S(LX + 76, 412), S(LX - 76, 412)]).t()
    img = over(img, E.C((1.0, 0.86, 0.62)), shade)
    img = img + E.C(WARM) * haze(shade[None], 60)[0][None] * 0.6
    # 盆栽（大叶子，轻轻晃）
    pot = Mask().poly([S(440, 760), S(520, 760), S(508, 818), S(452, 818)]).t()
    img = over(img, E.C((0.78, 0.55, 0.42)) * (0.6 + 0.4 * lamp), pot)
    lv = Mask()
    for k in range(7):
        ang = -2.6 + k * 0.38 + 0.05 * math.sin(t * 1.1 + k * 1.3) + 0.12 * win(t, TAP_T, 40, 0.3, 2) * math.sin(t * 6 + k)
        L_ = 120 + 40 * hash1(k, 71)
        cxl = 480 + math.cos(ang) * L_ * 0.55; cyl = 750 + math.sin(ang) * L_ * 0.55
        lv.ellipse(S(cxl, cyl), (L_ * 0.5 * z, 22 * z), math.degrees(ang))
        lv.line([S(480, 758), S(cxl, cyl)], 3 * z)
    img = over(img, E.C((0.28, 0.46, 0.32)) * (0.6 + 0.45 * lamp), lv.t())
    # 猫碗
    bowl = Mask().ellipse(S(565, 858), (34 * z, 12 * z), 0, 255, 0, 180).rect(*S(531, 846), *S(599, 858)).t()
    img = over(img, E.C((0.85, 0.45, 0.40)) * (0.7 + 0.3 * lamp), bowl)
    # 沙发
    sc = E.C((0.32, 0.47, 0.50)) * (0.7 + 0.45 * lamp)
    so = Mask()
    so.rrect(*S(610, 580), *S(1210, 760), 34 * z)        # 靠背
    so.rrect(*S(570, 700), *S(1250, 830), 26 * z)        # 坐垫底座
    img = over(img, sc * 0.82, so.t())
    cu = Mask(); cu.rrect(*S(640, 724), *S(918, 772), 20 * z); cu.rrect(*S(922, 724), *S(1180, 772), 20 * z)
    img = over(img, sc, cu.t())
    arms = Mask(); arms.rrect(*S(548, 640), *S(640, 830), 30 * z); arms.rrect(*S(1180, 640), *S(1272, 830), 30 * z)
    am = arms.t()
    img = over(img, sc * 0.95, am)
    img = add(img, E.C(WARM), rim_light(am, -2.5, 2.5) * 0.4)
    legs = Mask(); legs.rect(*S(590, 830), *S(606, 846)); legs.rect(*S(1214, 830), *S(1230, 846))
    img = over(img, E.C((0.20, 0.14, 0.10)), legs.t())
    # 扶手上的马克杯 + 热气
    mug = Mask(); mug.rrect(*S(1212, 606), *S(1240, 640), 5 * z); mug.circle(S(1244, 622), 9 * z, w=4 * z)
    img = over(img, E.C((0.96, 0.92, 0.84)) * (0.75 + 0.35 * lamp), mug.t())
    ag = (t * 0.35 + hash1(np.arange(14), 91)) % 1.0
    sx = 1226 + np.sin(ag * 7 + np.arange(14)) * 7 + ag * 8; sy = 600 - ag * 80
    sp = np.array([S(a, b) for a, b in zip(sx, sy)])
    img = img + glow_points(sp, np.sin(ag * np.pi) * 0.22, np.array([1.0, 0.95, 0.9]), sigma=5.0 * z, core=0.3)
    # 地上的背包（刚回家随手一放）
    bg_ = Mask(); bg_.rrect(*S(1040, 770), *S(1120, 850), 18 * z); bg_.line([S(1052, 772), S(1080, 735), S(1108, 772)], 5 * z)
    img = over(img, E.C((0.45, 0.30, 0.22)) * (0.6 + 0.4 * lamp), bg_.t())
    # 主角（坐在沙发上看手机）
    hx, hyg, hh = HERO_H
    breath = math.sin(t * 1.4)
    mh = Mask(); kh = figure(mh, *S(hx, hyg), hh * z, pose("sofa", breath * 0.3), pack=False, hood=True)
    ha = mh.t()
    img = over(img, E.C((0.78, 0.60, 0.44)) * (0.62 + 0.4 * lamp), ha)       # 到家换上的米色卫衣
    img = add(img, E.C(WARM), rim_light(ha, 3.0, -1.0) * 0.6)
    img = over(img, E.C((0.0, 0.0, 0.0)), (ha - E.shift(ha[None], 0, -5)[0]).clamp(0, 1) * 0.18)
    hp = kh["hand"]
    phm = Mask().rrect(hp[0] - 8 * z, hp[1] - 22 * z, hp[0] + 8 * z, hp[1] + 4 * z, 3 * z).t()
    img = over(img, E.C((0.06, 0.06, 0.07)), phm)
    face = torch.exp(-(((E.X - hp[0] + 18 * z) / (40 * z)) ** 2 + ((E.Y - hp[1] + 44 * z) / (40 * z)) ** 2))
    img = img + E.C((0.75, 0.85, 1.0)) * (face * ha * 0.35)[None]
    # 小猫
    cx_, cyg, cs, cp, cph, cfl, lit = h_cat(t)
    mc = Mask(); kc = cat(mc, *S(cx_, cyg), cs * z, cp, cph, cfl, t)
    ac = mc.t(0.4 if cs > 150 else 0.0)
    if lit < 0.5:   # 窗台上的背影：被屋里的灯勾出一圈暖边
        img = over(img, E.C((0.10, 0.07, 0.06)), ac)
        img = add(img, E.C(WARM), rim_light(ac, 4.0, 2.0) * 1.2)
    else:
        img = paint_cat(img, ac, E.C((1.0, 0.92, 0.82)) * (0.65 + 0.4 * lamp_mean), key=WARM, key_k=0.8,
                        rim_col=WARM, rim_k=0.7, rim_dir=(3.0, -1.5))
    return img, (hp[0], hp[1] - 10 * z)

# ---- 主观镜头：手里的手机
PH_C = (720.0, 560.0); PH_W, PH_H = 330.0, 600.0
TAP_P = (835.0, 735.0)
def pov_zoom(t):
    if t < PUSH[0]: return 1.0 + 0.04 * ease((t - H_POV[0]) / 2.6)
    u = eio((t - PUSH[0]) / (PUSH[1] - PUSH[0]))
    return 1.04 * (5.2 / 1.04) ** u

def scene_pov(t, nat):
    z = pov_zoom(t)
    def S(x, y): return ((x - PH_C[0]) * z + 720, (y - PH_C[1]) * z + 540)
    tap = ease((t - TAP_T) / 0.5)
    # 背景：低头看到的地毯与小屋，浅景深虚化
    room, _ = scene_h(t, cam=(1.5, 900.0, 860.0))
    bgi = E.gblur(room, 16 * E.PX) * 0.72
    bgi = E.zoom_about(bgi, z ** 0.35, 720, 540)
    img = bgi
    # 手机
    x0, y0 = S(PH_C[0] - PH_W / 2, PH_C[1] - PH_H / 2); x1, y1 = S(PH_C[0] + PH_W / 2, PH_C[1] + PH_H / 2)
    body = Mask().rrect(x0 - 12 * z, y0 - 12 * z, x1 + 12 * z, y1 + 12 * z, 46 * z).t()
    img = over(img, E.C((0.07, 0.07, 0.08)), body)
    img = add(img, E.C(WARM), rim_light(body, 3, -2) * 0.4)
    scr = Mask().rrect(x0, y0, x1, y1, 36 * z).t()
    sc = E.portal(nat, x0, y0, x1, y1)
    dim = 0.62 + 0.38 * tap
    l = (sc * E.C((0.299, 0.587, 0.114))).sum(0, keepdim=True)
    sc = (l + (sc - l) * (0.7 + 0.3 * tap)) * dim
    img = mix(img, sc, scr[None])
    # 屏幕 UI（点开前：锁屏时间）
    ui = 1 - ease((t - TAP_T) / 0.4)
    if ui > 0.01:
        cxs = (x0 + x1) / 2
        img = draw_text(img, "22:47", cxs, y0 + 90 * z, 64 * z, E.C((1, 1, 1)), ui * 0.92, shadow=0.5)
        img = draw_text(img, "周六 · 晴 22°", cxs, y0 + 140 * z, 20 * z, E.C((1, 1, 1)), ui * 0.85, shadow=0.5)
        dots = Mask()
        for k in range(3): dots.circle((cxs + (k - 1) * 16 * z, y1 - 40 * z), 4 * z, 255 if k == 1 else 120)
        img = add(img, E.C((1, 1, 1)), dots.t() * ui * 0.8)
    # 点击涟漪
    if TAP_T <= t < TAP_T + 1.0:
        u = (t - TAP_T) / 1.0
        tp = S(*TAP_P)
        ring = Mask().circle(tp, (20 + 160 * eout(u)) * z, w=4 * z).t(1.0)
        img = add(img, E.C((1, 1, 1)), ring * scr * (1 - u) * 0.7)
    # 屏幕的光照到手和周围
    glow_k = 0.25 + 0.55 * tap
    img = img + E.C((1.0, 0.85, 0.6)) * haze(scr[None], 80)[0][None] * glow_k * 0.35 * (1 - ease((z - 2) / 2.5))
    # 手
    skin = E.C((0.93, 0.74, 0.62)); sleeve = E.C((0.24, 0.27, 0.34))
    mhd = Mask(); msl = Mask()
    for sg in (-1, 1):
        ex = PH_C[0] + sg * (PH_W / 2 + 6)
        mhd.ellipse(S(ex + sg * 75, 900), (90 * z, 115 * z), sg * 18)
        if sg < 0:
            for k, fy in enumerate((650, 700, 750)):
                mhd.line([S(ex - 50, fy + 50), S(ex + 16, fy)], 24 * z); mhd.circle(S(ex + 16, fy), 12 * z)
        else:
            mhd.line([S(ex + 50, 880), S(ex - 34, 790)], 30 * z); mhd.circle(S(ex - 34, 790), 15 * z)   # 右手拇指搭在屏幕边
        msl.poly([S(ex + sg * 10, 930), S(ex + sg * 180, 900), S(ex + sg * 260, 1300), S(ex - sg * 60, 1300)])
    ah = mhd.t(); asl = msl.t()
    img = over(img, skin * (0.75 + 0.3 * glow_k), ah)
    img = add(img, E.C(WARM), rim_light(ah, 0, 3) * 0.5)
    img = over(img, sleeve, asl)
    # 小猫凑过来，伸爪拍屏幕
    peek = ease((t - 33.6) / 0.7) * (1 - ease((t - PUSH[0] - 0.4) / 1.2))
    if peek > 0.01:
        mc = Mask()
        hx_, hy_ = S(1110 + 180 * (1 - peek), 900 + 180 * (1 - peek))
        cat(mc, hx_ - 0.14 * 260 * z, hy_ + 1.02 * 260 * z, 260 * z, "lookup" if t < TAP_T - 0.5 else "sit", t=t)
        pu = ease((t - (TAP_T - 0.5)) / 0.5) * (1 - ease((t - TAP_T - 0.15) / 0.5))
        if pu > 0.01:
            tip = S(lerp(1060, TAP_P[0] + 14, pu), lerp(1060, TAP_P[1] + 18, pu))
            base = S(1040, 1060)
            mc.line([base, tip], 54 * z); mc.circle(tip, 30 * z)
        ac = mc.t()
        img = paint_cat(img, ac, E.C((1.0, 0.92, 0.82)) * 0.85, key=WARM, key_k=0.9, rim_col=(1.0, 0.85, 0.6), rim_k=0.6 + 0.5 * tap, rim_dir=(-3.0, -2.0))
    # 点开之后：山风从屏幕里吹出来（风线 + 金色花粉飘进屋里）
    spill = win(t, TAP_T + 0.1, PUSH[1], 0.6, 1.0)
    if spill > 0.01:
        img = _wind_lines(img, t, spill * 0.7)
        n = 90; idx = np.arange(n)
        age = (t * 0.32 + hash1(idx, 21)) % 1.0
        ang = hash1(idx, 22) * 6.283
        r = 40 + age * (260 + 300 * hash1(idx, 23))
        px = PH_C[0] + np.cos(ang) * r * 1.2 + age * 120; py = PH_C[1] + np.sin(ang) * r - age * 140
        pts = np.array([S(a, b) for a, b in zip(px, py)])
        bb = np.sin(age * np.pi) * 1.3 * spill
        img = img + glow_points(pts, bb, np.array([1.0, 0.85, 0.55]), sigma=2.6 * min(z, 2.5), core=0.45)
    return img

def nature_frame(tn):
    """自然段完整一帧（含风线、飞鸟、流星），tn 为自然段旧时间轴"""
    P = tod_at(tn)
    city = ease((tn - 90.0) / 2.6)
    img, info = nature(tn, tod=P, city=city)
    img = _wind_lines(img, tn, win(tn, 30.5, 43.5, 2.0, 2.0) * 1.0 + win(tn, 51.4, 55.0, 0.8, 1.5) * 0.6 + win(tn, 86.5, 92.0, 1, 1) * 0.5)
    img = _birds(img, tn, P)
    img = _meteor(img, tn, [(79.8, 1150, 150, -560, 250, 1.1), (82.6, 300, 120, 420, 180, 0.9), (84.0, 1000, 90, -300, 140, 0.7)])
    return img


# 雨滴玻璃（C→D 的对焦转场）
def raindrops(img_back, img_front, u, t):
    """u: 0=前景(C) 1=背景(D)；雨滴在两者之上"""
    out = mix(img_front, img_back, ease(u))
    N = 70
    xs = hash1(np.arange(N), 31) * DW; ys = (hash1(np.arange(N), 32) * DH + t * 90 * hash1(np.arange(N), 33) ** 3) % DH
    rs = 8 + 22 * hash1(np.arange(N), 34) ** 2
    m = Mask(); rim = Mask()
    for x, y, r in zip(xs, ys, rs):
        m.ellipse((x, y), (r, r * 1.2)); rim.circle((x - r * 0.35, y - r * 0.45), max(1.2, r * 0.16))
    a = math.sin(math.pi * cl(u)) ** 0.6
    mt = m.t(1.0)
    blur = E.gblur(out, 7 * E.PX)
    out = mix(out, blur, (a * 0.85))
    inner = E.gblur(out, 2 * E.PX) * 1.35 + 0.03     # 水珠里更亮、更清晰（像一颗颗小透镜）
    out = mix(out, inner, (mt * a)[None])
    out = out + E.C((0.8, 0.88, 1.0)) * (rim.t(0.6) * 0.5 * a)[None]
    out = out * (1 - (mt - E.gblur(mt[None], 2.5 * E.PX)[0]).clamp(0, 1) * 0.5 * a)[None]
    return out

# ======================================================================= 合成：分镜与连续转场
def grade_of(t):
    """冷暖调色：0=城市 1=自然（回家的小屋是暖的）"""
    return kf(t, [(0, 0.0), (25.6, 0.0), (H0, 0.62), (TAP_T, 0.66), (PUSH[1], 1.0),
                  (91.5 + NS, 1.0), (94.0 + NS, 0.25), (101.5 + NS, 0.3), (104.5 + NS, 1.0), (300, 1.0)], ease)

def frame(t, fidx=0):
    # ---------------- A：城市航拍 → 推进车窗（窗内就是 B 车厢）
    if t < 7.2:
        img, (p0, p1) = scene_a(t)
        if t >= 5.6:
            b = scene_b(t)
            sc = E.portal(b, p0[0], p0[1], p1[0], p1[1])
            rr = Mask().rrect(p0[0], p0[1], p1[0], p1[1], 3 * (p1[0] - p0[0]) / 38).t()
            img = mix(img, sc, (rr * ease((t - 5.6) / 0.6))[None])
        return img
    # ---------------- B：地铁车厢
    if t < 13.4:
        return notes_layer(scene_b(t), t, NOTES_B)
    # ---------------- B → C：柔和横移（比原来的甩镜更轻、更顺）
    if t < 14.6:
        u = eio((t - 13.4) / 1.2)
        b = notes_layer(scene_b(t), t, NOTES_B); c = scene_c(t)
        off = u * DW; bl = 260 * math.sin(math.pi * u)
        b = E.dir_blur(E.shift(b, -off, 0), bl, 0, 10); c = E.dir_blur(E.shift(c, DW - off, 0), bl, 0, 10)
        edge = sm(DW - off - 60, DW - off + 60, E.X)
        return mix(b, c, edge[None])
    # ---------------- C：深夜办公室
    if t < 20.6:
        return scene_c(t)
    # ---------------- C → D：雨滴玻璃上的焦点转移
    if t < 21.9:
        return raindrops(scene_d(t), scene_c(t), (t - 20.6) / 1.3, t)
    # ---------------- D：雨夜街道 → 自家窗户亮灯 → 推进窗户
    if t < H0:
        home = scene_h(t)[0] if t > 24.0 else None
        return scene_d(t, home)
    # ---------------- H：到家，小猫从窗台跳下来，跑到沙发上
    if t < H_POV[0]:
        return scene_h(t)[0]
    # ---------------- H → 主观镜头：向手机推近 + 焦点转移
    if t < H_POV[1]:
        u = ease((t - H_POV[0]) / (H_POV[1] - H_POV[0]))
        h, pp = scene_h(t)
        h = E.gblur(E.zoom_about(h, 1 + 1.4 * u, pp[0], pp[1]), 12 * u * E.PX)
        p = scene_pov(t, nature_frame(t - NS))
        p = E.gblur(p, 10 * (1 - u) * E.PX)
        return mix(h, p, u)
    # ---------------- 主观：小猫拍了一下屏幕，山风吹进屋里，推进手机 = 走进山野
    if t < PUSH[1]:
        return scene_pov(t, nature_frame(t - NS))
    return frame_late(t - NS)

def frame_late(t):
    """自然段及之后（旧时间轴）"""
    if t < 93.6:
        img = nature_frame(t)
        if t > 92.0:   # 山脊化作天际线后，推进回到早高峰车厢（拉长到 1.6 秒，更柔）
            u = ease((t - 92.0) / 1.6)
            img = mix(E.zoom_about(img, 1 + 0.25 * u, 720, 470), scene_b(t, warm=1.0), u)
        return img
    # ---------------- R：周一的车厢（暖）
    if t < 101.4:
        img = scene_b(t, warm=1.0)
        return notes_layer(img, t, NOTES_R, fade=1 - ease((t - 97.5) / 1.5))
    # ---------------- R → E：推进车窗，窗外是那晚的营地
    if t < 104.4:
        u = (t - 101.4) / 3.0
        z = 10.0 ** eio(u); cxp, cyp = 840, 320
        e_img = scene_e(t)
        x0 = (640 - cxp) * z + 720; x1 = (1040 - cxp) * z + 720; y0 = (250 - cyp) * z + 540; y1 = (520 - cyp) * z + 540
        pimg = E.portal(e_img, x0, y0, x1, y1)
        img = scene_b(t, warm=1.0, cam=(z, cxp, cyp), portal=pimg, beam=1 - u)
        return mix(img, e_img, ease((u - 0.5) / 0.35))
    return scene_e(t)

def scene_e(t):
    P = tod_at(0, force="night")
    img, info = nature(t, tod=P, cam=(1945, 120, 1.06), fire=0.95, hero=(1800.0, "situp", 0.0, False), catst=(1748.0, "loaf", 0.0, False, 0.0))
    img = _meteor(img, t, [(111.0, 1200, 140, -420, 180, 1.0)])
    # 标题
    a1 = ease((t - 106.2) / 1.6); a2 = ease((t - 107.2) / 1.8)
    fade = 1 - ease((t - 114.6) / 1.4)
    if a1 > 0:
        img = draw_text(img, "露营，", 720, 300, 96, E.C((1.0, 0.95, 0.86)), a1 * fade, glow=0.35, blur=(1 - a1) * 8, rise=(1 - a1) * 14, spacing=6)
    if a2 > 0:
        img = draw_text(img, "是最小成本的出逃", 720, 410, 54, E.C((1.0, 0.92, 0.80)), a2 * fade, glow=0.25, reveal=a2, spacing=10)
    return img * (0.25 + 0.75 * fade) if fade < 1 else img

def render_frame(fidx):
    t = fidx / FPS
    img = frame(t, fidx)
    g = grade_of(t)
    img = img * ease(t / 0.8)
    lift = (0.014 * g, 0.012 * g + 0.008 * (1 - g), 0.006 * g + 0.020 * (1 - g))
    # 轻动画调色：暗角更轻、饱和略高、对比更柔；不加颗粒
    img = E.post(img, t, fidx, bloom=0.30 + 0.10 * g, bloom_th=0.72 + 0.12 * g, vig=0.20, lift=lift, sat=0.86 + 0.26 * g, expo=1.0)
    if t > DUR - 1.0: img = img * (1 - ease((t - (DUR - 1.0)) / 1.0))
    img = E.letterbox(img, BAR)
    return img

# ======================================================================= 音效合成 + 混音（无旁白）
def synth_sfx():
    sr = E.SR; n = int(DUR * sr); out = np.zeros((n, 2), np.float32)
    g = np.random.default_rng(7)
    tt = np.arange(n) / sr
    O = NS                                    # 自然段及之后：旧时间 + O
    def env(a, b, fi=0.5, fo=0.5):
        return np.clip((tt - a) / fi, 0, 1) * np.clip((b - tt) / fo, 0, 1)
    try:
        from scipy.signal import butter, sosfilt
        def bp(x, lo, hi): return sosfilt(butter(2, [lo, hi], "bandpass", fs=sr, output="sos"), x).astype(np.float32)
        def lpf(x, f): return sosfilt(butter(2, f, "lowpass", fs=sr, output="sos"), x).astype(np.float32)
        def hpf(x, f): return sosfilt(butter(2, f, "highpass", fs=sr, output="sos"), x).astype(np.float32)
    except ImportError:
        def _fft_filter(x, lo, hi):
            Xf = np.fft.rfft(x); fr = np.fft.rfftfreq(len(x), 1 / sr)
            Xf[(fr < lo) | (fr > hi)] = 0; return np.fft.irfft(Xf, len(x)).astype(np.float32)
        def bp(x, lo, hi): return _fft_filter(x, lo, hi)
        def lpf(x, f): return _fft_filter(x, 0, f)
        def hpf(x, f): return _fft_filter(x, f, sr / 2)
    def noise(): return g.standard_normal(n).astype(np.float32)
    def put(sig, t0, gain=1.0, pan=0.0):
        i0 = int(t0 * sr); i1 = min(n, i0 + len(sig))
        if i1 <= i0 or i0 < 0: return
        s = sig[:i1 - i0] * gain
        out[i0:i1, 0] += s * (1 - max(0, pan)); out[i0:i1, 1] += s * (1 + min(0, pan))
    def bed(sig, e, gain, roll=0):
        out[:, 0] += sig * e * gain; out[:, 1] += (np.roll(sig, roll) if roll else sig) * e * gain
    def tone(f, d, dec=6.0, harm=(1.0,)):
        k = np.arange(int(d * sr)) / sr
        s = sum(a * np.sin(2 * np.pi * f * (i + 1) * k) for i, a in enumerate(harm))
        return (s * np.exp(-k * dec)).astype(np.float32)
    def sadd(*xs):
        L = max(len(x) for x in xs); o = np.zeros(L, np.float32)
        for x in xs: o[:len(x)] += x
        return o
    def whoosh(d):
        k = int(d * sr); x = g.standard_normal(k).astype(np.float32)
        return (bp(x, 300, 3000) * np.sin(np.linspace(0, np.pi, k)) ** 2).astype(np.float32)
    def burst(d, lo, hi, dec):
        k = int(d * sr); return bp(g.standard_normal(k).astype(np.float32), lo, hi) * np.exp(-np.arange(k) / sr * dec)
    # ---------- 城市（0–21.9）
    hum = lpf(noise(), 180) * 0.9
    bed(hum, env(0, 25.5, 0.5, 2.0) + env(93 + O, 104 + O, 1.0, 2.0) * 0.6, 0.45)
    traffic = bp(noise(), 200, 1200) * (0.6 + 0.4 * np.sin(tt * 0.3))
    bed(traffic, env(0, 7.5, 0.5, 1.0), 0.09)
    rumble = lpf(noise(), 120) * 2.5
    bed(rumble, env(0.5, 7.2, 1.0, 0.6) * (1 - 0.6 * np.clip((tt - 4.0) / 1.0, 0, 1)), 0.3)
    k = np.arange(int(1.2 * sr)) / sr
    put((np.sin(2 * np.pi * (2400 + 300 * np.sin(k * 7)) * k) * np.sin(np.pi * k / 1.2) ** 2 * 0.04).astype(np.float32), 3.7, 1.0, 0.2)
    put(tone(880, 0.6, 4, (1, 0.3)), 5.2, 0.10); put(tone(660, 0.9, 3, (1, 0.3)), 5.55, 0.10)
    put(whoosh(1.4), 5.9, 0.2)
    sub = lpf(noise(), 160) * 2.0 + bp(noise(), 300, 900) * 0.3
    for a, b, gn in ((7.2, 14.2, 0.28), (93.0 + O, 102.0 + O, 0.2)):
        bed(sub, env(a, b, 0.4, 0.6), gn)
        tk = a + 0.3
        while tk < b - 0.3:
            clk = burst(0.05, 400, 2500, 80); put(clk, tk, 0.3 * gn * 3); put(clk, tk + 0.16, 0.22 * gn * 3); tk += 0.92
    for (t0, *_r) in NOTES_B: put(sadd(tone(1320, 0.25, 14, (1, 0.25)), tone(1980, 0.2, 18) * 0.5), t0, 0.06, (hash1(int(t0 * 10), 2) - 0.5))
    put(whoosh(1.2), 13.4, 0.25)
    hm = (np.sin(2 * np.pi * 60 * tt) * 0.02 + bp(noise(), 2000, 6000) * 0.02).astype(np.float32)
    bed(hm, env(14, 21, 0.3, 0.4), 1.0)
    tk = 14.6
    while tk < 20.2:
        put(burst(0.02, 1500, 6000, 300), tk, 0.09 * g.uniform(0.6, 1.0), g.uniform(-0.3, 0.3)); tk += g.uniform(0.06, 0.22) if g.uniform() < 0.85 else 0.6
    for (t0, _) in MSGS: put(tone(1567, 0.22, 16, (1, 0.2)), t0, 0.05, 0.15)
    tick = 14.0; per = 0.5
    while tick < 20.4:
        put(burst(600 / sr, 2500, 7000, 400), tick, 0.1, 0.5); tick += per; per = max(0.07, per * 0.93)
    put(lpf(g.standard_normal(int(0.08 * sr)).astype(np.float32), 800) * np.exp(-np.arange(int(0.08 * sr)) / sr * 50), 20.38, 0.35)
    # ---------- 雨夜街道 → 回家（雨声进屋后变闷）
    rain = bp(noise(), 1500, 9000) * 0.25 + lpf(noise(), 400) * 0.15
    bed(rain, env(20.6, 28.6, 1.2, 1.6), 0.5, 3000)
    rain_in = lpf(noise(), 900) * 0.35
    bed(rain_in, env(27.6, 36.5, 1.0, 2.5), 0.35, 2000)
    step_t = 21.6
    while step_t < 25.5:
        st = lpf(g.standard_normal(int(0.09 * sr)).astype(np.float32), 900) * np.exp(-np.arange(int(0.09 * sr)) / sr * 40)
        st += burst(0.09, 2000, 7000, 60) * 0.4
        put(st, step_t, 0.2); step_t += 0.4
    put(lpf(g.standard_normal(int(0.25 * sr)).astype(np.float32), 300) * np.exp(-np.arange(int(0.25 * sr)) / sr * 18), 25.75, 0.5)   # 关门
    put(burst(0.012, 2000, 6000, 500), 26.0, 0.25)                                                                              # 开灯
    room = lpf(noise(), 250) * 0.4
    bed(room, env(28.0, PUSH[0] + 0.5, 0.8, 1.0), 0.25)
    # 小猫：落地的轻响、小碎步、咕噜咕噜
    put(lpf(g.standard_normal(int(0.06 * sr)).astype(np.float32), 600) * np.exp(-np.arange(int(0.06 * sr)) / sr * 60), 30.1, 0.25, -0.3)
    tp_ = 30.25
    while tp_ < 31.8: put(burst(0.03, 600, 2500, 120), tp_, 0.05, -0.2 + (tp_ - 30.2) * 0.2); tp_ += 0.21
    put(lpf(g.standard_normal(int(0.08 * sr)).astype(np.float32), 500) * np.exp(-np.arange(int(0.08 * sr)) / sr * 45), 32.25, 0.2)
    purr = (lpf(noise(), 180) * (0.55 + 0.45 * np.sin(2 * np.pi * 24 * tt)) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.45 * tt) ** 2)).astype(np.float32)
    bed(purr, env(32.3, PUSH[0] + 1.0, 0.6, 1.2), 0.5)
    put(sadd(tone(1760, 0.18, 22, (1, 0.3)), tone(2637, 0.12, 30) * 0.4), TAP_T, 0.08)                                          # 轻点屏幕
    # 推进手机 → 自然：上扬的空气声 + 一声清亮的和弦
    k = int(4.2 * sr); sw = g.standard_normal(k).astype(np.float32)
    sw = bp(sw, 500, 5000) * (np.linspace(0, 1, k) ** 2) * np.exp(-np.maximum(0, np.arange(k) / sr - 3.6) * 6)
    put(sw.astype(np.float32), TAP_T + 0.2, 0.16)
    put(sadd(tone(523, 3.0, 0.9, (1, 0.4, 0.15)), tone(784, 3.0, 1.0, (1, 0.3)) * 0.7), PUSH[1] + 0.2, 0.06)
    # ---------- 自然（旧时间 + O）
    wnd = bp(noise(), 150, 1400)
    xs_ = np.linspace(-O, DUR, 2400)
    genv = np.interp(tt - O, xs_, np.array([gust(x) for x in xs_])).astype(np.float32)
    lfo = (0.65 + 0.35 * np.sin(tt * 0.5) * np.sin(tt * 0.23 + 1)).astype(np.float32)
    nenv = env(TAP_T, 93.5 + O, 1.5, 1.5) * np.clip((tt - TAP_T) / 4, 0.25, 1) + env(103.5 + O, DUR, 1.5, 1.0) * 0.5
    out[:, 0] += wnd * genv * lfo * nenv * 0.38; out[:, 1] += np.roll(wnd, 2400) * genv * lfo * nenv * 0.38
    def chirp():
        d = g.uniform(0.06, 0.16); k = np.arange(int(d * sr)) / sr
        f0 = g.uniform(2800, 4200); f = f0 + g.uniform(-1500, 1500) * k / d
        return (np.sin(2 * np.pi * np.cumsum(f) / sr) * np.sin(np.pi * k / d) ** 2).astype(np.float32)
    for a, b, dens in ((34.5 + O, 56.0 + O, 0.9), (86.5 + O, 93.5 + O, 1.4)):
        tb = a
        while tb < b:
            pan = g.uniform(-0.8, 0.8)
            for j in range(g.integers(2, 6)): put(chirp(), tb + j * g.uniform(0.09, 0.16), 0.045 * g.uniform(0.4, 1.0), pan)
            tb += g.uniform(0.8, 2.6) / dens
    lap = lpf(noise(), 500) * (0.5 + 0.5 * np.sin(tt * 1.3)) ** 3
    bed(lap, env(50 + O, 66 + O, 2, 3), 0.22)
    for t0 in (46.4, 47.3, 48.2):
        k = int(0.45 * sr); z_ = bp(g.standard_normal(k).astype(np.float32), 1500, 6000) * (0.5 + 0.5 * np.sign(np.sin(np.arange(k) / sr * 2 * np.pi * 38)))
        put(z_ * np.sin(np.linspace(0, np.pi, k)), t0 + O, 0.06)
    bed(bp(noise(), 3000, 9000), env(47.8 + O, 61.5 + O, 0.4, 1.5), 0.022)
    cr = (np.sin(2 * np.pi * 4600 * tt) * (np.sin(2 * np.pi * 28 * tt) > 0.3) * (np.sin(2 * np.pi * 1.6 * tt) > 0)).astype(np.float32)
    cr2 = (np.sin(2 * np.pi * 5200 * tt) * (np.sin(2 * np.pi * 33 * tt) > 0.4) * (np.sin(2 * np.pi * 1.1 * tt + 1) > 0.2)).astype(np.float32)
    cenv = env(64 + O, 87 + O, 3, 2) + env(103 + O, DUR, 1, 1)
    out[:, 0] += cr * cenv * 0.010 + cr2 * cenv * 0.005; out[:, 1] += cr2 * cenv * 0.010 + cr * cenv * 0.005
    fenv = env(66 + O, 87 + O, 1.5, 2.0) + env(103 + O, DUR, 0.5, 1.0)
    bed(lpf(noise(), 300) * 0.5, fenv, 0.11)
    tc = 66.0 + O
    while tc < DUR:
        if fenv[min(n - 1, int(tc * sr))] > 0.05:
            k = int(g.uniform(0.004, 0.02) * sr)
            put(hpf(g.standard_normal(k).astype(np.float32), 1200) * np.exp(-np.arange(k) / sr * 250), tc, 0.1 * g.uniform(0.3, 1.0) * fenv[int(tc * sr)], g.uniform(-0.4, 0.4))
        tc += g.exponential(0.12)
    # 夜里的小猫咕噜（靠着主人）
    bed(purr, env(58.5 + O, 66.0 + O, 1.0, 1.5) * 0.6 + env(104.5 + O, DUR, 1.0, 1.0) * 0.4, 0.35)
    for t0 in (79.8, 82.6, 111.0):
        put(sadd(tone(2637, 1.2, 3, (1, 0.5)) * 0.5, tone(3520, 1.0, 4) * 0.3), t0 + O, 0.04, -0.3)
    put(whoosh(1.6), 92.0 + O, 0.25)
    put(whoosh(2.4), 101.6 + O, 0.2)
    put(sadd(tone(392, 4.0, 0.8, (1, 0.5, 0.2)), tone(587, 4.0, 0.9, (1, 0.3)) * 0.8, tone(784, 3.5, 1.2) * 0.4), 106.2 + O, 0.07)
    return out

def mix_audio(video_path, out_path):
    print("合成音效…")
    sfx = synth_sfx(); n = len(sfx); sr = E.SR
    tt = np.arange(n) / sr
    # BGM：Windswept（吉他 + 弦乐渐强）。自动对齐，让曲子自然的结尾落在片尾
    bgm = E.load_audio(BGM_FILE) if os.path.exists(BGM_FILE) else np.zeros((0, 2), np.float32)
    if len(bgm):
        off_env = os.environ.get("EP_BGM_OFFSET")
        tail = len(bgm) / sr
        off = float(off_env) if off_env else max(0.0, tail - DUR - 0.6)
        bgm = bgm[int(off * sr):]
        if len(bgm) < n: bgm = np.concatenate([bgm, np.zeros((n - len(bgm), 2), np.float32)])
        bgm = bgm[:n]
    else:
        bgm = np.zeros((n, 2), np.float32)
    O = NS
    lvl = np.interp(tt, [0, 2.5, 21.0, 26.0, 29.0, TAP_T, PUSH[1], 90 + O, 93 + O, 101 + O, 104 + O, DUR - 2.5, DUR],
                    [0.0, 0.40, 0.40, 0.32, 0.52, 0.55, 0.85, 0.85, 0.55, 0.55, 0.85, 0.85, 0.0]).astype(np.float32)
    mix_ = bgm * lvl[:, None] + sfx * 0.85
    wav = "mix_tmp.wav"; E.save_wav(wav, mix_ * 0.8)
    print("响度标准化并封装…")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", video_path, "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-af", "loudnorm=I=-14:TP=-1.5:LRA=11", "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-shortest", out_path], check=True)
    os.remove(wav)

# ======================================================================= 渲染入口
def render_range(f0, f1, path):
    w = E.open_writer(path)
    t0 = time.perf_counter()
    window_t0 = t0
    window_f0 = f0
    try:
        for f in range(f0, f1):
            img = render_frame(f)
            w.stdin.write(E.to_u8(img).tobytes())
            if (f - f0 + 1) % 60 == 0 or f == f1 - 1:
                now = time.perf_counter()
                el = now - t0; done = f - f0 + 1
                instant_fps = (f - window_f0 + 1) / max(1e-6, now - window_t0)
                eta = el / done * (f1 - f0 - done)
                print(f"\r  帧 {f + 1}/{f1}  最近60帧 {instant_fps:.2f} fps  累计 {done / el:.2f} fps  剩余约 {eta / 60:.1f} 分钟   ", end="", flush=True)
                window_t0 = now
                window_f0 = f + 1
    finally:
        w.stdin.close(); w.wait()
    print(f"\n  完成 {path}，用时 {(time.perf_counter() - t0) / 60:.1f} 分钟")

def save_png(t, path):
    import cv2
    img = render_frame(int(round(t * FPS)))
    cv2.imwrite(path, cv2.cvtColor(E.to_u8(img), cv2.COLOR_RGB2BGR))

def main():
    global torch
    dev = E.init(); torch = E.torch
    print(f"设备：{dev}  分辨率：{E.W}x{E.H}  时长：{DUR}s")
    if E.DEV.type == "cuda":
        print("  GPU：", torch.cuda.get_device_name(0), f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "full"
    if cmd == "bench":
        ts = [3, 10, 17, 24, 27, 30, 34, 37, 46, 64, 78, 88, 103, 115]
        render_frame(int(40 * FPS))
        if E.DEV.type == "cuda": torch.cuda.synchronize()
        t0 = time.time()
        for t in ts:
            render_frame(int(t * FPS)); 
        if E.DEV.type == "cuda": torch.cuda.synchronize()
        sp = (time.time() - t0) / len(ts)
        print(f"平均 {sp:.3f} 秒/帧 → 全片约 {sp * DUR * FPS / 60:.1f} 分钟（另加编码）")
        if E.DEV.type == "cuda": print(f"显存峰值 {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
    elif cmd in ("pv", "gpu_pv"):
        for s in sys.argv[2:]:
            p = f"pv_{s}.png"; save_png(float(s), p); print("  ", p)
    elif cmd in ("seg", "gpu_seg"):
        f0, f1 = int(sys.argv[2]), int(sys.argv[3]); out = sys.argv[4] if len(sys.argv) > 4 else f"seg_{f0}_{f1}.mp4"
        render_range(f0, f1, out)
    elif cmd == "audio":
        mix_audio(sys.argv[2] if len(sys.argv) > 2 else "video_tmp.mp4", OUT_NAME)
    elif cmd == "full":
        print("1/2 渲染画面…"); render_range(0, int(DUR * FPS), "video_tmp.mp4")
        print("2/2 音效与混音…"); mix_audio("video_tmp.mp4", OUT_NAME)
        print("完成 →", OUT_NAME)
    else:
        print(__doc__)

if __name__ == "__main__":
    main()
