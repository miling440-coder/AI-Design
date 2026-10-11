# 《虚无主义》抖音竖屏视频 · 生成工程

纯代码生成：Python（Pillow + NumPy）逐帧绘制动画，ffmpeg 编码并混音。
输出 1080×1920、30fps、160 秒 MP4，响度约 -14 LUFS。

## 环境
- Python 3.9+
- ffmpeg（需在 PATH 中，含 libx264）
- `pip install -r requirements.txt`

## 准备素材（放在本目录）
1. `font.otf`：运行 `./get_font.sh` 下载思源宋体 Light（也可换成任意中文字体，改名为 font.otf）
2. `bgm.mp3`：背景音乐《千年之后（氛围节奏版）》（版权原因未打包，放入你自己的音频文件即可）

## 运行
```bash
python3 make_video.py            # 完整渲染，生成 nihilism_douyin.mp4（双核约 5 分钟）
python3 make_video.py pv 1.5 60  # 只预览指定秒数的单帧，输出 p3_<秒>.png
```
完整渲染时还会生成：
- `sfx3.wav`：自动合成的音效轨（风声转场、打字声、坠落低频、粒子微光）
- `storyboard3.json`：按时间轴导出的分镜表

## 怎么改
- **文案与时间轴**：`make_video.py` 中「script」一节。每条 `B(开始秒, 结束秒, [(文字, 样式), ...], 参数)` 是一屏文字。
  - 样式：`main` 正文 / `big` 大字 / `sub` 注释 / `q` 引文 / `small` 小字
  - 参数：`per` 逐字间隔，`line_gap` 行间延迟，`type_=True` 打字机，`fall_at=秒` 文字坠落，
    `glitch_at=秒` 故障撕碎，`uline=(行号, 秒)` 暖色下划线，`strike=(行号, 秒)` 删除线，
    `wipe_at=秒` 光线擦除，`pop=True` 大字弹入，`track` 字距，`drift` 镜头漂移
- **特效段**：`K("类型", 开始, 结束, ...)`，类型有 `dot` 白点、`title` 标题、`words` 词语快闪、
  `press` 边框挤压、`list` 清单划掉并坠落、`stars` 星空穿梭（`warp_at` 加速）、`collapse` 画面缩成点、
  `ring` 光环扩散、`particles` 粒子聚合成句、`acts` 暖色清单
- **卡点**：开始时间会自动吸附到 `BEAT = 0.5315` 秒（约 112 BPM）的节拍网格；换音乐时改这个值
- **配色/字号**：`BG`、`FG`、`DIM`、`WARM` 和 `STY`
- **画幅**：改 `W, H`（横屏 1920×1080 时同步调整 `TY` 文字中心）
- **音量**：ffmpeg 命令里 `[2:a]volume=0.45` 为音效占比

## 4:3 横屏版
`python3 make_video_4x3.py` 生成 1440×1080 的 `nihilism_4x3.mp4`，文案和时间轴与竖屏版一致，
文字中心、星空分布、边框挤压已按横屏重新排版。
