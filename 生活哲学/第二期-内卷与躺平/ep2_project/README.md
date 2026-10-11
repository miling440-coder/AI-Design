# 第二期《内卷与躺平 · 西西弗》工程

输出：1440×1080（4:3）、30fps、160 秒，H.264 + AAC，响度约 -14 LUFS。

## 准备
1. 安装 Python 3.9+ 和 ffmpeg（需带 libx264）
2. `pip install -r requirements.txt`
3. `bash get_font.sh`（下载思源宋体 Light，保存为 font.otf）
4. 把背景音乐《千年之后（氛围节奏版）》放进本目录，命名为 `bgm.mp3`

## 生成
- 完整视频：`python make_video_ep2_4x3.py` → 输出 `ep2_neijuan_4x3.mp4`
- 预览单帧：`python make_video_ep2_4x3.py pv 17.2 64 104.5` → 输出 `e2_<秒>.png`

## 修改
- 文案和时间轴：脚本里 `# ---- ACT 1/2/3` 三段，`B(开始, 结束, [(文字, 样式)], ...)` 是文字块，`K("类型", 开始, 结束, ...)` 是特效
- 本期新增特效：spiral 螺旋卷入、iris 圆形遮罩、grid 满屏文字塌落、counter 时间滚动、chart 增长曲线、stone 推石上山、morph 粒子变形换句、lie_at 文字躺倒，所有文字块带模糊入场/出场
- 节拍按 112 BPM（0.5315 秒）卡点，换音乐时改 `BEAT` 并重新对时间轴
