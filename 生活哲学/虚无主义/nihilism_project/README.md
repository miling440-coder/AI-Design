# 《虚无主义》4:3 视频生成工程

运行入口：`make_video_4x3.py`。Python（Pillow + NumPy）逐帧绘制画面，ffmpeg 编码并混入背景音乐和音效。

成片规格：**1440×1080（4:3）、30fps、160 秒**。

## 运行条件

- Python，以及 `requirements.txt` 中的 NumPy、Pillow。
- ffmpeg 已加入 PATH，并支持 libx264；校验视频时使用 ffprobe。
- 项目目录中有 `font.otf`（思源宋体 Light）和 `bgm.mp3`。

当前环境的依赖和素材已齐全，无需重复安装或下载。2026-10-11 实际运行使用 Python 3.12.9、NumPy 2.2.3、Pillow 11.1.0、ffmpeg 6.1.1。

## Windows PowerShell 运行

```powershell
Set-Location -LiteralPath 'F:\AI Video\生活哲学\虚无主义\nihilism_project'
$env:PYTHONUTF8 = '1'
python -u make_video_4x3.py
```

`PYTHONUTF8` 确保中文分镜文件使用 UTF-8 编码，`-u` 让运行输出及时显示。本次完整渲染约 11 分钟，耗时随机器性能变化。结束时输出 `done 0`。

## 输出文件

所有输出位于项目目录：

- `nihilism_4x3.mp4`：4:3 成片。
- `sfx4.wav`：脚本合成的音效轨。
- `storyboard4.json`：分镜时间轴。

## 成片校验

在项目目录中执行：

```powershell
ffprobe -v error -show_entries format=duration:stream=codec_name,width,height,avg_frame_rate,nb_frames -of json nihilism_4x3.mp4
ffmpeg -v error -xerror -i nihilism_4x3.mp4 -f null -
```

2026-10-11 已确认：视频为 H.264、1440×1080、30fps、4800 帧、160 秒，音频为 AAC；音视频完整解码通过，未发现错误。
