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

## 加速生成（NVIDIA GPU）
- 推荐：`python render_ep2_fast.py --encoder nvenc --workers 2`
- 输出：`ep2_neijuan_4x3_fast.mp4`，不会覆盖原视频；已有同名文件时用 `--output 新文件名.mp4`。
- 短片段验证：`python render_ep2_fast.py --encoder nvenc --workers 2 --start 12 --duration 3 --output test_gpu.mp4`
- 本机测速：`python render_ep2_fast.py --benchmark --workers 2`；可用 `--report benchmark.json` 保存结果。
- 不指定 `--encoder` 时先实际检测 NVENC，不可用则回退 CPU；`--encoder cpu` 可强制使用 libx264。

加速入口直接调用原脚本的绘图函数，用多个 CPU 进程按帧并行绘图，再由 NVIDIA NVENC 编码。Pillow 的字体、模糊和缩放仍在 CPU 上执行；没有迁移到 CUDA。无需安装额外 Python 包或 CUDA Toolkit，FFmpeg 需要支持 `h264_nvenc`，显卡驱动需正常工作。

保留原画面尺寸、帧率、动画、背景颗粒设置、音效和混音流程。NVENC 使用 `p4 / VBR / CQ 24`，保留 `4M` 码率上限；CQ 与原 libx264 的 CRF 不能按相同数字认为画质完全相同。测速使用覆盖全时间轴的采样帧，不包含音效生成与混音，不能当作整片耗时保证。

## 修改
- 文案和时间轴：脚本里 `# ---- ACT 1/2/3` 三段，`B(开始, 结束, [(文字, 样式)], ...)` 是文字块，`K("类型", 开始, 结束, ...)` 是特效
- 本期新增特效：spiral 螺旋卷入、iris 圆形遮罩、grid 满屏文字塌落、counter 时间滚动、chart 增长曲线、stone 推石上山、morph 粒子变形换句、lie_at 文字躺倒，所有文字块带模糊入场/出场
- 节拍按 112 BPM（0.5315 秒）卡点，换音乐时改 `BEAT` 并重新对时间轴
