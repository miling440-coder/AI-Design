《内卷与躺平》v2 GPU 渲染版

一、安装（只做一次）
1. 安装 ffmpeg 并加入 PATH（命令行输入 ffmpeg -version 能看到版本即可）
2. 安装 Python 依赖：
   pip install numpy opencv-python-headless pillow
3. 安装带 CUDA 的 PyTorch（NVIDIA 显卡）：
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   验证：python -c "import torch; print(torch.cuda.is_available())"  → 输出 True 才算成功

二、文件放在同一个文件夹
   engine.py  scene_ep2.py  font.otf  bgm.mp3

三、运行
   先测速度：   python scene_ep2.py bench
   看预览图：   python scene_ep2.py gpu_pv 16.6 56.2 64 150
   一键出片：   python scene_ep2.py full
   → 自动渲染、生成音效、混音并做响度标准化，输出 ep2_neijuan_v2_4x3.mp4

四、说明
- 自动选设备：NVIDIA 用 CUDA，Mac 用 MPS，都没有就用 CPU
- 强制指定设备：Windows 先运行 set EP_DEVICE=cuda；Mac/Linux 在命令前加 EP_DEVICE=cuda
- NVIDIA 显卡会自动用 h264_nvenc 硬件编码，失败自动改用 libx264
- 显存占用约 300–600MB，4GB 显存足够
- 原来的 CPU 命令（pv / sheet / seg / audio）都保留，画面与 GPU 版逐像素一致
