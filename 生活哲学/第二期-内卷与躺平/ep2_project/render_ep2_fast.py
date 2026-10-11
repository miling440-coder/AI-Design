"""Reuse the original frame renderer, with bounded parallel drawing and NVENC."""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

PROJECT = Path(__file__).resolve().parent
SOURCE = PROJECT / "make_video_ep2_4x3.py"
_composition = None


def load_composition():
    global _composition
    if _composition is None:
        # The original script resolves font.otf relative to its working directory.
        os.chdir(PROJECT)
        spec = importlib.util.spec_from_file_location("ep2_composition", SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _composition = module
    return _composition


def draw_frame(timestamp):
    return load_composition().draw(timestamp).tobytes()


def frame_bytes(timestamps, workers):
    if workers == 1:
        for timestamp in timestamps:
            yield draw_frame(timestamp)
        return
    # Each worker initializes its own seeded particles and font cache. Bound the
    # queue: RGB frames are 4.67 MB each; never enqueue the entire 160s film.
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=load_composition,
    ) as pool:
        timestamps = iter(timestamps)
        pending = deque()
        for _ in range(workers * 2):
            timestamp = next(timestamps, None)
            if timestamp is None:
                break
            pending.append(pool.submit(draw_frame, timestamp))
        while pending:
            # Submission order preserves the exact video timeline.
            yield pending.popleft().result()
            timestamp = next(timestamps, None)
            if timestamp is not None:
                pending.append(pool.submit(draw_frame, timestamp))


def encoder_options(encoder):
    if encoder == "nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-tune", "hq",
                "-rc", "vbr", "-cq", "24", "-b:v", "0"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "24", "-tune", "grain"]


def nvenc_available(ffmpeg):
    # Listed encoders alone do not prove that the installed GPU/driver can run it.
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-v", "error", "-f", "lavfi", "-i",
         "color=size=1440x1080:rate=30", "-frames:v", "1", "-an",
         *encoder_options("nvenc"), "-pix_fmt", "yuv420p", "-f", "null", "-"],
        capture_output=True, text=True, timeout=30,
    )
    return result.returncode == 0, result.stderr.strip()


def encode(ffmpeg, timestamps, output, encoder, workers, audio=False, start=0.0):
    composition = load_composition()
    count = len(timestamps)
    command = [ffmpeg, "-n", "-hide_banner", "-v", "error", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-s", f"{composition.W}x{composition.H}",
               "-r", str(composition.FPS), "-i", "-"]
    if audio:
        fade_at = max(0.0, composition.DUR - 2.5 - start)
        command += ["-ss", str(start), "-i", str(PROJECT / "bgm.mp3"),
                    "-ss", str(start), "-i", str(PROJECT / "sfx_ep2.wav"),
                    "-filter_complex",
                    "[1:a]volume=1.0[m];[2:a]volume=0.45[s];"
                    "[m][s]amix=inputs=2:normalize=0,"
                    f"afade=t=out:st={fade_at}:d=2.5,loudnorm=I=-14:TP=-1:LRA=11[a]",
                    "-map", "0:v", "-map", "[a]", "-c:a", "aac", "-b:a", "192k"]
    else:
        command += ["-an"]
    command += [*encoder_options(encoder), "-maxrate", "4M", "-bufsize", "8M",
                "-pix_fmt", "yuv420p", "-t", str(count / composition.FPS),
                "-movflags", "+faststart", str(output)]
    begun = time.perf_counter()
    wait_for_pipe = 0.0
    next_update = begun + 3
    frames = frame_bytes(timestamps, workers)
    # A file avoids stderr pipe deadlocks and retains the error on a failed run.
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors)
        try:
            for index, data in enumerate(frames, 1):
                write_start = time.perf_counter()
                process.stdin.write(data)
                wait_for_pipe += time.perf_counter() - write_start
                now = time.perf_counter()
                if now >= next_update:
                    fps = index / (now - begun)
                    print(f"{index}/{count} frames | {fps:.1f} fps | "
                          f"ETA {(count-index)/fps:.0f}s", flush=True)
                    next_update = now + 3
            process.stdin.close()
            code = process.wait()
        except BaseException:
            process.kill()
            process.wait()
            errors.seek(0)
            detail = errors.read().decode("utf-8", errors="replace").strip()
            if detail:
                print(detail, flush=True)
            raise
        finally:
            if not process.stdin.closed:
                try:
                    process.stdin.close()
                except OSError:
                    pass
            frames.close()
        if code:
            errors.seek(0)
            raise RuntimeError(errors.read().decode("utf-8", errors="replace").strip())
    elapsed = time.perf_counter() - begun
    return {"encoder": encoder, "workers": workers, "frames": count,
            "seconds": round(elapsed, 3), "fps": round(count / elapsed, 2),
            "pipe_write_seconds": round(wait_for_pipe, 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=["auto", "nvenc", "cpu"], default="auto")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--output", type=Path, default=PROJECT / "ep2_neijuan_4x3_fast.mp4")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-frames", type=int, default=120)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    # Resolve command-line paths before the original renderer changes cwd.
    args.output = args.output.resolve()
    if args.report:
        args.report = args.report.resolve()
    if args.workers < 1 or args.workers > 8:
        parser.error("--workers must be between 1 and 8 (try 2 or 3 on this computer)")
    if args.benchmark_frames < 16:
        parser.error("--benchmark-frames must be at least 16")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        parser.error("ffmpeg was not found in PATH")
    for name in ["font.otf"] + ([] if args.benchmark else ["bgm.mp3"]):
        if not (PROJECT / name).is_file():
            parser.error(f"Missing {PROJECT / name}")
    composition = load_composition()
    supported, detail = nvenc_available(ffmpeg) if args.encoder != "cpu" else (False, "")
    if args.encoder == "nvenc" and not supported:
        parser.error(f"NVENC initialization failed: {detail}")
    encoder = "nvenc" if supported else "cpu"
    if args.encoder == "auto" and not supported:
        print(f"NVENC unavailable; using CPU encoding. {detail}", flush=True)
    if args.benchmark:
        # Consecutive bursts across the timeline cover text, transitions and particles.
        timestamps = [(i // 8) * (composition.DUR - 1) /
                      math.ceil(args.benchmark_frames / 8) + (i % 8) / composition.FPS
                      for i in range(args.benchmark_frames)]
        cases = [("cpu", 1)]
        if supported:
            cases.append(("nvenc", 1))
        if args.workers > 1:
            cases.append((encoder, args.workers))
        results = []
        with tempfile.TemporaryDirectory(prefix="ep2_benchmark_") as directory:
            for chosen, workers in cases:
                print(f"Benchmark: {chosen}, workers={workers}", flush=True)
                result = encode(ffmpeg, timestamps,
                                Path(directory) / f"{chosen}_{workers}.mp4", chosen, workers)
                results.append(result)
                print(json.dumps(result), flush=True)
        if args.report:
            args.report.resolve().write_text(json.dumps(results, indent=2), encoding="utf-8")
        return
    duration = composition.DUR - args.start if args.duration is None else args.duration
    if args.start < 0 or args.start >= composition.DUR or duration <= 0:
        parser.error("Use a valid --start and positive --duration")
    count = max(1, round(min(duration, composition.DUR - args.start) * composition.FPS))
    output = args.output.resolve()
    if output.exists():
        parser.error(f"Output exists; choose another --output: {output}")
    if not output.parent.is_dir():
        parser.error(f"Output directory does not exist: {output.parent}")
    composition.storyboard()
    composition.build_audio()
    print(f"Encoding: {encoder}, drawing workers: {args.workers}, frames: {count}", flush=True)
    result = encode(ffmpeg, [args.start + i / composition.FPS for i in range(count)],
                    output, encoder, args.workers, audio=True, start=args.start)
    print(json.dumps(result), flush=True)
    print(f"Saved: {output}", flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
