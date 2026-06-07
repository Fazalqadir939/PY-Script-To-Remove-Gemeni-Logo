import cv2
import numpy as np
import argparse
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Supported video formats
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}

def default_region(width: int, height: int) -> tuple[int, int, int, int]:
    sx = int(1115 * width / 1280)
    sy = int(535  * height / 720)
    sw = int(100  * width / 1280)
    sh = int(115  * height / 720)
    return sx, sy, sw, sh

def build_mask(cap: cv2.VideoCapture, total: int, x: int, y: int, w: int, h: int, width: int, height: int) -> np.ndarray:
    print("  Building sparkle mask...")
    rois = []
    step = max(1, total // 120)
    
    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if ok:
            rois.append(f[y:y+h, x:x+w].astype(np.float32))
            
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    min_f = np.stack(rois).min(axis=0)
    gray = min_f.mean(axis=2)
    
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    mask = np.maximum((gray - blurred > 8).astype(np.uint8),
                      (gray > 55).astype(np.uint8))
                      
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.dilate(mask, k, iterations=1)

    full_mask = np.zeros((height, width), dtype=np.uint8)
    full_mask[y:y+h, x:x+w] = mask
    
    print(f"  Sparkle pixels: {mask.sum()}  Region: x={x} y={y} w={w} h={h}")
    return full_mask

def check_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False

def process_single_video(input_path: Path, output_path: Path, pos: tuple[int, int, int, int] | None = None):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"[error] Cannot open: {input_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nProcessing: {input_path.name}")
    print(f"  Resolution : {width}x{height}  {fps:.2f}fps  {total} frames")

    x, y, w, h = pos if pos else default_region(width, height)
    full_mask = build_mask(cap, total, x, y, w, h, width, height)

    tmp_dir = Path(tempfile.mkdtemp())
    tmp_out = tmp_dir / "intermediate.mp4"
    
    writer = cv2.VideoWriter(str(tmp_out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        print("[error] VideoWriter failed. Skipping file.")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        cap.release()
        return

    print("  Removing sparkle...")
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        
        clean_frame = cv2.inpaint(frame, full_mask, 7, cv2.INPAINT_TELEA)
        writer.write(clean_frame)
        
        n += 1
        if not HAS_TQDM and n % 60 == 0:
            print(f"    {n}/{total}")

    cap.release()
    writer.release()

    print("  Encoding final output...")
    if check_ffmpeg():
        cmd = [
            "ffmpeg", "-y", "-i", str(tmp_out),
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p", str(output_path)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        shutil.copy2(tmp_out, output_path)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"  Saved Clean Video -> {output_path.name}")

def main():
    ap = argparse.ArgumentParser(description="Remove watermarks from a single video or a batch folder.")
    ap.add_argument("target", type=Path, help="Path to a video file OR a directory containing videos.")
    ap.add_argument("-o", "--output", type=Path, default=None, help="Output file path (or output directory if targeting a folder).")
    ap.add_argument("--pos", nargs=4, type=int, metavar=("X", "Y", "W", "H"), help="Custom region coordinates.")
    args = ap.parse_args()

    if not args.target.exists():
        sys.exit(f"[error] Path does not exist: {args.target}")

    pos_tuple = tuple(args.pos) if args.pos else None

    # Scenario 1: Single file
    if args.target.is_file():
        out_file = args.output if args.output else args.target.with_name(f"{args.target.stem}_clean.mp4")
        process_single_video(args.target, out_file, pos=pos_tuple)

    # Scenario 2: Folder
    elif args.target.is_dir():
        print(f"Scanning folder: {args.target}")
        
        videos_to_process = [
            item for item in args.target.iterdir()
            if item.is_file() 
            and item.name.lower().startswith('t') 
            and item.suffix.lower() in VIDEO_EXTENSIONS
            and not item.stem.endswith('_clean')
        ]

        if not videos_to_process:
            print("No matching video files starting with 't' were found.")
            return

        print(f"Found {len(videos_to_process)} matching video(s) to process.")

        out_dir = args.output if args.output else args.target
        if not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=True)

        for video_path in videos_to_process:
            out_file = out_dir / f"{video_path.stem}_clean.mp4"
            process_single_video(video_path, out_file, pos=pos_tuple)
            
        print("\nAll batch processing tasks completed!")

if __name__ == "__main__":
    main()