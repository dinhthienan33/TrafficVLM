import argparse
import os
import shutil
import subprocess
from typing import List


def list_avi_files(input_folder: str, recursive: bool) -> List[str]:
    if not os.path.isdir(input_folder):
        return []

    avi_files: List[str] = []
    if recursive:
        for root, _, files in os.walk(input_folder):
            for name in files:
                if name.lower().endswith(".avi"):
                    avi_files.append(os.path.join(root, name))
    else:
        for name in os.listdir(input_folder):
            full_path = os.path.join(input_folder, name)
            if os.path.isfile(full_path) and name.lower().endswith(".avi"):
                avi_files.append(full_path)

    avi_files.sort()
    return avi_files


def target_mp4_path(src_avi: str, output_dir: str) -> str:
    base_name = os.path.splitext(os.path.basename(src_avi))[0]
    return os.path.join(output_dir, f"{base_name}.mp4")


def convert_one(src_avi: str, dst_mp4: str, overwrite: bool, crf: int, preset: str) -> None:
    os.makedirs(os.path.dirname(dst_mp4), exist_ok=True)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        src_avi,
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        dst_mp4,
    ]
    subprocess.run(cmd, check=True)


def run(args: argparse.Namespace) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not installed or not in PATH.")

    if not os.path.isdir(args.input_folder):
        raise RuntimeError(f"Input folder does not exist: {args.input_folder}")

    src_files = list_avi_files(args.input_folder, args.recursive)
    if len(src_files) == 0:
        raise RuntimeError(f"No .avi files found in: {args.input_folder}")

    output_dir = args.output_dir if args.output_dir else args.input_folder
    if output_dir == "":
        output_dir = "."

    failed = 0
    for idx, src_avi in enumerate(src_files, start=1):
        dst_mp4 = target_mp4_path(src_avi, output_dir)
        try:
            convert_one(src_avi, dst_mp4, args.overwrite, args.crf, args.preset)
            print(f"[{idx}/{len(src_files)}] OK  {src_avi} -> {dst_mp4}")
        except subprocess.CalledProcessError as exc:
            failed += 1
            print(f"[{idx}/{len(src_files)}] FAIL {src_avi}: {exc}")

    success = len(src_files) - failed
    print(f"Done. Success: {success}, Failed: {failed}, Total: {len(src_files)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert .avi video(s) to .mp4 using ffmpeg.")
    parser.add_argument("--input-folder", type=str, required=True, help="Directory containing .avi files.")
    parser.add_argument("--output-dir", type=str, default="", help="Destination directory for .mp4 files.")
    parser.add_argument("--recursive", action="store_true", help="Scan subdirectories in input folder.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .mp4 files.")
    parser.add_argument("--crf", type=int, default=23, help="H.264 quality (lower means better quality).")
    parser.add_argument(
        "--preset",
        type=str,
        default="medium",
        choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"],
        help="H.264 encoding speed/efficiency preset.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
