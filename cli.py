import argparse
from pathlib import Path


def script_parser() -> tuple[Path, Path]:
    """process arg from cli and returns Path for video and output file"""
    ap = argparse.ArgumentParser(
        prog="Bars Poster",
        description="Generate a poster composed of cropped frames from a video file",
    )
    ap.add_argument(
        "video",
        metavar="video",
        help="Path to video file",
        type=Path,
    )
    args = ap.parse_args()
    video_path = args.video
    output_filename = video_path.parent.joinpath("barcode.jpg")
    return video_path, output_filename


def question_change() -> bool:
    """determine if user wants to change a frame"""
    while True:
        question_change = input("Change any frame? (y/n)")
        if question_change == "y":
            return True
        elif question_change == "n":
            return False


def question_frame(frame_positions_list: list[int]) -> list[int]:
    """determine which frames a user wants to change"""
    while True:
        try:
            question = input(
                f"Which frame numbers? (list position) {frame_positions_list}"
            )
            position = list(map(int, question.split()))
            return position
        except ValueError:
            print("the positions should be integers separated by spaces")
