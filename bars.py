from collections.abc import Generator
from pathlib import Path
import cv2
import numpy as np
from cli import question_change, question_frame, script_parser
from contextlib import contextmanager
import logging

# ####### PARAMETERS ########### #
OUT_WIDTH = 600
OUT_HEIGHT = 800
number_of_cuts = 5
# ############################## #


logging.basicConfig(level=logging.INFO)


@contextmanager
def video_capture(video_path: str) -> Generator[cv2.VideoCapture, None, None]:
    cap = cv2.VideoCapture(video_path)
    try:
        yield cap
    finally:
        logging.info("VideoCapture is released")
        cap.release()


def show_image(image: cv2.Mat) -> None:
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def generate_barcode(video_path: Path, output_filename: Path) -> None:
    with video_capture(video_path.as_posix()) as cap:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        nbr_of_captured_frames = np.arange(1, number_of_cuts + 1)
        fr_pos = int(
            total_frames / (len(nbr_of_captured_frames) + 2)
        )  # always keep 2 block frames at beginning & end
        frame_positions_list = [i * fr_pos for i in nbr_of_captured_frames]
        logging.info(frame_positions_list)
        frame_show(frame_positions_list, cap)

        while question_change():
            positions = question_frame(frame_positions_list)
            modify_frame_list(frame_positions_list, positions)
            frame_show(frame_positions_list, cap)

        avg_cols = ROI_capture_manual(frame_positions_list, cap, nbr_of_captured_frames)
        concatenated = np.concatenate(avg_cols, axis=1)
        logging.info(
            "Resizing {} frames to {}".format(concatenated.shape[1], OUT_WIDTH)
        )
        barcode = cv2.resize(concatenated, (OUT_WIDTH, OUT_HEIGHT))
        cv2.imwrite(output_filename.as_posix(), barcode)
        show_image(barcode)


def resize_frame(frame_name, scale_percent):
    width = int(frame_name.shape[1] * scale_percent / 100)
    height = int(frame_name.shape[0] * scale_percent / 100)
    dim = (width, height)
    frame_resized = cv2.resize(frame_name, dim)
    return frame_resized


def frame_show(frame_list, cap: cv2.VideoCapture) -> None:
    frame_column = []
    palette = []
    for i in frame_list:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        _, frame = cap.read()
        frame_column.append(resize_frame(frame, 10))
    palette = np.concatenate(frame_column, axis=0)
    show_image(palette)


def modify_frame_list(list: list[int], position: list[int]) -> list[int]:
    for i in position:
        list[i - 1] += 100
    logging.info(list)
    return list


def ROI_capture_manual(frame_list, cap: cv2.VideoCapture, nbr_of_captured_frames):
    avg_cols = []
    for i in frame_list:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        _, frame = cap.read()
        frame_height = frame.shape[0]
        frame = resize_frame(frame, 50)
        r = cv2.selectROI(
            windowName="Select ROI", img=frame, showCrosshair=False, fromCenter=False
        )
        output = frame.copy()
        rectangle = cv2.rectangle(
            frame,
            (r[0] + int(OUT_WIDTH / nbr_of_captured_frames[-1]) - 5, 0),
            (r[0] + int(OUT_WIDTH / nbr_of_captured_frames[-1]), frame_height),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(rectangle, 1, output, 0, 0, output)

        avg_cols.append(
            output[
                0:frame_height,
                r[0] : r[0] + int(OUT_WIDTH / nbr_of_captured_frames[-1]),
            ]
        )
    return avg_cols


def main() -> None:
    video_path, output_filename = script_parser()
    generate_barcode(video_path, output_filename)


if __name__ == "__main__":
    main()
