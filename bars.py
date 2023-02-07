from collections.abc import Generator
from pathlib import Path
import cv2
import numpy as np
from cli import question_change, question_frame, script_parser
from contextlib import contextmanager
import logging

# ####### PARAMETERS ########### #
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


def get_video_width_height_frames(
    video_capture: cv2.VideoCapture,
) -> tuple[int, int, int]:
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return width, height, total_frames


def show_image(image: cv2.Mat) -> None:
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def generate_barcode(video_path: Path, output_filename: Path) -> None:
    video_path_str = video_path.as_posix()
    with video_capture(video_path_str) as cap:
        width, height, total_frames = get_video_width_height_frames(cap)
        nbr_of_captured_frames = np.arange(1, number_of_cuts + 1)
        logging.info(nbr_of_captured_frames)
        logging.info(
            f"Video width: {width}, height: {height}, Nbr frames:{total_frames}"
        )
        fr_pos = int(
            total_frames / (len(nbr_of_captured_frames) + 2)
        )  # always keep 2 block frames at beginning & end
        frame_positions_list = [i * fr_pos for i in nbr_of_captured_frames]
        logging.info(frame_positions_list)
        images = frame_position_to_image(frame_positions_list, cap)
        palette = image_in_column(images)
        palette = resize_frame(palette, 50)
        show_image(palette)

        while question_change():
            positions = question_frame(frame_positions_list)
            modify_frame_list(frame_positions_list, positions)
            images = frame_position_to_image(frame_positions_list, cap)
            palette = image_in_column(images)
            show_image(palette)

        avg_cols = ROI_capture_manual(
            images, cap, nbr_of_captured_frames, width, height
        )
        concatenated = np.concatenate(avg_cols, axis=1)
        barcode = cv2.resize(concatenated, (width, height))
        cv2.imwrite(output_filename.as_posix(), barcode)
        cv2.destroyAllWindows()
        show_image(barcode)


def resize_frame(frame_name: cv2.Mat, scale_percent: int) -> cv2.Mat:
    """resize a frame given a wished percentage. Keeps original aspect ratio"""
    width = int(frame_name.shape[1] * scale_percent / 100)
    height = int(frame_name.shape[0] * scale_percent / 100)
    return cv2.resize(frame_name, (width, height))


def frame_position_to_image(
    frame_pos: list[int], cap: cv2.VideoCapture
) -> list[cv2.Mat]:
    """transform a list of frame positions into opencv images"""
    frame_images = []
    for i in frame_pos:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        _, frame_image = cap.read()
        frame_images.append(frame_image)
    return frame_images


def image_in_column(images: list[cv2.Mat]) -> cv2.Mat:
    """organize a list of opencv images in column"""
    palette = []
    palette = np.concatenate(images, axis=0)
    return palette


def modify_frame_list(list: list[int], position: list[int]) -> list[int]:
    """Change frame numbers by adding 100 to change image captured in the video"""
    for i in position:
        list[i - 1] += 100
    logging.info(list)
    return list


def ROI_capture_manual(
    image_list, cap: cv2.VideoCapture, nbr_of_captured_frames, width: int, height: int
):
    avg_cols = []
    for im in image_list:
        """decrease image size by 50% to display on screen"""
        frame = resize_frame(im, 50)
        r = cv2.selectROI(
            windowName="Select ROI", img=frame, showCrosshair=False, fromCenter=False
        )
        output = frame.copy()
        rectangle = cv2.rectangle(
            frame,
            (r[0] + int(width / nbr_of_captured_frames[-1]) - 5, 0),
            (r[0] + int(width / nbr_of_captured_frames[-1]), height),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(rectangle, 1, output, 0, 0, output)

        avg_cols.append(
            output[
                0:height,
                r[0] : r[0] + int(width / nbr_of_captured_frames[-1]),
            ]
        )
    return avg_cols


def main() -> None:
    video_path, output_filename = script_parser()
    generate_barcode(video_path, output_filename)


if __name__ == "__main__":
    main()
