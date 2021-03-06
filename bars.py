import os
import time
import argparse
import cv2
import numpy as np

# ####### PARAMETERS ########### #
OUT_WIDTH = 600
OUT_HEIGHT = 800
number_of_cuts = 5
# ############################## #


def generate_barcode():
    cap = cv2.VideoCapture(VIDEO)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    nbr_of_captured_frames = np.arange(1, number_of_cuts + 1)
    fr_pos = int(total_frames / (len(nbr_of_captured_frames) + 2))  # always keep 2 block frames at beginning & end
    frame_positions_list = [i*fr_pos for i in nbr_of_captured_frames]
    print(frame_positions_list)

    avg_cols = []
    cap.release()

    def resize_frame(frame_name, scale_percent):
        width = int(frame_name.shape[1] * scale_percent / 100)
        height = int(frame_name.shape[0] * scale_percent / 100)
        dim = (width, height)
        frame_resized = cv2.resize(frame_name, dim)
        return frame_resized

    def frame_show(frame_list):
        frame_column = []
        palette = []
        for i in frame_list:
            cap = cv2.VideoCapture(VIDEO)
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            res, frame = cap.read()
            frame_column.append(resize_frame(frame, 10))
            palette = np.concatenate(frame_column, axis=0)
        cv2.imshow("Image", palette)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def question():
        answer = input("Change any frame? (y/n) ")
        if answer == "y":
            position = list(map(int, input("Which frame number? (list position) {0} ".format(frame_positions_list)).split()))
            modify_frame_list(frame_positions_list, position)
            frame_show(frame_positions_list)
            question()
        elif answer == "n":
            print("OK, let's a go")
        else:
            print("incorrect value")
            question()

    def modify_frame_list(list, position):
        for i in position:
            list[i-1] += 100
        print(list)
        return list

    frame_show(frame_positions_list)
    question()
    avg_cols = []
    cap.release()

    def ROI_capture_manual(frame_list):
        for i in frame_list:
            cap = cv2.VideoCapture(VIDEO)
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            res, frame = cap.read()
            frame_height = frame.shape[0]
            frame = resize_frame(frame, 50)
            r = cv2.selectROI(frame, False, False)
            output = frame.copy()
            rectangle = cv2.rectangle(frame, (r[0] + int(OUT_WIDTH / nbr_of_captured_frames[-1]) - 5, 0), (r[0] + int(OUT_WIDTH / nbr_of_captured_frames[-1]), frame_height), (0, 0, 0), -1)
            cv2.addWeighted(rectangle, 1, output, 0, 0, output)
            avg_cols.append(output[0:frame_height, r[0]:r[0] + int(OUT_WIDTH / nbr_of_captured_frames[-1])])
            cap.release()
        return avg_cols

    ROI_capture_manual(frame_positions_list)
    concatenated = np.concatenate(avg_cols, axis=1)
    print("Resizing {} frames to {}".format(concatenated.shape[1], OUT_WIDTH))
    barcode = cv2.resize(concatenated, (OUT_WIDTH, OUT_HEIGHT))
    cv2.imwrite(OUT_NAME, barcode)
    cv2.destroyAllWindows()
    cv2.imshow("Result", barcode)
    cv2.waitKey(0)


if __name__ == "__main__":
    start_time = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("-video", help="Path to video file", required=True)
    args = vars(ap.parse_args())
    VIDEO = args["video"]
    OUT_NAME = os.path.splitext(os.path.basename(VIDEO))[0] + "_barcode.jpg"

    generate_barcode()

    elapsed_time = time.time() - start_time
    print("Time elapsed: {}"
        .format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
