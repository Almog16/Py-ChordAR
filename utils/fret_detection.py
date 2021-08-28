import math
from itertools import zip_longest
from operator import itemgetter

import cv2
import numpy as np
from matplotlib import pyplot as plt

from utils.image import Image
from utils.image import apply_threshold


def cm_to_pixels(cm: float):
    PIXELS_PER_CENTIMETER = 37.7952755906
    return cm * PIXELS_PER_CENTIMETER


def fret_detection(cropped_neck_img: Image) -> np.array:
    gray = cropped_neck_img.img_to_gray(cropped_neck_img.color_img)
    gray = Image.enhance_gray_image(gray_img=gray)
    # edges = cv2.Sobel(cropped_neck_img.blur_gray, cv2.CV_64F, 1, 0)
    edges = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    # edges = cv2.Canny(image=cropped_neck_img.blur_gray, threshold1=20, threshold2=240)
    # edges = cv2.Canny(image=gray, threshold1=20, threshold2=180)
    edges1 = apply_threshold(img=edges, threshold=90)
    edges2 = apply_threshold(img=edges, threshold=20)
    kernel = np.ones((5, 5), np.uint8)
    closing1 = cv2.morphologyEx(edges1, cv2.MORPH_CLOSE, kernel)
    lines1 = cv2.HoughLinesP(image=closing1.astype(np.uint8), rho=1, theta=np.pi / 180, threshold=18,
                            minLineLength=cropped_neck_img.height * 0.5, maxLineGap=10)
    closing2 = cv2.morphologyEx(edges2, cv2.MORPH_CLOSE, kernel)
    lines2 = cv2.HoughLinesP(image=closing2.astype(np.uint8), rho=1, theta=np.pi / 180, threshold=18,
                             minLineLength=cropped_neck_img.height * 0.1, maxLineGap=10)

    lines1 = [line[0] for line in lines1 if 1.01 * line[0][2] >= line[0][0] >= 0.99 * line[0][2]]
    lines2 = [line[0] for line in lines2 if 1.01 * line[0][2] >= line[0][0] >= 0.99 * line[0][2]]
    lines = lines1 + lines2

    lines = sorted(lines, key=lambda line: line[0])
    lines = np.array(remove_duplicate_vertical_lines(lines=lines, width = cropped_neck_img.width))

    low_ys = np.array([min(line[1], line[3]) for line in lines])
    high_ys = np.array([max(line[1], line[3]) for line in lines])
    avg_low_y = int(np.average(low_ys))
    avg_high_y = int(np.average(high_ys))
    for line in lines:
        x1, x2 = line[0], line[2]
        y1 = line[1] if 1.05 * avg_low_y >= line[1] >= 0.95 * avg_low_y else avg_low_y - 5
        y2 = line[3] if 1.05 * avg_high_y >= line[3] >= 0.95 * avg_high_y else avg_high_y + 5
        cv2.line(img=cropped_neck_img.color_img, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0),
                 thickness=int(cropped_neck_img.width * 0.002))
    return lines


def fret_detection_with_hough_lines(cropped_neck_img: Image) -> np.array:
    gray = cropped_neck_img.img_to_gray(cropped_neck_img.color_img)
    # dst = cv2.Sobel(src=gray, ddepth=cv2.CV_8U, dx=1, dy=0)
    dst = cv2.Canny(image=gray, threshold1=50, threshold2=200, apertureSize=3)
    # kernel = np.ones((5, 5), np.uint8)
    # dst = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel)
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)
    height = cropped_neck_img.height
    width = cropped_neck_img.width
    lines = cv2.HoughLines(image=dst.astype(np.uint8), rho=1, theta=np.pi / 180, threshold=80)
    vertical_lines = []
    horizontal_lines = []
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))

            if pt2[0] - pt1[0] != 0:
                slope = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
            else:
                slope = 100000
            y_axis_intr = pt1[1] - slope * pt1[0]
            if math.fabs(slope) < 1:
                y_in_middle = slope * width / 2 + y_axis_intr
                horizontal_lines.append((slope,
                                         y_axis_intr,
                                         (abs(pt1[0]), abs(pt1[1])),
                                         (abs(pt2[0]),abs(pt2[1])),
                                         y_in_middle))
            else:
                x_in_middle = (height / 2 - y_axis_intr) / slope
                vertical_lines.append((slope,
                                       y_axis_intr,
                                       pt1, #(abs(pt1[0]), abs(pt1[1])),
                                       pt2, #(abs(pt2[0]), abs(pt2[1])),
                                       x_in_middle))

    vertical_lines = [[line[2], line[3]] for line in vertical_lines] #if 1.4 * line[2][0] >= line[3][0] >= 0.6 * line[2][0]]

    lines = sorted(vertical_lines, key=lambda line: line[0][0])
    lines = np.array(remove_duplicate_vertical_lines(lines=lines, width = cropped_neck_img.width))

    for line in lines:
        cv2.line(cropped_neck_img.color_img, line[0], line[1], (0,0,255), 3, cv2.LINE_AA)
        # cv2.imshow(str(line[0]) + " " + str(line[1]), cdst)
        # cv2.waitKey()
    #
    # plt.imshow(cdst)
    # # cv2.imshow("Detected lines - Probabilistic Houh Line Transform", cdstP)
    # plt.show()
    return lines

def calculate_fret_gaps(detected_frets, number_of_frets=19):
    scale_length = 65
    frets = [0]
    magic_constant = 17.817

    detected_frets_pairwise = [
        (t[0][0], t[1][1]) for t in zip(detected_frets[:len(detected_frets)], detected_frets[1:])
    ]
    for i in range(1, number_of_frets):
        next_fret = frets[i-1] + ((scale_length - frets[i-1]) / magic_constant)
        frets.append(next_fret)
    frets.sort(reverse=True)
    for fret, detected_fret in zip_longest(frets, detected_frets_pairwise):
        print(f"{fret} | {detected_fret[1] - detected_fret[0]}")


def remove_duplicate_vertical_lines(lines, width):
    new_lines = []
    thresh = 37 # width * 0.0099
    lines_pairwise = zip(lines[:len(lines)], lines[1:])
    for line1, line2 in lines_pairwise:
        if line2[0][0] - line1[0][0] > thresh or line2[1][0] - line1[1][0] > thresh:
            new_lines.append(line1)
    if lines[-1][0][0] - new_lines[-1][0][0] > thresh or lines[-1][1][0] - new_lines[-1][1][0] > thresh:
        new_lines.append(lines[-1])
    return new_lines
