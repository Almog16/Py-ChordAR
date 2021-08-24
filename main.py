import os
import glob
from pathlib import Path

import cv2
from matplotlib import pyplot as plt

from utils.guitar_image import GuitarImage

if __name__ == '__main__':
    Emaj_chord = "x,7,6,4,5,4"

    for filename in filter(lambda x: 'gitkeep' not in x and not Path(x).is_file(), os.listdir(Path(r"C:\Users\almogsh\PycharmProjects\Py_ChordAR\photos\asd"))):
        try:
            guitar = GuitarImage(
            img_path=Path(rf"C:\Users\almogsh\PycharmProjects\Py_ChordAR\photos\asd\{filename}"))  # , file_name=filename)
            # guitar.cropped.plot_img()
            guitar.get_chord_coordinates(Emaj_chord)
            cv2.imshow("", cv2.cvtColor(guitar.cropped.color_img, cv2.COLOR_BGR2RGB))
            cv2.waitKey()
            # guitar.flipped.plot_img()
        except Exception as e:
            print(rf"{filename} : {e}")
    # guitar = GuitarImage(img_path=Path(
    #     rf"C:\Users\almogsh\PycharmProjects\Py_ChordAR\photos\asd\1629816972577.jpg"))  # , file_name=r"1_.jpg")
    # guitar.get_chord_coordinates(Emaj_chord)
    # guitar.flipped.plot_img()
    # cv2.imshow("enhanced", guitar.enhanced_color)
    # cv2.imshow("gray", guitar.gray)
    # plt.show()
    # cv2.waitKey()
