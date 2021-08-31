import os
import glob
from pathlib import Path

import cv2
from matplotlib import pyplot as plt

from utils.guitar_image import GuitarImage

if __name__ == '__main__':
    Emaj_chord = "2,4,4,2,2,2"

    for filename in filter(lambda x: 'gitkeep' not in x and not Path(x).is_file(), os.listdir(Path(r"C:\Users\Ella\Documents\Ella Studies\year3\ChordAR\images"))):
        try:

            # img_path=Path(rf"C:\Users\almogsh\PycharmProjects\Py_ChordAR\photos\asd\ella\{filename}"))  # , file_name=filename)
            # guitar.cropped.plot_img()
            # guitar.get_chord_coordinates(Emaj_chord)
            # guitar.plot_img()
            guitar = GuitarImage(
            img_path=Path(rf"C:\Users\Ella\Documents\Ella Studies\year3\ChordAR\images\{filename}")) # , file_name=filename)
            # guitar.cropped.plot_img()
            guitar.get_chord_coordinates(Emaj_chord)
            cv2.imshow("", cv2.cvtColor(guitar.color_img, cv2.COLOR_BGR2RGB))
            cv2.waitKey()
            # guitar.flipped.plot_img()
        except Exception as e:
            print(rf"{filename} : {e}")
    # guitar = GuitarImage(img_path=Path(
    #     rf"C:\Users\almogsh\PycharmProjects\Py_ChordAR\photos\asd\ella\02.jpg"))  # , file_name=r"1_.jpg")
    # guitar.get_chord_coordinates(Emaj_chord)
    # guitar.cropped.plot_img()
    # guitar.plot_img()
    # # cv2.imshow("enhanced", guitar.enhanced_color)
    # # cv2.imshow("gray", guitar.gray)
    # plt.show()
    # # cv2.waitKey()
