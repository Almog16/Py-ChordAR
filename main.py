import os
import glob
from pathlib import Path

import cv2
from matplotlib import pyplot as plt

from error_handling.fingers_hiding_neck import FingersHidingNeckError
from error_handling.not_enough_strings_detected import NotEnoughStringsDetected
from utils.guitar_image import GuitarImage

if __name__ == '__main__':
    Emaj_chord = "x,7,6,4,5,4"

    for filename in filter(lambda x: 'gitkeep' not in x and not Path(x).is_file(), os.listdir(Path(r"C:\Users\almogsh\PycharmProjects\Py_ChordAR\photos\asd\ella"))):
        try:
            guitar = GuitarImage(
            img_path=Path(rf"C:\Users\almogsh\PycharmProjects\Py_ChordAR\photos\asd\ella\{filename}"))  # , file_name=filename)
            # guitar.cropped.plot_img()
            guitar.get_chord_coordinates(Emaj_chord)
            # guitar.save_img(step=f"{guitar.step}_draw_chords", i=guitar.i)
            # GuitarImage.i += 1
            guitar.plot_img()
            print(filename + " SUCCESS")
        except FingersHidingNeckError:
            print("Fingers are hiding the guitar's neck")
        except NotEnoughStringsDetected:
            print("Not enough strings detected")
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
