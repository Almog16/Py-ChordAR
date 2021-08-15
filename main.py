import os
import glob
from pathlib import Path

from utils.guitar_image import GuitarImage

if __name__ == '__main__':
    Emaj_chord = "x,7,6,4,5,4"

    for filename in filter(lambda x: 'gitkeep' not in x, os.listdir(Path(r"C:\Users\almogsh\PycharmProjects\Py_ChordAR\photos"))):
        guitar = GuitarImage(
            img_path=Path(rf"C:\Users\almogsh\PycharmProjects\Py_ChordAR\photos\{filename}"))  # , file_name=filename)
        guitar.get_chord_coordinates(Emaj_chord)
    # guitar = GuitarImage(img_path=Path(
    #     rf"C:\Users\almogsh\PycharmProjects\Py_ChordAR\photos\working\guitar5_good.jfif"))  # , file_name=r"1_.jpg")
    # guitar.get_chord_coordinates(Emaj_chord)
