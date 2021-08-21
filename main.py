import os
import glob
from pathlib import Path

from utils.guitar_image import GuitarImage

if __name__ == '__main__':
    Emaj_chord = "x,7,6,4,5,4"

    for filename in filter(lambda x: 'gitkeep' not in x and not Path(x).is_file(), os.listdir(Path(r"C:\Users\almogsh\PycharmProjects\Py_ChordAR\photos\asd"))):
        try:
            guitar = GuitarImage(
            img_path=Path(rf"C:\Users\almogsh\PycharmProjects\Py_ChordAR\photos\asd\{filename}"))  # , file_name=filename)
            guitar.get_chord_coordinates(Emaj_chord)
        except Exception as e:
            print(rf"{filename} : {e}")
    # guitar = GuitarImage(img_path=Path(
    #     rf"C:\Users\almogsh\PycharmProjects\Py_ChordAR\photos\guitar5_FLIPPED.jfif"))  # , file_name=r"1_.jpg")
    # guitar.get_chord_coordinates(Emaj_chord)
