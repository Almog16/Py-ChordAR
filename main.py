from pathlib import Path
import matplotlib.pyplot as plt

from utils.fret_detection import fret_detection
from utils.image import Image
from utils.rotate_and_crop_neck import crop_neck
from utils.string_detection import string_detection

if __name__ == '__main__':
    guitar = Image(str(Path(r"C:\Users\Almog\Documents\College\Sadna\opencv_practice\photos\guitar5.jfif")))
    # guitar.plot_img()
    cropped = crop_neck(guitar)
    # cropped.plot_img()
    fret_lines = fret_detection(cropped)
    string_detection(cropped_neck_img=cropped, fret_lines=fret_lines)
    plt.show()
