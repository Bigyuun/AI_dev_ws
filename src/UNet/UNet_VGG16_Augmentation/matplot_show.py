import matplotlib.pyplot as plt
from glob import glob
import cv2
import os


if __name__ == "__main__":
    plt.figure()
    img_result = sorted(glob(os.path.join("./results", "*.png")))
    for i, filename in enumerate(img_result):
        img = cv2.imread(filename)
        ax = plt.subplot(5, 1, i+1)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if i is 4:
            break
    plt.show()