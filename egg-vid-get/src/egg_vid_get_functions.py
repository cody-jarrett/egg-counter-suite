import csv
import time
import cv2 as cv


class Timer(object):

    def __init__(self, precision, quiet=False):
        self._precision = precision
        self._quiet = quiet

    def start(self):
        self._start = time.time()

    def stop(self):
        self._stop = time.time()
        total = round((self._stop - self._start), self._precision)
        if self._quiet:
            pass
        else:
            print('Time: {:f}'.format(total))

    def elapsed(self):
        total = round((time.time() - self._start), self._precision)
        return str(total)


def access_csv(file_str, data, usage_str):
    with open(file_str, usage_str, newline='') as file:
        writer = csv.writer(file, delimiter=',')
        for el in data:
            writer.writerow(el)


def create_video_writer(file_name, width=1280, height=720, fps=30.0):
    """ If have issues with OpenCV writer, could possibly use skvideo writer"""
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    writer = cv.VideoWriter(file_name, fourcc, fps, (width, height))
    return writer


def load_image(image_str):
    image = cv.imread(image_str, 0)
    return image


def subtract_images(image_1, image_2, blur=True):
    subtracted_image = cv.subtract(image_2, image_1)
    if blur:
        subtracted_image = cv.GaussianBlur(subtracted_image, (5, 5), 0)
    return subtracted_image
