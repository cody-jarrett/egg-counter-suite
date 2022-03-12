"""
Functions for the egg-vid-get software for the Egg Counter research system

Author: Cody Jarrett
Organization: Phillips Lab, Institute of Ecology and Evolution,
              University of Oregon
"""
import csv
import time
import numpy
import cv2 as cv


class Timer(object):
    def __init__(self, precision: int, quiet=False):
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
            print("Time: {:f}".format(total))

    def elapsed(self):
        total = round((time.time() - self._start), self._precision)
        return str(total)


def access_csv(file_str: str, data: list, usage_str: str):
    """
    Takes a string that represents a file name, a list of lists that represents the data, and a
    string that represents the usage of the file. It then opens the file, writes the data to the file,
    and closes the file

    Args:
      file_str (str): the name of the file you want to write to
      data (list): list of lists
      usage_str (str): "w" means write, "a" means append
    """
    with open(file_str, usage_str, newline="") as file:
        writer = csv.writer(file, delimiter=",")
        for el in data:
            writer.writerow(el)


def create_video_writer(
    file_name: str, width: int = 1280, height: int = 720, fps: float = 30.0
) -> cv.VideoWriter:
    """
    Create a video writer

    Args:
      file_name (str): name of the output video file
      width (int): width of the output video in pixels
      height (int): height of the output video in pixels
      fps (float): frames per second of the written video

    Returns:
      A cv.VideoWriter object
    """
    # If have issues with OpenCV writer, could possibly use skvideo writer
    fourcc = cv.VideoWriter_fourcc(*"XVID")
    writer = cv.VideoWriter(file_name, fourcc, fps, (width, height))
    return writer


def load_image(image_str: str) -> numpy.ndarray:
    """
    Loads an image from a file path and returns it as a numpy array

    Args:
      image_str (str): path to the image file

    Returns:
      A numpy array of the image
    """
    image = cv.imread(image_str, 0)
    return image


def subtract_images(
    image_1: numpy.ndarray, image_2: numpy.ndarray, blur: bool = True
) -> numpy.ndarray:
    """
    Subtracts image_2 from image_1. Optionally applies a Gaussian blur to the
    subtracted image

    Args:
      image_1 (numpy.ndarray): first image to subtract from
      image_2 (numpy.ndarray): image that will be subtracted from the first image
      blur (bool): if True, the subtracted image will be blurred using a Gaussian blur

    Returns:
      The subtracted image
    """
    subtracted_image = cv.subtract(image_2, image_1)
    if blur:
        subtracted_image = cv.GaussianBlur(subtracted_image, (5, 5), 0)
    return subtracted_image
