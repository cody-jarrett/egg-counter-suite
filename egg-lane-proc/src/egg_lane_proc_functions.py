import os
import csv


def access_csv(file_str, data, usage_str):
    with open(file_str, usage_str, newline='') as file:
        writer = csv.writer(file, delimiter=',')
        for el in data:
            writer.writerow(el)


def is_obj_in_lane(x, lanes, i):
    x1 = lanes['x1'][i]
    x2 = lanes['x2'][i]
    x3 = lanes['x3'][i]
    x4 = lanes['x4'][i]
    if (x > x1 and x < x4) or (x > x2 and x < x3):
        return True
