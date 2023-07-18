# Mark processing within detected region

import numpy as np
import cv2
from matplotlib import pyplot as plt
from utils import golden_section_search, ternary_search, fill_out_circle, calc_diff


class MarkOnImage:
    def __init__(self, x_pos = 255., y_pos = 255., diam = 255., rot_angle = 0.):
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.diam = diam
        self.rot_angle = rot_angle


    def get_center(self):
        return (self.x_pos, self.y_pos)


    def get_array_param(self):
        return [self.x_pos, self.y_pos, self.diam, self.rot_angle]
    

def create_np_template(diam = 256, rot_angle = 0, linewidth=None, channels=1, filename = None):
    if not linewidth:
        linewidth = diam * 0.015
    image = np.zeros((diam, diam))
    c = (np.array(image.shape) -1 )* 0.5
    rad = image.shape[0]* 0.5

    i = 0
    while i <= c[0]:
        j = 0
        while j <= i:
            if (rad * rad)*0.25 < ((i - c[0])**2 + (j - c[1])**2) <= (rad * rad):
                image[i][j] = 255
                image[j][i] = 255
                image[diam-i-1][diam-j-1] = 255
                image[diam-j-1][diam-i-1] = 255
            j += 1
        i += 1
    i = image.shape[0] - 1

    i = 0
    while i <= c[0]:
        j = image.shape[1] - 1
        while i + j >= diam - 1:
            if ((i - c[0])**2 + (j - c[1])**2) <= (rad * rad) * 0.25:
                image[i][j] = 255
                image[j][i] = 255
                image[diam-i-1][diam-j-1] = 255
                image[diam-j-1][diam-i-1] = 255
            j -= 1
        i += 1

    i = image.shape[0] - 1

    (h, w) = image.shape[:2]

    # rotate our image by 45 degrees around the center of the image
    M = cv2.getRotationMatrix2D(c, 90 + rot_angle, 1.0)
    res = 255 - cv2.warpAffine(image, M, (w, h))   
    
    inner = (rad - linewidth)**2
    outer = rad**2
    i = 0
    while i <= c[0]:
        j = 0
        while j <= i:
            if inner < ((i - c[0])**2 + (j - c[1])**2) <= outer:
                res[i][j] = 0
                res[j][i] = 0
                res[i][diam-j-1] = 0
                res[j][diam-i-1] = 0
                res[diam-i-1][j] = 0
                res[diam-j-1][i] = 0
                res[diam-i-1][diam-j-1] = 0
                res[diam-j-1][diam-i-1] = 0
            j += 1
        i += 1

    res = res.astype(int)

    if channels == 3:
        res = np.moveaxis(np.stack([res, res, res]), 0, -1)
    
    if filename:
        cv2.imwrite(filename, res)

    return res


def create_np_template_old(diam = 256, rot_angle = 0, filename = None, channels = 1):
    image = np.zeros((diam, diam))
    c = (np.array(image.shape) -1 )* 0.5
    rad = image.shape[0]* 0.5

    i = 0
    while i <= c[0]:
        j = 0
        while j <= c[1]:
            if (rad * rad)*0.25 < ((i - c[0])**2 + (j - c[1])**2) <= (rad * rad):
                image[i][j] = 255
            j += 1
        i += 1
    i = image.shape[0] - 1
    while i >= c[0]:
        j = image.shape[1] - 1
        while j >= c[1]:
            if (rad * rad)*0.25 < ((i - c[0])**2 + (j - c[1])**2) <= (rad * rad):
                image[i][j] = 255
            j -= 1
        i -= 1

    i = 0
    while i <= c[0]:
        j = image.shape[1] - 1
        while j >= c[1]:
            if ((i - c[0])**2 + (j - c[1])**2) <= (rad * rad) * 0.25:
                image[i][j] = 255
            j -= 1
        i += 1

    i = image.shape[0] - 1
    while i >= c[0]:
        j = 0
        while j <= c[1]:
            if ((i - c[0])**2 + (j - c[1])**2) <= (rad * rad) * 0.25:
                image[i][j] = 255
            j += 1
        i -= 1
    (h, w) = image.shape[:2]

    # rotate our image by 45 degrees around the center of the image
    M = cv2.getRotationMatrix2D(c, rot_angle, 1.0)
    res = cv2.warpAffine(image, M, (w, h))
    if channels == 3:
        return np.moveaxis(np.stack([res, res, res]), 0, -1)
    else:
        return res


def detect_mark(image, center, rad, plt_vis = False):

    # Удаление большей части фона
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    center_int = np.array(center).astype(int)
    rad_int = int(rad)
    # print(center_int, rad_int)
    # print(center_int[1] - rad_int,center_int[1] + rad_int, center_int[0] - rad,center_int[0] + rad_int)
    mark_part = image_gray[center_int[1] - rad_int:center_int[1] + rad_int, center_int[0] - rad_int:center_int[0] + rad_int]
    # try float?
    label_pos_x, label_pos_y = center[0] - rad, center[1] - rad
    label_pos_x_int, label_pos_y_int = center_int[0] - rad_int, center_int[1] - rad_int

    mark_part = fill_out_circle(mark_part)
    if plt_vis:
        plt.imshow(mark_part)
        plt.title('Удаление большей части фона')
        plt.show()

    # Распознование контрастных точек
    ret, thresh = cv2.threshold(mark_part, 170, 255, 0)
    thresh = 255 - thresh
    if plt_vis:
        plt.imshow(thresh)
        plt.title('Распознование контрастных точек')
        plt.show()
    # print(np.unique(thresh))
    contours_image, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Распознавание контура круглой метки
    part_area = mark_part.shape[0]*mark_part.shape[1]
    min_area_lim = part_area / 12
    max_area_lim = part_area * 0.9
    best_cntr = contours_image[0]
    best_cntr_area = cv2.contourArea(best_cntr)
    for cntr in contours_image:
        area = cv2.contourArea(cntr)
        if area > min_area_lim and area < max_area_lim:
            if (area > best_cntr_area):
                best_cntr = cntr
                best_cntr_area = area
            cntr_np = np.array(cntr).reshape(-1, 2)

    if plt_vis:
        fig, ax = plt.subplots(1,3, figsize=(18,6))
        # ax[0].imshow(mark_part)
        ax[0].imshow(mark_part)
        ax[1].imshow(np.zeros_like(mark_part))
        ax[2].imshow(np.zeros_like(mark_part))
        best_cntr_np = np.array(best_cntr).reshape(-1, 2)
        ax[0].plot(best_cntr_np[:,0], best_cntr_np[:,1])
        ax[1].plot(best_cntr_np[:,0], best_cntr_np[:,1])
        ax[2].fill(best_cntr_np[:,0], best_cntr_np[:,1])
        fig.suptitle('Распознавание расположения круглой метки')
        plt.show()

    # Квадрат, описывающий метку
    x,y,w,h = cv2.boundingRect(best_cntr)
    # try float?
    wh_mean = (w + h) / 2.
    dx = (w - wh_mean) / 2.
    dy = (h - wh_mean) / 2.

    wh_mean_int, dx_int, dy_int = map(int,(wh_mean, dx, dy))
    
    if plt_vis:
        plt.imshow(mark_part[y + dy_int:y + dy_int + wh_mean_int, x + dx_int:x + dx_int + wh_mean_int])
        plt.title('Найденная метка')
        plt.show()

    # Координаты угла метки
    # try float?
    label_pos_x += x + dx
    label_pos_y += y + dy
    label_pos_x_int, label_pos_y_int = int(label_pos_x), int(label_pos_y)
    if plt_vis:
        image_copy = image.copy()
        cv2.rectangle(image_copy, (label_pos_x_int, label_pos_y_int), (label_pos_x_int + wh_mean_int, label_pos_y_int + wh_mean_int), (36,255,12), 2)
        plt.imshow(image_copy)

    # Вырез метки с картинки
    detected_template = image_gray[label_pos_y_int:label_pos_y_int + wh_mean_int,label_pos_x_int:label_pos_x_int + wh_mean_int]
    # надо нормировать
    ret, thresh = cv2.threshold(detected_template, 170, 255, 0)

    # Задаем интервал значений и точность
    min_val = 0.
    mean_val = 90.
    max_val = 180.
    eps = 1e-6

    # Вызываем быстрый поиск для функции func полученной от calc_diff
    func = lambda grad_val, thr: calc_diff(thr,create_np_template(diam=wh_mean_int,rot_angle=grad_val))
    result_1 = ternary_search(func, min_val, mean_val, eps, thresh)
    result_2 = ternary_search(func, mean_val, max_val, eps, thresh)
    # print("Значение угла, при котором достигается минимум ошибки в первой половине:", result_1)
    val_1 = func(result_1, thresh)
    # print("Минимальное значение ошибки:", val_1)
    # print("Значение угла, при котором достигается минимум ошибки во второй половине:", result_2)
    val_2 = func(result_2, thresh)
    # print("Минимальное значение ошибки:", val_2)
    result = result_1 if val_1 <= val_2 else result_2
    val = min(val_1, val_2)
    # print('______________________')
    # print("Значение угла, при котором достигается минимум ошибки:", result)
    # print("Минимальное значение ошибки:", val)
    most_acc_template = create_np_template(diam=wh_mean_int,rot_angle=result)
    # print(f'Координаты метки: ({label_pos_y},{label_pos_x})')
    # print(f'Угол поворота метки: {result}')
    if plt_vis:
        image_copy = image.copy()
        image_copy[label_pos_y_int:label_pos_y_int + wh_mean_int,label_pos_x_int:label_pos_x_int + wh_mean_int, 0] = 255
        image_copy[label_pos_y_int:label_pos_y_int + wh_mean_int,label_pos_x_int:label_pos_x_int + wh_mean_int, 1] = most_acc_template
        image_copy[label_pos_y_int:label_pos_y_int + wh_mean_int,label_pos_x_int:label_pos_x_int + wh_mean_int, 2] = 255
        plt.imshow(image_copy)
        plt.title(f'Координаты метки: ({label_pos_y + wh_mean//2}, {label_pos_x + wh_mean//2})\nУгол поворота метки: {result}')
        plt.show()
    return MarkOnImage(label_pos_x + wh_mean/2, label_pos_y + wh_mean/2, wh_mean, result)