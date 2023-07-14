# Mark processing within detected region

import numpy as np
import cv2
from matplotlib import pyplot as plt
from utils import golden_section_search, fill_out_circle, calc_diff


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
    

def create_np_template(diam = 256, rot_angle = 0, filename = None):
    # Создание черно-белого изображения
    np_template = np.ones((diam, diam), dtype=np.uint8)*255

    # Определение цветов для секторов
    colors = [255, 0]  # белый, чёрный

    # Определение размеров секторов
    outer_radius = diam // 2
    inner_radius = diam // 4

    # Определение угловых градусов для секторов
    angles = np.array([0, 90, 180, 270]) - rot_angle

    # Рисование чередующихся секторов
    for i, angle in enumerate(angles):
        start_angle = angle
        end_angle = angle + 90
        color = colors[i % 2]
        cv2.ellipse(np_template, (outer_radius, outer_radius), (outer_radius, outer_radius), 0, start_angle, end_angle, color, -1)

    # Рисование внутренних секторов
    for i, angle in enumerate(angles):
        start_angle = angle
        end_angle = angle + 90
        color = colors[(i + 1) % 2]
        cv2.ellipse(np_template, (outer_radius, outer_radius), (inner_radius, inner_radius), 0, start_angle, end_angle, color, -1)

    # Преобразование в черно-белую картинку-массив numpy
    return np.array(np_template, dtype=np.uint8)


def detect_mark(image, center, rad, plt_vis = False):

    # Удаление большей части фона
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mark_part = image_gray[center[1] - rad:center[1] + rad, center[0] - rad:center[0] + rad]
    label_pos_x, label_pos_y = center[0] - rad, center[1] - rad

    mark_part = fill_out_circle(mark_part)
    if plt_vis:
        plt.imshow(mark_part)
        plt.title('Удаление большей части фона')
        plt.show()

    # Распознование контрастных точек
    ret, thresh = cv2.threshold(mark_part, 170, 255, 0)
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    thresh = 255 - thresh
    contours_image, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if plt_vis:
        plt.imshow(thresh)
        plt.title('Распознование контрастных точек')
        plt.show()

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
    wh_mean = (w + h) // 2
    dx = (w - wh_mean) // 2
    dy = (h - wh_mean) // 2
    if plt_vis:
        plt.imshow(mark_part[y + dy:y + dy + wh_mean, x + dx:x + dx + wh_mean])
        plt.title('Найденная метка')
        plt.show()

    # Координаты угла метки
    label_pos_x += x + dx
    label_pos_y += y + dy
    if plt_vis:
        image_copy = image.copy()
        cv2.rectangle(image_copy, (label_pos_x, label_pos_y), (label_pos_x + wh_mean, label_pos_y + wh_mean), (36,255,12), 2)
        plt.imshow(image_copy)
    # cv2.imshow('Image', image_copy)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # Вырез метки с картинки
    detected_template = image_gray[label_pos_y:label_pos_y + wh_mean,label_pos_x:label_pos_x + wh_mean]
    ret, thresh = cv2.threshold(detected_template, 170, 255, 0)

    # Задаем интервал значений и точность
    min_val = 0
    mean_val = 90
    max_val = 180
    eps = 0.001

    # Вызываем быстрый поиск для функции func полученной от calc_diff
    func = lambda grad_val, thr: calc_diff(thr,create_np_template(diam=wh_mean,rot_angle=grad_val))
    result_1 = golden_section_search(func, min_val, mean_val, eps, thresh)
    result_2 = golden_section_search(func, mean_val, max_val, eps, thresh)
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
    most_acc_template = create_np_template(diam=wh_mean,rot_angle=result)
    # print(f'Координаты метки: ({label_pos_y},{label_pos_x})')
    # print(f'Угол поворота метки: {result}')
    if plt_vis:
        image_copy = image.copy()
        image_copy[label_pos_y:label_pos_y + wh_mean,label_pos_x:label_pos_x + wh_mean, 0] = 255
        image_copy[label_pos_y:label_pos_y + wh_mean,label_pos_x:label_pos_x + wh_mean, 1] = most_acc_template
        image_copy[label_pos_y:label_pos_y + wh_mean,label_pos_x:label_pos_x + wh_mean, 2] = 255
        # cv2.imshow('Result',image_copy)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        plt.imshow(image_copy)
        plt.title(f'Координаты метки: ({label_pos_y + wh_mean//2}, {label_pos_x + wh_mean//2})\nУгол поворота метки: {result}')
        plt.show()
    return MarkOnImage(label_pos_x + wh_mean//2, label_pos_y + wh_mean//2, wh_mean, result)