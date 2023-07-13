# Mark processing, manipulating

import numpy as np
import cv2

class MarkOnImage:
    def __init__(self, x_pos = 255., y_pos = 255., diam = 255., rot_angle = 0.):
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.diam = diam
        self.rot_angle = rot_angle
    def get_center(self):
        return (self.x_pos, self.y_pos)
    
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