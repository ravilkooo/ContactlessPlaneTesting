import numpy as np
import cv2
from matplotlib import pyplot as plt


def detect_region_by_mouse(image, win_width = 1280, win_height = 720, plt_vis = False):
        # Список для хранения координат кликов
    click_coordinates = []
    image_copy = np.copy(image)

    cv2.namedWindow('Image')

    # Масштабирование изображения для отображения на виджете
    height, width, _ = image.shape
    # print(height, width)
    aspect_ratio = width / height
    
    scaled_width = win_width
    scaled_height = win_height
    if width > win_width or height > win_height:
        if aspect_ratio > 1:
            scaled_width = win_width
            scaled_height = int(win_width / aspect_ratio)
        else:
            scaled_height = win_height
            scaled_width = int(win_height * aspect_ratio)
    
    scaled_image = cv2.resize(image, (scaled_width, scaled_height))
    
    scaled_image_copy = scaled_image.copy()
    # Функция обработки событий мыши
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # Если была нажата левая кнопка мыши
            # global click_coordinates
            # global scaled_image_copy
            click_coordinates.append((x, y))  # Добавляем координаты клика в список
            cv2.circle(scaled_image_copy, (x, y), 3, (0, 255, 0), -1)  # Рисуем круг на месте клика
    click_coordinates = []

    # Установка функции обратного вызова для событий мыши
    cv2.setMouseCallback('Image', mouse_callback)

    while True:
        # Отображение изображения
        cv2.imshow('Image', scaled_image_copy)
        # Ожидание нажатия клавиши "Esc" или "Enter" для выхода
        if cv2.waitKey(1) == 27 or cv2.waitKey(1) == 13:
            break

    click_coordinates = np.array(click_coordinates)*1.
    click_coordinates[:,0] *= (height / scaled_height)
    click_coordinates[:,1] *= (width / scaled_width)

    center = click_coordinates.mean(axis=0)

    if plt_vis:
        # Отображение мест кликов
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for coordinate in click_coordinates:
            plt.plot(coordinate[0], coordinate[1], 'ro')

        plt.plot(center[0], center[1], 'bo')
        plt.title('Результаты нажатия мышью')
        plt.show()

    # Закрытие окна
    cv2.destroyAllWindows()
    # определение центра квадрата
    click_coordinates = np.array(click_coordinates)
    click_coordinates
    rad = int(max(((click_coordinates - center)**2).sum(axis=1)**0.5))
    center = center.astype(int)
    
    if plt_vis:
        image_copy = image.copy()
        # print(image_copy.shape)
        # cv2.namedWindow('Image')
        cv2.circle(image_copy, center, rad, (0, 255, 255), -1)
        plt.imshow(image_copy)
        plt.title('Выбранная окрестность')
        plt.show()
    # cv2.imshow('Image', image_copy)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return center, rad