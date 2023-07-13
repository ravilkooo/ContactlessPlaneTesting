import numpy as np
import cv2

# фунцкия поиска лок минимума функции
def golden_section_search(func, min_val, max_val, eps, *args):
    golden_ratio = (1 + 5 ** 0.5) / 2  # Золотое сечение

    a = min_val
    b = max_val

    while abs(b - a) > eps:
        x1 = b - (b - a) / golden_ratio
        x2 = a + (b - a) / golden_ratio

        f1 = func(x1, *args)
        f2 = func(x2, *args)

        if f1 < f2:
            b = x2
        else:
            a = x1

    return (a + b) / 2

def low_clip_value(x, low = -np.inf):
    return max(x, low)

def clip_value(x, low = -np.inf, high = np.inf):
    return min(max(x, low), high)

def calc_diff(a, b, mask = None, reduction='MEAN'):
    if mask:
        if reduction == 'MEAN':
            return (((a - b)**2)*mask).sum() / mask.sum()
        else:#if reduction == 'SUM':
            return (((a - b)**2)*mask).sum()
    else:
        if reduction == 'MEAN':
            return ((a - b)**2).mean()
        else:#if reduction == 'SUM':
            return ((a - b)**2).sum()
        
def fill_out_circle(arr):
    np_arr = 255 - np.array(arr)
    radius = np_arr.shape[0] // 2
    cv2.ellipse(np_arr, (radius, radius), (radius, radius), 0, 0, 360, 0, -1)
    return arr + np_arr