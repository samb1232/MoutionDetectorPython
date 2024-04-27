from collections import deque
from typing import Sequence

from numba import jit
# Numba - это библиотека для ускорения выполнения Python-кода с использованием JIT (just-in-time) компиляции.
# Мы можем использовать ее, чтобы ускорить выполнение кода.
import numpy as np


def resize_frame(frame, output_size: tuple):
    h, w, _ = frame.shape
    new_h, new_w = output_size

    # Проверяем, что новый размер не превышает исходный
    if new_h > h or new_w > w:
        raise ValueError("New size is bigger than the original size")

    # Вычисляем шаг, с которым будем брать пиксели
    step_h = h // new_h
    step_w = w // new_w

    # Создаем новый массив с нужным размером
    new_frame = np.zeros((new_h, new_w, 3), dtype=np.uint8)

    # Заполняем новый массив, беря пиксели из старого с вычисленным шагом
    new_frame[:, :] = frame[::step_h, ::step_w]

    return new_frame


def convert_to_grayscale(frame: np.ndarray) -> np.ndarray:
    # # Используем умножение матриц NumPy для преобразование в оттенки серого
    grayscale_image = np.dot(frame, [0.2989, 0.5870, 0.1140])

    return grayscale_image.astype(np.uint8)


@jit(nopython=True)
def gaussian_blur(img: np.ndarray, ksize: Sequence[int], sig_max: float) -> np.ndarray:
    # Создаем фильтр Гаусса
    kernel = gaussian_kernel(ksize, sig_max)

    # Применяем фильтр к изображению
    blurred_img = convolve(img, kernel)

    return blurred_img


@jit(nopython=True)
def gaussian_kernel(ksize: tuple, sig_max: float) -> np.ndarray:
    # Создаем ядро фильтра Гаусса
    kernel = np.zeros(ksize)
    center = ksize[0] // 2

    # Вычисляем значения ядра
    for x in range(ksize[0]):
        for y in range(ksize[1]):
            kernel[x, y] = gaussian(x - center, y - center, sig_max)

    # Нормализуем ядро
    kernel /= np.sum(kernel)

    return kernel


@jit(nopython=True)
def gaussian(x: int, y: int, sig_max: float) -> float:
    # Вычисляем значение функции Гаусса для заданных координат и стандартного отклонения
    return np.exp(-(x ** 2 + y ** 2) / (2 * sig_max ** 2))


@jit(nopython=True)
def convolve(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    # Получаем размеры изображения и ядра
    img_height, img_width = img.shape[:2]
    kernel_height, kernel_width = kernel.shape[:2]

    # Вычисляем размеры результирующего изображения
    result_height = img_height - kernel_height + 1
    result_width = img_width - kernel_width + 1

    # Создаем результирующее изображение
    result = np.zeros((result_height, result_width), dtype=np.float32)

    # Применяем свертку
    for y in range(result_height):
        for x in range(result_width):
            # Вычисляем сумму произведений элементов ядра на соответствующие пиксели изображения
            result[y, x] = np.sum(img[y:y + kernel_height, x:x + kernel_width] * kernel)

    return result.astype(np.uint8)


def abs_diff(img1, img2):
    return np.abs(img1 - img2)


def threshold(img: np.ndarray, thresh_val: int):
    thresh = np.where(img > thresh_val, 255, 0)
    return thresh.astype(np.uint8)


def dilate(img: np.ndarray, iterations: int):
    kernel = np.ones((3, 3))
    for _ in range(iterations):
        img = convolve(img, kernel)
    return img


def find_contours(img: np.ndarray):
    contours = []
    visited = set()

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (i, j) not in visited and img[i, j] == 255:
                contour = []
                bfs(img, i, j, contour, visited)
                contours.append(contour)

    return contours


def bfs(img: np.ndarray, start_i: int, start_j: int, contour, visited):
    queue = deque([(start_i, start_j)])
    visited.add((start_i, start_j))

    while queue:
        i, j = queue.popleft()
        contour.append((i, j))

        for x, y in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_i, new_j = i + x, j + y
            if (0 <= new_i < img.shape[0]) and (0 <= new_j < img.shape[1]) and (new_i, new_j) not in visited and img[
                new_i, new_j] == 255:
                queue.append((new_i, new_j))
                visited.add((new_i, new_j))


def bounding_rect(contour):
    min_x = min(contour, key=lambda p: p[0])[0]
    max_x = max(contour, key=lambda p: p[0])[0]
    min_y = min(contour, key=lambda p: p[1])[1]
    max_y = max(contour, key=lambda p: p[1])[1]
    width = max_x - min_x
    height = max_y - min_y
    return min_x, min_y, width, height


def contour_area(contour):
    area = 0.0
    for i in range(len(contour)):
        x1, y1 = contour[i]
        x2, y2 = contour[(i + 1) % len(contour)]
        area += (x1 * y2 - x2 * y1)
    return abs(area) / 2.0
