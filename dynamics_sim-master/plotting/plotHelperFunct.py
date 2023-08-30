from matplotlib import colors
from math import floor
import numpy as np


def graphLines(lineArray, plt):
    for line in lineArray:
        plt.plot([line[0], line[1]], [line[2], line[3]], 'w', lw=10)


def graphColoredLines(lineArray, plt):  # List of lines and associated colors
    for (line, color) in lineArray:
        if type(color) is list:
            plt.plot([line[0], line[1]], [line[2], line[3]], color=tuple(color), lw=15)
        else:
            plt.plot([line[0], line[1]], [line[2], line[3]], color=color, lw=15)


def colorAvg(colorList, proportionList):
    newColor = [0, 0, 0]
    for color, prop in zip(colorList, proportionList):
        if type(color) is str:
            color = colors.colorConverter.to_rgb(color)
        for idx, value in enumerate(color):
            newColor[idx] += value * prop

    return newColor


def stack_proportions(data):  # Turns proportional data into total data
    for generation in data:
        for i, cat in enumerate(generation):
            if not i == 0:
                generation[i] += generation[i-1]


def normalize(value_range, normalize_to=1):  # Normalizes data to 1, useful for type proportions
    for i, step in enumerate(value_range):
        value_range[i] = step / (step + normalize_to)
    return value_range


def plotText(textList, plt, fontsize):
    for position, text in textList:
        plt.text(*position, text, fontsize=fontsize)


def find_wol_props(frequencies, total_thresh_count):
    cwol_prop = dwol_prop = 0  # Mapping thresholds of 1/2 to the appropriate WOL evenly
    for idx, freq in enumerate(frequencies):  # Finding proportions of CWOL and DWOL
        if idx % total_thresh_count == 0:
            dwol_prop += freq
        if floor(idx / total_thresh_count) == 0:
            cwol_prop += freq

    return cwol_prop, dwol_prop


def map_strat_num_to_xy(strat_num, thresholds):
    thresh_count = len(thresholds)

    if strat_num < thresh_count:  # CWOL
        return 0.75, 0.5
    elif strat_num % thresh_count == 0:  # DWOL
        return 0.5, 0.25

    x_val = 1/2 + thresholds[strat_num % thresh_count]
    y_val = 1/2 - thresholds[floor(strat_num / thresh_count)]

    return x_val, y_val


def generate_wol_lines(color=None):
    line_array = []
    count = 90
    for x_cord in range(count):
        line_array.append((x_cord * 0.01 / count + 0.5, x_cord * 0.01 / count + 0.5, 0.485, 0.5))

    line_array = [(0.5, 1, 0.485, 0.485), (0.51, 0.51, 0, 0.5)] + line_array

    if color:
        for idx, line in enumerate(line_array):
            line_array[idx] = (line, color)

    return line_array


def interpolate_values(x_values, y_values, t_values, step_count=1000):
    interp_x_values, interp_y_values, interp_t_values = [], [], []

    for idx, x_val in enumerate(x_values[:-1]):  # Only interpolates for non WOL values
        if (y_values[idx] < 0.495 and y_values[idx + 1] < 0.495) and (x_val > 0.505 and x_values[idx + 1] > 0.505):
            x_list = [x_val, x_values[idx + 1]]
            y_list = [y_values[idx], y_values[idx + 1]]
            t_list = [t_values[idx], t_values[idx + 1]]

            for count in range(step_count):
                x_new = (x_list[1] - x_list[0]) / step_count * count + x_list[0]
                interp_x_values.append(x_new)

                y_new = np.interp(x_new, x_list, y_list)
                interp_y_values.append(y_new)

                t_new = np.interp(x_new, x_list, t_list)
                interp_t_values.append(t_new)

    return interp_x_values, interp_y_values, interp_t_values


def generate_axes(graph, x_values, y_values, font_size):
    orig_x_values = x_values
    x_values = np.array(x_values)
    y_values.reverse()
    y_values = np.array(y_values)

    if len(x_values) < 5:
        graph.xticks(x_values, ["DWOL"] + orig_x_values[1:])
        graph.xticks(y_values, ["CWOL"] + orig_x_values[:-1])
    else:
        limited_x_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
        new_x_values = ["DWOL"] + limited_x_values[1:]
        graph.xticks(limited_x_values, new_x_values)

        limited_y_values = [0.5, 0.4, 0.3, 0.2, 0.1, 0]
        new_y_values = ["CWOL"] + limited_y_values[1:]
        graph.yticks(limited_y_values, new_y_values)
    graph.tick_params(axis='both', which='major', labelsize=font_size, left=False, right=True, labelleft=False, labelright=True, direction='in')
