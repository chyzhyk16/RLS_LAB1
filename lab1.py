import numpy as np

INPUT_ARRAY = [1065, 1009, 727, 1097, 161, 836, 2351,
               2590, 3006, 1271, 243, 1283, 283, 1591,
               421, 1068, 240, 2836, 516, 2794, 328, 5701,
               52, 1832, 53, 221, 880, 661, 174, 531, 396,
               1714, 1075, 1014, 1174, 1056, 2097, 961,
               855, 662, 24, 1000, 92, 1619, 450, 1451,
               721, 1157, 64, 295, 911, 759, 501, 1211,
               339, 1574, 198, 777, 682, 338, 1627, 278,
               776, 1113, 594, 238, 256, 1224, 60, 1449,
               310, 1612, 1136, 1015, 3078, 145, 443,
               1566, 4440, 446, 580, 798, 2662, 1088, 89,
               1251, 1042, 125, 1898, 532, 6859, 479, 846,
               811, 1199, 331, 721, 323, 130, 61]

N = len(INPUT_ARRAY)

Y = 0.86

T_RELIABILITY = 6274

T_INTENSITY = 763

INTERVALS_COUNT = 10


def get_borders(arr, a):
    for i in range(len(arr)):
        if a > arr[i]:
            return [arr[i - 1], arr[i], i - 1]


def get_denial_intencity(iba, da, t, h):
    borders = get_borders(iba[::-1], t)
    x_0 = 0
    for i in range(len(iba) - borders[2] - 2):
        x_0 += da[i] * h
    x = 1 - x_0 - da[len(iba) - borders[2] - 2] * (t - borders[1])
    return [x, da[len(iba) - borders[2] - 2] / x]


def lab1_function():
    modified_array = np.asarray(sorted(INPUT_ARRAY))
    h = modified_array[-1] / INTERVALS_COUNT
    array_mean = np.mean(modified_array)
    intervals_border_array = np.linspace(0, modified_array[-1], num=INTERVALS_COUNT + 1)
    intervals_array = np.vstack((intervals_border_array[:-1], intervals_border_array[1:])).T

    density_array = np.asarray([
        ((intervals_array[i][0] < modified_array) & (modified_array <= intervals_array[i][1])).sum() / (N * h)
        for i in range(intervals_array.shape[0])])

    probability_array = np.asarray([1] + [
        1 - h * sum(density_array[:i + 1])
        for i in range(len(density_array))])

    ty_borders = get_borders(probability_array, Y)
    ty = h * (ty_borders[0] - Y) / (ty_borders[0] - ty_borders[1]) + h * ty_borders[2]

    print('Середній наробіток до відмови Tср: {:}\n'
          'γ-відсотковий наробіток на відмову Tγ при γ = {:}: {:}\n'
          'Ймовірність безвідмовної роботи на час {:} годин: {:.3}\n'
          'Інтенсивність відмов на час {:} годин: {:.3}'
          .format(array_mean,
                  Y,
                  ty,
                  T_RELIABILITY,
                  get_denial_intencity(intervals_border_array, density_array, T_RELIABILITY, h)[0],
                  T_INTENSITY,
                  get_denial_intencity(intervals_border_array, density_array, T_INTENSITY, h)[1], )
          )


if __name__ == '__main__':
    lab1_function()
