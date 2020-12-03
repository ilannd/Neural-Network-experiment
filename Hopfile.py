# -*- coding: utf-8 -*-
"""
Created on Thu May 16 15:37:25 2019

@author: Dell
"""

import numpy as np
from neupy import algorithms
import random


# 绘图
def draw_bin_image(image_matrix):
    for row in image_matrix.tolist():
        print('| ' + ' '.join(' *'[val] for val in row))


# 加噪函数，在记忆样本的基础上增加30%的噪声：
def addnoise(Data):
    for x in range(0, 30):
        if random.randint(0, 10) > 8:
            Data[0, x] = random.randint(0, 1)

    return Data


zero = np.matrix([
    0, 1, 1, 1, 0,
    1, 0, 0, 0, 1,
    1, 0, 0, 0, 1,
    1, 0, 0, 0, 1,
    1, 0, 0, 0, 1,
    0, 1, 1, 1, 0
])

one = np.matrix([
    0, 1, 1, 0, 0,
    0, 0, 1, 0, 0,
    0, 0, 1, 0, 0,
    0, 0, 1, 0, 0,
    0, 0, 1, 0, 0,
    0, 0, 1, 0, 0
])

two = np.matrix([
    1, 1, 1, 0, 0,
    0, 0, 0, 1, 0,
    0, 0, 0, 1, 0,
    0, 1, 1, 0, 0,
    1, 0, 0, 0, 0,
    1, 1, 1, 1, 1,
])
three = np.matrix([
    1, 1, 1, 1, 1,
    0, 0, 0, 0, 1,
    1, 1, 1, 1, 0,
    1, 1, 1, 1, 0,
    0, 0, 0, 0, 1,
    1, 1, 1, 1, 1,
])
four = np.matrix([
    0, 0, 1, 1, 0,
    0, 1, 0, 1, 0,
    1, 0, 0, 1, 0,
    1, 1, 1, 1, 1,
    0, 0, 0, 1, 0,
    0, 0, 0, 1, 0,
])
five = np.matrix([
    1, 1, 1, 1, 1,
    1, 0, 0, 0, 0,
    1, 1, 1, 0, 0,
    0, 0, 1, 1, 1,
    0, 0, 0, 0, 1,
    1, 1, 1, 1, 1,
])
six = np.matrix([
    0, 0, 1, 1, 0,
    0, 1, 0, 0, 0,
    1, 1, 1, 1, 0,
    1, 0, 0, 0, 1,
    1, 0, 0, 0, 1,
    0, 1, 1, 1, 0,
])
seven = np.matrix([
    1, 1, 1, 1, 1,
    0, 0, 0, 0, 1,
    0, 0, 0, 1, 0,
    0, 0, 1, 0, 0,
    0, 1, 0, 0, 0,
    1, 0, 0, 0, 0,
])
eight = np.matrix([
    1, 1, 1, 1, 1,
    1, 0, 0, 0, 1,
    0, 1, 1, 1, 0,
    0, 1, 1, 1, 0,
    1, 0, 0, 0, 1,
    1, 1, 1, 1, 1,
])
nine = np.matrix([
    1, 1, 1, 1, 1,
    1, 0, 0, 0, 1,
    1, 0, 0, 0, 1,
    1, 1, 1, 1, 1,
    0, 0, 0, 1, 0,
    1, 1, 1, 0, 0,
])


def main():
    draw_bin_image(zero.reshape((6, 5)))
    print("\n")
    draw_bin_image(one.reshape((6, 5)))
    print("\n")
    draw_bin_image(two.reshape((6, 5)))
    print("\n")
    draw_bin_image(three.reshape((6, 5)))
    print("\n")
    draw_bin_image(four.reshape((6, 5)))
    print("\n")
    draw_bin_image(five.reshape((6, 5)))
    print("\n")
    draw_bin_image(six.reshape((6, 5)))
    print("\n")
    draw_bin_image(seven.reshape((6, 5)))
    print("\n")
    draw_bin_image(eight.reshape((6, 5)))
    print("\n")
    draw_bin_image(nine.reshape((6, 5)))
    print("\n")

    data = np.concatenate([zero, one, two, three, four], axis=0)
    dhnet = algorithms.DiscreteHopfieldNetwork(mode='sync')
    dhnet.train(data)
    '''
    half_zero = np.matrix([
        0, 1, 1, 1, 0,
        1, 0, 0, 0, 1,
        1, 0, 0, 0, 1,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    ])'''
    ran_zero = addnoise(zero)
    print("对数字0进行随机添加噪声")
    draw_bin_image(ran_zero.reshape((6, 5)))
    print("\n")
    result = dhnet.predict(ran_zero)
    print("对数字0进行联想记忆得到结果")
    draw_bin_image(result.reshape((6, 5)))
    print("\n")

    ran_one = addnoise(one)
    print("对数字1进行随机添加噪声")
    draw_bin_image(ran_one.reshape((6, 5)))
    print("\n")
    result = dhnet.predict(ran_one)
    print("对数字1进行联想记忆得到结果")
    draw_bin_image(result.reshape((6, 5)))
    print("\n")

    ran_two = addnoise(two)
    print("对数字2进行随机添加噪声")
    draw_bin_image(ran_two.reshape((6, 5)))
    print("\n")
    result = dhnet.predict(ran_two)
    print("对数字2进行联想记忆得到结果")
    draw_bin_image(result.reshape((6, 5)))
    print("\n")

    ran_three = addnoise(three)
    print("对数字3进行随机添加噪声")
    draw_bin_image(ran_three.reshape((6, 5)))
    print("\n")
    result = dhnet.predict(ran_three)
    print("对数字3进行联想记忆得到结果")
    draw_bin_image(result.reshape((6, 5)))
    print("\n")

    ran_four = addnoise(four)
    print("对数字4进行随机添加噪声")
    draw_bin_image(ran_four.reshape((6, 5)))
    print("\n")
    result = dhnet.predict(ran_four)
    print("对数字4进行联想记忆得到结果")
    draw_bin_image(result.reshape((6, 5)))
    print("\n")

    data = np.concatenate([five, six, seven, eight, nine], axis=0)
    dhnet = algorithms.DiscreteHopfieldNetwork(mode='sync')
    dhnet.train(data)
    '''
    from neupy import utils
    utils.reproducible()

    dhnet.n_times = 400
    '''
    ran_five = addnoise(five)
    print("对数字5进行随机添加噪声")
    draw_bin_image(ran_five.reshape((6, 5)))
    print("\n")
    result = dhnet.predict(ran_five)
    print("对数字5进行联想记忆得到结果")
    draw_bin_image(result.reshape((6, 5)))
    print("\n")

    ran_six = addnoise(six)
    print("对数字6进行随机添加噪声")
    draw_bin_image(ran_six.reshape((6, 5)))
    print("\n")
    result = dhnet.predict(ran_six)
    print("对数字6进行联想记忆得到结果")
    draw_bin_image(result.reshape((6, 5)))
    print("\n")

    ran_seven = addnoise(seven)
    print("对数字7进行随机添加噪声")
    draw_bin_image(ran_seven.reshape((6, 5)))
    print("\n")
    result = dhnet.predict(ran_seven)
    print("对数字7进行联想记忆得到结果")
    draw_bin_image(result.reshape((6, 5)))
    print("\n")

    ran_eight = addnoise(eight)
    print("对数字8进行随机添加噪声")
    draw_bin_image(ran_eight.reshape((6, 5)))
    print("\n")
    result = dhnet.predict(ran_eight)
    print("对数字8进行联想记忆得到结果")
    draw_bin_image(result.reshape((6, 5)))
    print("\n")

    ran_nine = addnoise(nine)
    print("对数字9进行随机添加噪声")
    draw_bin_image(ran_nine.reshape((6, 5)))
    print("\n")
    result = dhnet.predict(ran_nine)
    print("对数字9进行联想记忆得到结果")
    draw_bin_image(result.reshape((6, 5)))
    print("\n")


if __name__ == "__main__":
    main()
'''
half_two = np.matrix([
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 1, 1, 0, 0,
    1, 0, 0, 0, 0,
    1, 1, 1, 1, 1,
])

draw_bin_image(half_two.reshape((6, 5)))
print("\n")

result = dhnet.predict(half_zero)
draw_bin_image(result.reshape((6, 5)))
print("\n")

result = dhnet.predict(half_two)
draw_bin_image(result.reshape((6, 5)))
print("\n")
'''

'''
half_two = np.matrix([
    1, 1, 1, 0, 0,
    0, 0, 0, 1, 0,
    0, 0, 0, 1, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
])

result = dhnet.predict(half_two)
draw_bin_image(result.reshape((6, 5)))
print("\n")

from neupy import utils
utils.reproducible()

dhnet.mode = 'async'
dhnet.n_times = 400

result = dhnet.predict(half_two)
draw_bin_image(result.reshape((6, 5)))
print("\n")

result = dhnet.predict(half_two)
draw_bin_image(result.reshape((6, 5)))
print("\n")

from neupy import plots
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 12))
plt.title("Hinton diagram")
plots.hinton(dhnet.weight)
plt.show()
'''
