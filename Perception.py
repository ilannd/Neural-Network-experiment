#!/usr/bin/env python
# -*- coding:utf-8 -*-
import random

import numpy as np
import matplotlib.pyplot as plt

xh_number = 5000


class My_Per:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.errors = []
        self.b = random.random()
        self.w = [random.uniform(-1, 1) for _ in range(2)]

    def train(self, Px, t):
        update = self.lr * (t - self.predict(Px))
        self.b += update
        self.w[0] += update * Px[0]
        self.w[1] += update * Px[1]

    def predict(self, Px):
        number = self.w[0] * Px[0] + self.w[1] * Px[1] + self.b
        return np.where(number >= 0, 1, 0)


def main():
    P = [[3.6, 6.6], [9.3, 6.3], [7.1, 8.1], [4.0, 4.1], [4.2, 4.2], [2.8, 2.9], [7.1, 7.3], [9.2, 7.6], [8.1, 7.8],
         [2.9, 4.9], [9.3, 8.2], [4.2, 3.5]]
    T = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    my_per = My_Per(0.01)
    plt.figure()
    x1 = [3.6, 2.9, 4.2, 4.0, 4.2, 2.8, 7.1, 9.2, 8.1, 9.3, 9.3, 7.1]
    x2 = [6.6, 4.9, 3.5, 4.1, 4.2, 2.9, 7.3, 7.6, 7.8, 6.3, 8.2, 8.1]
    plt.plot(x1[:6], x2[:6], "ro")
    plt.plot(x1[7:], x2[7:], "bo")
    # plt.scatter(x1, x2)
    for i in range(xh_number):
        for i in range(12):
            Px = P[i]
            t = T[i]
            my_per.train(Px, t)
        print(-my_per.w[0] / my_per.w[1])
        print(-my_per.b / my_per.w[1])
    x = np.arange(1, 13)
    y = -my_per.w[0] / my_per.w[1] * x - (my_per.b / my_per.w[1])*2.9
    # y=-(my_per.w[0]*x+my_per.b)/my_per.w[1]
    plt.plot(x, y)

    plt.show()


if __name__ == "__main__":
    main()
