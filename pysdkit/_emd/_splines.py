# -*- coding: utf-8 -*-
"""
Created on Sat Mar 5 22:09:45 2024
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
The following code is mainly used to find extreme points in the EMD algorithm
"""
import numpy as np
from scipy.interpolate import (
    Akima1DInterpolator,
    CubicHermiteSpline,
    CubicSpline,
    PchipInterpolator,
)


def cubic_spline_3pts(x, y, T):
    """
    Custom Cubic Spline Interpolation for 3 Data Points.
    This function implements a cubic spline interpolation specifically for 3 data points.
    Scipy's interp1d does not support cubic spline interpolation for less than 4 points.
    The function calculates the interpolated values at specified points T.

    :param x: Array-like, the x-coordinates of the 3 data points.
    :param y: Array-like, the y-coordinates of the 3 data points.
    :param T: Array-like, the x-coordinates where interpolation is desired.
    :return: Tuple (t, q), where t is the interpolated x-coordinates and q is the interpolated y-coordinates.
    """
    x0, x1, x2 = x
    y0, y1, y2 = y

    x1x0, x2x1 = x1 - x0, x2 - x1
    y1y0, y2y1 = y1 - y0, y2 - y1
    _x1x0, _x2x1 = 1.0 / x1x0, 1.0 / x2x1

    m11, m12, m13 = 2 * _x1x0, _x1x0, 0
    m21, m22, m23 = _x1x0, 2.0 * (_x1x0 + _x2x1), _x2x1
    m31, m32, m33 = 0, _x2x1, 2.0 * _x2x1

    v1 = 3 * y1y0 * _x1x0 * _x1x0
    v3 = 3 * y2y1 * _x2x1 * _x2x1
    v2 = v1 + v3

    M = np.array([[m11, m12, m13], [m21, m22, m23], [m31, m32, m33]])
    v = np.array([v1, v2, v3]).T
    k = np.linalg.inv(M).dot(v)

    a1 = k[0] * x1x0 - y1y0
    b1 = -k[1] * x1x0 + y1y0
    a2 = k[1] * x2x1 - y2y1
    b2 = -k[2] * x2x1 + y2y1

    t = T[np.r_[T >= x0] & np.r_[T <= x2]]
    t1 = (T[np.r_[T >= x0] & np.r_[T < x1]] - x0) / x1x0
    t2 = (T[np.r_[T >= x1] & np.r_[T <= x2]] - x1) / x2x1
    t11, t22 = 1.0 - t1, 1.0 - t2

    q1 = t11 * y0 + t1 * y1 + t1 * t11 * (a1 * t11 + b1 * t1)
    q2 = t22 * y1 + t2 * y2 + t2 * t22 * (a2 * t22 + b2 * t2)
    q = np.append(q1, q2)

    return t, q


def akima(X, Y, x):
    """
    Akima Interpolation.

    This function uses the Akima interpolation method to interpolate values at specified points x.

    Akima interpolation is known for its smoothness and is suitable for curves with rapid changes.

    :param X: Array-like, the x-coordinates of the data points.
    :param Y: Array-like, the y-coordinates of the data points.
    :param x: Array-like, the x-coordinates where interpolation is desired.
    :return: Array-like, the interpolated y-coordinates.
    """
    spl = Akima1DInterpolator(X, Y)
    return spl(x)


def cubic_hermite(X, Y, x):
    """
    Cubic Hermite Spline Interpolation.

    This function uses the Cubic Hermite Spline interpolation method to interpolate values at specified points x.

    Cubic Hermite Spline interpolation ensures that the first derivatives are continuous.

    :param X: Array-like, the x-coordinates of the data points.
    :param Y: Array-like, the y-coordinates of the data points.
    :param x: Array-like, the x-coordinates where interpolation is desired.
    :return: Array-like, the interpolated y-coordinates.
    """
    dydx = np.gradient(Y, X)
    spl = CubicHermiteSpline(X, Y, dydx)
    return spl(x)


def cubic(X, Y, x):
    """
    Cubic Spline Interpolation.

    This function uses the Cubic Spline interpolation method to interpolate values at specified points x.

    Cubic Spline interpolation ensures that the second derivatives are continuous, resulting in a smooth curve.

    :param X: Array-like, the x-coordinates of the data points.
    :param Y: Array-like, the y-coordinates of the data points.
    :return: Array-like, the interpolated y-coordinates.
    """
    spl = CubicSpline(X, Y)
    return spl(x)


def pchip(X, Y, x):
    """
    Piecewise Cubic Hermite Interpolating Polynomial (PCHIP) Interpolation.

    This function uses the PCHIP interpolation method to interpolate values at specified points x.

    PCHIP interpolation ensures that the interpolated curve is monotonic in regions where the data is monotonic.

    :param X: Array-like, the x-coordinates of the data points.
    :param Y: Array-like, the y-coordinates of the data points.
    :param x: Array-like, the x-coordinates where interpolation is desired.
    :return: Array-like, the interpolated y-coordinates.
    """
    spl = PchipInterpolator(X, Y)
    return spl(x)
