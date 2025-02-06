# -*- coding: utf-8 -*-
"""
Created on Sat Mar 4 21:31:05 2024
@author: Whenxuan Wang
@email: wwhenxuan@gmail.com
Some auxiliary function modules for data visualization in the PySDKit library
"""
import random
from matplotlib import pyplot as plt
from matplotlib.colors import rgb2hex


def set_themes(choice: str = "classic") -> dict:
    """
    创建绘图主题的函数
    :param choice: 使用的绘图主题名称
    :return: rcParams dict
    """
    dict_themes = {key: value for key, value in plt.rcParams.items()}
    # 设置通用样式
    dict_themes["figure.dpi"] = 400  # 设置整张图像的分辨率
    dict_themes["figure.facecolor"] = "white"  # 设置全局图表背景色
    dict_themes["axes.facecolor"] = "white"  # 设置全局轴背景色
    dict_themes["savefig.bbox"] = "tight"  # 设置自动剪切边缘
    dict_themes["savefig.pad_inches"] = 0.05  # 设置边缘填充
    dict_themes["savefig.format"] = "jpg"  # 设置默认保存格式
    dict_themes["savefig.dpi"] = 800  # 设置保存图像的分辨率
    dict_themes["axes.labelsize"] = 12.5  # 设置坐标轴字体大小
    dict_themes["axes.titlesize"] = 14  # 设置图像标题字体大小
    dict_themes["axes.edgecolor"] = "black"  # 边框颜色（黑色）
    dict_themes["axes.linewidth"] = 1  # 边框线宽

    # 选择具体的主题
    if choice == "classic":
        dict_themes["grid.color"] = "gray"  # 设置网格颜色
        dict_themes["grid.alpha"] = 0.6
        dict_themes["grid.linestyle"] = "--"  # 设置网格线样式
        dict_themes["xtick.bottom"] = True
        dict_themes["ytick.left"] = True
        dict_themes["xtick.direction"] = "in"  # 设置x轴刻度线方向为'内'
        dict_themes["ytick.direction"] = "in"  # 设置y轴刻度线方向为'内'
        dict_themes["xtick.major.width"] = 1  # 设置主刻度的宽度
        dict_themes["xtick.color"] = "k"  # 设置刻度的颜色
        dict_themes["ytick.major.width"] = 1  # 设置主刻度的宽度
        dict_themes["ytick.color"] = "k"  # 设置刻度的颜色
        print("run classic")
    elif choice == "plot_imfs":
        dict_themes["axes.grid"] = False
    elif choice == "kenan":
        dict_themes["figure.facecolor"] = "#DCE3EF"
        dict_themes["axes.linewidth"] = (1.6,)
        dict_themes["axes.labelsize"] = ("large",)
        dict_themes["grid.color"] = "white"
        dict_themes["grid.alpha"] = 1.0
        dict_themes["grid.linestyle"] = "-"
    elif choice == "ggplot":
        plt.style.use("ggplot")
    else:
        raise ValueError

    # 将上述设置保存到默认样式中
    for key, value in dict_themes.items():
        plt.rcParams[key] = value

    return dict_themes


def set_chinese() -> None:
    """设置中文字体与符号的显示"""
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 显示中文
    plt.rcParams["axes.unicode_minus"] = False  # 显示负号


def select_colors(number: int, base_cmap: str = "Set3") -> list:
    """
    根据输入所需的颜色数目返回一个颜色列表
    :param number: 使用的绘图颜色数目
    :param base_cmap: 指定的默认离散颜色映射类型
    :return: color list
    """
    if number <= 3:
        return ["#4169E1", "#DC143C", "#FF8C00"][:number]
    else:
        cmap = plt.get_cmap(base_cmap)
        return [rgb2hex(cmap(i)) for i in range(cmap.N)][:number]


def generate_random_hex_color():
    """
    Generates a random hex color string using matplotlib.
    :return: str: A string representing a hex color code.
    This function generates a random color in RGB format with each color component
    being a random float between 0 and 1. Then, it uses matplotlib's rgb2hex
    function to convert the RGB color to a hex color code.
    """
    # Generate three random numbers between 0 and 1 to represent RGB components
    random_color = [random.random() for _ in range(3)]

    # Convert the RGB color to a hex color code
    hex_color = rgb2hex(random_color)

    return hex_color
