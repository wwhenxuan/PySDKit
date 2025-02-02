# Fourier Transform
from ._fft import fft, ifft, fftshift, ifftshift  # 傅里叶变换的正变换、逆变换、频谱移动和逆移动

# Signal mirroring extension
from .mirror import fmirror  # 信号的镜像扩展

# Various functions for Hilbert Transform
from .hilbert import hilbert_transform, hilbert_real, hilbert_imaginary  # 希尔伯特变换的相关函数，包括获取实部和虚部
from .hilbert import plot_hilbert, plot_hilbert_complex_plane  # 绘制希尔伯特变换结果的相关函数

from .process import normalize_signal  # 归一化信号
from .process import common_dtype  # 检查数据的通用数据类型
from .process import not_duplicate  # 检查数据是否没有重复

from .process import find_zero_crossings  # 查找信号的零交叉点
from .process import get_timeline  # 获取信号的时间轴

# Algorithm for discrete signal differencing
from ._differ import differ  # 实现离散信号差分的算法

# Functions for 1D signal smoothing
from ._smooth1d import simple_moving_average, weighted_moving_average  # 实现一维信号的简单移动平均和平滑移动平均
from ._smooth1d import gaussian_smoothing, savgol_smoothing, exponential_smoothing  # 高斯平滑、Savitzky-Golay平滑和指数平滑

# Min-max normalization
from ._function import max_min_normalization  # 最大最小归一化
# Z-score standardization
from ._function import z_score_normalization  # Z-score 标准化
# Max-absolute normalization
from ._function import max_absolute_normalization  # 最大绝对值标准化
# Logarithmic transformation
from ._function import log_transformation  # 对数变换
# Decimal scaling normalization
from ._function import decimal_scaling_normalization  # 小数定标标准化

# 寻找一位输入信号的瞬时振幅和瞬时频率
from .instantaneous import inst_freq_local
# 将输入的一维信号分解为两个本征模态函数的形式
from .instantaneous import divide2exp

# Any pair of IMFs is locally orthogonal
from ._Index_of_Orthogonality import index_of_orthogonality
