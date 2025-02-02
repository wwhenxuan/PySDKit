import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='PySDKit',  # 使用包的名称
    packages=setuptools.find_packages(),
    version='0.4.7',  # 包的版本号，应遵循语义版本控制规则
    description='A Python library for signal decomposition algorithms with a unified interface.',  # 包的简短描述
    url='https://github.com/wwhenxuan/PySDKit',  # 项目的地址通常来说是github
    author='whenxuan',
    author_email='wwhenxuan@gmail.com',
    keywords=['signal decomposition', 'signal processing', 'machine learning'],  # 在PyPI上搜索的相应的关键词
    long_description=long_description,  # 这个会将README.md文件的描述放在PyPI网页中
    long_description_content_type="text/markdown",
    classifiers=[
        'Development Status :: 3 - Alpha',  # Alpha表示当前包并不稳定
        'Intended Audience :: Science/Research',  # 当前包使用的人群这里是科研和研究人员
        'Topic :: Scientific/Engineering :: Mathematics',  # 应用的领域
        'Topic :: Scientific/Engineering :: Artificial Intelligence',  # 应用的领域
        'License :: OSI Approved :: MIT License',  # 使用的执照
        'Programming Language :: Python :: 3',  # 适用的Python版本 这里是3.6以上的所有版本
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.6',  # 最低的Python版本限制
    install_requires=[
        'numpy>=1.24.3,<=1.26.4',
        'scipy>=1.11.1,<=1.13.1',
        'matplotlib>=3.7.2,<=3.8.4'
    ],  # 手动指定依赖的Python以及最低的版本
    package_data={'': ['*.txt']},  # 连同数据一起打包
    include_package_data=True
)
