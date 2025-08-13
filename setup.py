import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PySDKit",
    packages=setuptools.find_packages(),
    version="0.4.21",
    description="A Python library for signal decomposition algorithms with a unified interface.",
    url="https://github.com/wwhenxuan/PySDKit",
    author="whenxuan, changewam, josefinez, Yuan Feng",
    author_email="wwhenxuan@gmail.com",
    keywords=[
        "signal decomposition",
        "signal processing",
        "machine learning",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.24.3",
        "scipy>=1.11.1",
        "matplotlib>=3.7.2",
        "tqdm>=4.66.5",
        "requests>=2.32.3",
    ],
    package_data={"": ["*.txt"]},
    include_package_data=True,
)
