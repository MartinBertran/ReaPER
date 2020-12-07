from os.path import join, dirname, realpath
from setuptools import setup
import sys

assert sys.version_info.major == 3 and sys.version_info.minor >= 6, \
    "The Spinning Up repo is designed to work with Python 3.6 and greater." \
    + "Please install it before proceeding."

with open("reaper/version.py") as version_file:
    exec(version_file.read())

setup(
    name='reaper',
    py_modules=['reaper'],
    version=0.0,#'0.1',
    description="Minimal implementation of ReaPER: Improving Sample Efficiency in Model-Based Latent Imagination.",
    author="Martin Bertran",
    install_requires=[
            'opencv-python',
            'matplotlib>=3.1.1',
            'plotly',
            'numpy',
            'tqdm',
            'pytest',
            'psutil',
            'scipy',
            'seaborn>=0.8.1',
            'torch>=1.4',
            'tqdm',
            'Cython',
            'glfw',
            'PyOpenGL',
            'PyOpenGL_accelerate',
            'mujoco-py',
            'dm_control',
            'atari_py',
            'moviepy',
            'tensorboard',
            'tensorboardX',
            'scikit-image'

        ],
)
