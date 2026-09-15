"""Installation script for the standalone rl_lab package."""

from setuptools import find_packages, setup


setup(
    name="rl_lab",
    version="0.1.0",
    packages=find_packages(),
    description="Standalone reinforcement-learning algorithms and Isaac Lab wrappers.",
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "pybullet",
        "torch",
    ],
    zip_safe=False,
)
