from pathlib import Path
from setuptools import setup, find_packages


def requirements():
    with open("requirements.txt") as f:
        return f.read().strip().split("\n")


# read the description from the README.md
def readme():
    with open("README.md") as f:
        return f.read().strip()


def version():
    with open("hymc/__version__.py") as f:
        loc = dict()
        exec(f.read(), loc, loc)
        return loc["__version__"]


setup(
    name="HyMC",
    license="GPL-3.0",
    version=version(),
    author="Eduardo Acuña Espinoza",
    author_email="eduardo.espinoza@kit.edu",
    description="Library to calibrate hydrological models for rainfall-runoff prediction",
    long_description=readme(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    install_requires=requirements(),
    packages=find_packages(),
    keywords="hydrology streamflow discharge rainfall-runoff",
)
