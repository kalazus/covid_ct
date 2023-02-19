import os

from setuptools import find_packages, setup

_PATH_ROOT = os.path.realpath(os.path.dirname(__file__))
_PATH_REQUIREMENTS = os.path.join(_PATH_ROOT, "requirements.txt")


def _load_requirements(file_path: str = "requirements.txt"):
    """Load requirements from a file."""
    with open(file_path) as file:
        lines = [line.rstrip() for line in file]
    return lines


if __name__ == "__main__":
    setup(
        name="ct_covid_segmentation",
        version="1.0",
        description="Covid19 segmentation on ct scans",
        packages=find_packages("src"),
        package_dir={"": "src"},
        python_requires=">=3.8",
        install_requires=_load_requirements(_PATH_REQUIREMENTS),
    )
