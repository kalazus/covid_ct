from setuptools import find_packages, setup

if __name__ == "__main__":
    setup(
        name="ct_covid_segmentation",
        version="1.0",
        description="Covid19 segmentation on ct scans",
        packages=find_packages("src"),
        package_dir={"": "src"},
    )
