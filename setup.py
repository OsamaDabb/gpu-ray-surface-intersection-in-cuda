from setuptools import setup, find_packages

setup(
    name="gpu_ray_surface_intersection",
    version="0.1",
    packages=find_packages(where="scripts"),
    package_dir={"": "scripts"},
)
