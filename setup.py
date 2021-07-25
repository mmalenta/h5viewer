from h5viewer import __version__ 
from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="h5viewer",
    version=__version__,
    author="Mateusz Malenta",
    author_email="mateusz.malenta@gmail.com",
    description="HDF5 archive viewer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={"": "h5viewer"},
    packages=[],
    scripts=["bin/h5viewer.py"]
)
