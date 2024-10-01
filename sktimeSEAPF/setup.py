from pyModbusTCP import constants
from setuptools import setup

with open('README.md') as f:
    readme = f.read()

setup(
    name="BatteryStorageManagementBasedOnPhotovoltaicPowerPrediction",
    version="v0.0.1",
    description="A experimental stuff",
    long_description=readme,
    author="Paweł Parczyk",
    author_email="",
    license="MIT",
    url="",
    packages=["SEAPF"],
    platforms="any",
)