"""Lever Analysis functions and scripts for M1 lever push projects.
"""
import subprocess
from setuptools import setup

try:  # Try to create an rst long_description from README.md
    ARGS = "pandoc", "--to", "rst", "README.md"
    LONG_DESC = subprocess.check_output(ARGS)
    LONG_DESC = LONG_DESC.decode()
except Exception as error:  # pylint: disable=broad-except
    print("README.md conversion to reStructuredText failed. Error:\n",
          error, "Setting long_description to None.")
    LONG_DESC = None

setup(
    name="lever",
    version=0.1,
    packages=['lever'],
    install_requires=['seaborn', 'numpy', 'scipy', 'pandas', 'fastdtw', 'noformat',
                      'algorithm', 'mplplot >= v0.1.1', 'matplotlib', 'networkx'],
    dependency_links=["git+https://github.com/Palpatineli/mplplot@master#egg=mplplot-v0.1.1",
                      "git+https://github.com/Palpatineli/algorithm@master#egg=algorithm-0.1"],
    package_data={
        '': ['*.csv', 'src/*'],
        'data': ['*.conf', '*.toml', '*.json'],
    },
    author='Keji Li',
    author_email='mail@keji.li',
    description='',
    tests_require=['pytest'],
    long_description=LONG_DESC
)
