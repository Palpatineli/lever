from setuptools import setup, find_packages

setup(
    name="lever",
    version=0.1,
    packages=find_packages(),
    install_requires=['matplotlib', 'seaborn', 'numpy', 'scipy', 'pandas', 'fastdtw', 'reader'],
    dependency_links=['git+ssh://git@github.com/Palpatineli/reader.git'],
    package_data={
        '': ['*.csv', 'src/*'],
        'data': ['*.conf', '*.toml', '*.json'],
    },
    author='Keji Li',
    author_email='mail@keji.li',
    description='',
)
