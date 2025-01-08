# prompt: create files for a pip pypi package called boxly, a shapely alternative for bounding boxes. Also create a readme. Prefix each file with a comment line of the filename, create at least a setup.py and a README.md

# setup.py
from setuptools import setup, find_packages

setup(
    name='boxly',
    version='0.1.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
    ],
    author='H.T. Kruitbosch',
    author_email='boxly@herbertkruitbosch.com',
    description='Shapely, but for bounding boxes represented by `np.ndarrays`. Not even close to a drop-in replacement though.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/prhbrt/boxly',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Or your chosen license
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.0',
)
