from setuptools import setup, find_packages

__version__ = "0.0.1"

CLASSIFIERS = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Natural Language :: English",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3.9",
    "Topic :: Scientific/Engineering :: Information Analysis",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Software Development :: Libraries :: Python Modules",
]

setup(
    name="kgraph",
    version=__version__,
    description="kGraph",
    classifiers=CLASSIFIERS,
    author="Paul Boniol",
    author_email="paul.boniol@inria.fr",
    packages=find_packages(),
    zip_safe=True,
    license="",
    url="https://github.com/boniolp/kGraph",
    entry_points={},
    install_requires=[
        'numpy==1.24.4',
        'pandas==2.0.3',
        'matplotlib==3.7.2',
        'scipy==1.11.3',
        'scikit-learn==1.2.2',
        'networkx==3.1',
        'pygraphviz==1.11',
        ]
)