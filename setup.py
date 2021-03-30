from setuptools import setup, find_packages

VERSION = "0.0.1.6"
DESCRIPTION = "AIpaca TT benchmark"
LONG_DESCRIPTION = (
    "used to quickly test the performance of TT predictor for various models"
)

# Setting up
setup(
    # the name must match the folder name 'verysimplemodule'
    name="TTBenchmark",
    version=VERSION,
    author="Yuqi Li",
    author_email="yuqil@aipaca-corp.com",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[],  # add any additional packages that
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.7",
)
