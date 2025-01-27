from setuptools import setup, find_packages

setup(
    name="validation_correction",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "statsmodels>=0.12.0",
        "scipy>=1.5.0",
        "patsy>=0.5.1",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0"
    ],
    author="Paul Connell",
    author_email="paul.connell@columbia.edu",
    description="A package for misclassification error correction in regression using validation data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/pjconnell/validation_correction",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
