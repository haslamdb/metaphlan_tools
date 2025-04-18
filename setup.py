# setup.py

from setuptools import setup, find_packages

setup(
    name="metaphlan_tools",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0", 
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scikit-bio>=0.5.6",
        "statsmodels>=0.12.0",
    ],
    entry_points={
        "console_scripts": [
            "metaphlan_tools=metaphlan_tools.cli:main",
        ],
    },
    author="David Haslam",
    author_email="dbhaslam@gmail.com",
    description="A package for analyzing nasal microbiome data from MetaPhlAn outputs",
    keywords="microbiome, metagenomics, metaphlan, bioinformatics",
    python_requires=">=3.12",
)
