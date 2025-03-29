# setup.py

from setuptools import setup, find_packages

setup(
    name="metaphlan_tools",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scipy>=1.7.0",
        "scikit-bio>=0.5.6",
        "statsmodels>=0.12.0",
        "scikit-learn>=1.0.0",
        "networkx>=2.6.0",
    ],
    entry_points={
        "console_scripts": [
            "metaphlan_tools=metaphlan_tools.cli:main",
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A package for analyzing nasal microbiome data from MetaPhlAn outputs",
    keywords="microbiome, metagenomics, metaphlan, bioinformatics",
    python_requires=">=3.8",
)
