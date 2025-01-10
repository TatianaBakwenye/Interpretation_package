from setuptools import setup, find_packages
import os

# Read the README for the long description
with open(os.path.join(os.path.dirname(__file__), "README.md"), "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="interpretation_package",
    version="1.0.0",
    description="A package for generating SHAP and Shapley-based plots for model interpretation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Tatiana Bakwenye",
    author_email="tatiana.bakwenye@bse.eu",
    url="https://github.com/TatianaBakwenye/Interpretation_package",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.18.0",
        "pandas>=1.1.0",
        "matplotlib>=3.2.0",
        "seaborn>=0.11.0",
        "shap>=0.40.0",
        "scikit-learn>=0.24.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    include_package_data=True,
    license="MIT",
    license_files=["LICENSE"],
    keywords=["SHAP", "Shapley", "machine learning", "interpretability", "visualization"],
    project_urls={
        "Source": "https://github.com/TatianaBakwenye/Interpretation_package",
    },
)
