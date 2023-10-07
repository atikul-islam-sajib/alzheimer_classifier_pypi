from setuptools import setup, find_packages

setup(
    name = "alzheimer_classifier",
    version = "0.1.0",  # Update with the version number of your package
    description = "A deep learning project that is build for predicting the Alzheimer Disease Image dataset",
    author="Atikul Islam Sajib",
    author_email="atikul.sajib@ptb.de",
    url="https://github.com/atikul-islam-sajib/alzheimer_classifier_pypi",  # Update with your project's GitHub repository URL
    packages=find_packages(),
    install_requires=[
        "numpy>=1.0",  # Add any required dependencies here
        "torch>=1.0",
        "scikit-learn>=0.0.1",
        "numpy",
        "pandas", 
        "opencv-python",
        "zipfile36",
        "matplotlib"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    keywords="Mnist classification machine-learning",
    project_urls={
        "Bug Tracker": "https://github.com/atikul-islam-sajib/alzheimer_classifier_pypi/issues",
        "Documentation": "https://github.com/atikul-islam-sajib/alzheimer_classifier_pypi/blob/main/README.md",
        "Source Code": "https://github.com/atikul-islam-sajib/alzheimer_classifier_pypi.git",
    },
    
)













