# LDA Explanation

An LDA wrapper for explaining a black-box classifier's predictions (reference to the black-box NLP workshop here?).

Currently supports only binary predictors.

The module was developed for domain-ruled data (see demo below), although data without domains is supported as well (see API below).


## **TODO**
* ~~Add support for PyTorch (remove support for sklearn)~~
* ~~Stress in the docs that domains are optional~~
* ~~Write a description in README.md~~
* ~~Test the code~~
* ~~Update the demo notebook~~
* Add wheel installation option (w/o git)

## Installation
`pip install git+https://github.com/EllRos/LDA-Explanation.git`

### Requirements
Installation requiers (and includes) the installation of the following libraries (of any version):
* NumPy
* Pandas
* Matplotlib
* Gensim

Also requires Python version of 3.6 and above.

## Documentation
API documentation: https://ellros.github.io/LDA-Explanation/

Functionality demonstration: https://ellros.github.io/LDA-Explanation/demo/demo.html
