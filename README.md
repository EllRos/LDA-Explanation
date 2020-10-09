# LDA Explanation

An LDA wrapper for explaining a black-box classifier's predictions (reference to the black-box NLP workshop here?).

Currently supports only binary predictors.

The module was developed for domain-ruled data (see demo below), although data without domains is supported as well (see API below).


## Installation
#### Git
Get the latest version using git (recommended):
`pip install git+https://github.com/EllRos/LDA-Explanation.git`

#### Wheel
In order to avoid git, get the latest wheel build (might not be updated, but should be):
1. Download https://github.com/EllRos/LDA-Explanation/blob/main/dist/LDA_Explanation-0.0.1-py3-none-any.whl
2. run `pip install LDA_Explanation-0.0.1-py3-none-any.whl` from the download directory.

Note: While typically wanting to just run `pip install https://github.com/EllRos/LDA-Explanation/blob/main/dist/LDA_Explanation-0.0.1-py3-none-any.whl`,
this might cause a strange `BadZipFile` error (even with pip cache disabled).

### Requirements
Installation requiers (and includes) the installation of the following libraries (of any version):
* NumPy
* Pandas
* Matplotlib
* Gensim

Also requires Python version of 3.6 and above.

## Documentation
API documentation: https://ellros.github.io/LDA-Explanation/

Functionality and usage demonstration: https://ellros.github.io/LDA-Explanation/demo/demo.html
