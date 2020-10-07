import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="LDA Explanation", # Replace with your own username
    version="0.0.1",
    author="Ellie Rosenman",
    author_email="rosenman@campus.technion.ac.il",
    description="An LDA wrapper for explaining black-box models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EllRos/LDA_Explanation",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'gensim'
    ]
)
