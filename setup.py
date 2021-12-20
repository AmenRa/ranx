import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ranx",
    version="0.1.5",
    author="Elias Bassani",
    author_email="elias.bssn@gmail.com",
    description="ranx: A Blazing Fast Python Library for Ranking Evaluation and Comparison",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AmenRa/ranx",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "numba>=0.54.1",
        "pandas",
        "tabulate",
        "tqdm",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Topic :: Text Processing :: General",
    ],
    keywords=["trec_eval", "information retrieval", "evaluation", "ranking", "numba"],
    python_requires=">=3.7",
)
