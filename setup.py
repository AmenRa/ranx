import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rank_eval",
    version="0.1.2",
    author="Elias Bassani",
    author_email="elias.bssn@gmail.com",
    description="rank_eval: A Blazing Fast Python Library for Ranking Evaluation and Comparison",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AmenRa/rank_eval",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy>=1.20.3",
        "numba>=0.54.1",
        "pandas>=1.3.4",
        "tabulate>=0.8.9",
        "tqdm>=4.62.3",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Topic :: Text Processing :: General",
    ],
    keywords=["trec_eval", "information retrieval", "evaluation", "ranking", "numba"],
    python_requires=">=3.8",
)
