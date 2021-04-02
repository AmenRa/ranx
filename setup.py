import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rank_eval",  # Replace with your own username
    version="0.1",
    author="Elias Bassani",
    author_email="elias.bssn@gmail.com",
    description="A collection of fast ranking evaluation metrics built with Numba",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AmenRa/rank_eval",
    packages=setuptools.find_packages(),
    install_requires=["numpy", "numba"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Topic :: Text Processing :: General",
    ],
    keywords=["trec_eval", "information retrieval", "evaluation", "ranking", "numba"],
    python_requires=">=3.6",
)
