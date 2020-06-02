import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="metrics_eval",  # Replace with your own username
    version="0.0.1",
    author="Elias Bassani",
    author_email="elias.bssn@gmail.com",
    description="A collection of fast metrics built with Numba",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AmenRa/metrics_eval",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
