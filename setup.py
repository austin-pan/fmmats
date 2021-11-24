import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fmmats",
    version="0.0.1",
    author="Austin Pan",
    author_email="austinpan8@gmail.com",
    description="A Python library to simplify the creation of files needed for Stephen Rendle's LibFM in relational contexts.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/austin-pan/fmmats",
    project_urls={
        "Bug Tracker": "https://github.com/austin-pan/fmmats/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=[
        "pandas",
        "numpy",
        "sklearn",
    ],
    python_requires=">=3.6",
)
