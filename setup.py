import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="TA-Lib-py",
    version="0.0.1",
    author="Jun Wang",
    author_email="jstzwj@aliyun.com",
    description="A Python implementation for TA-LIB",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/quantmew/ta-lib",
    project_urls={
        "Bug Tracker": "https://github.com/quantmew/ta-lib/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires='>=3.8',
)