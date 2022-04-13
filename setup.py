import setuptools
from setuptools import Extension
from Cython.Build import cythonize

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

extensions = [
    Extension("*", ["tabox/*/*.py"]) # extra_compile_args=["-O3", "/Ox"]
    # Extension("mandheling", ["mandheling/*/*.pyx"]),
    # Everything but primes.pyx is included here.
    # Extension("*", ["*.pyx"],
    #     include_dirs=[],
    #     libraries=[],
    #     library_dirs=[]),
]

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
    ext_modules=cythonize(
        extensions,
        language_level = "3",
        annotate=True,
        compiler_directives={'language_level' : "3"},   # or "2" or "3str"
    ),
)