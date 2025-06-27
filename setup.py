import setuptools
from setuptools import Extension
from Cython.Build import cythonize
from setuptools.command.build_ext import build_ext

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

class CustomBuildExt(build_ext):
    def build_extensions(self):
        compiler_type = self.compiler.compiler_type
        for ext in self.extensions:
            if compiler_type == 'msvc':
                ext.extra_compile_args = ['/Ox']
            elif compiler_type in {'unix', 'mingw32'}:
                ext.extra_compile_args = ['-O3']
            else:
                print(f"Warning: Unknown compiler {compiler_type}, using default flags")
        super().build_extensions()

extensions = [
    Extension("*", ["tabox/ta_func/*.py"])
]



setuptools.setup(
    name="TA-Box",
    version="0.0.1a1.dev8",
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
        "Operating System :: OS Independent",
    ],
    packages=['tabox'],
    python_requires='>=3.6',
    ext_modules=cythonize(
        extensions,
        language_level = "3",
        annotate=True,
        compiler_directives={'language_level' : "3"},   # or "2" or "3str"
    ),
    cmdclass={'build_ext': CustomBuildExt},
    install_requires=[
        'numpy>=1.19.2',
        'cython>=0.29.21',
    ]
)