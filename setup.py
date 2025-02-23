import pybind11
from setuptools import setup, Extension

extensions = [
    Extension(
        "mixture_bind",
        sources=["src/cpp_ext/mixture_bind.cpp", "src/cpp_ext/mixture_wrapper.cpp",
                 "src/cpp_ext/src/mixture.cpp", "src/cpp_ext/src/mixturelevel.cpp",
                 "src/cpp_ext/src/pointindex.cpp",
                 "src/cpp_ext/src/featurevector.cpp"],
        include_dirs=[pybind11.get_include(), "src/cpp_ext/include/", "src/models/"],
        language="c++",
        extra_compile_args=["/O2", "/std:c++14", "/openmp", "/Ot"]
    )
]

setup(
    name='mixture_bind',
    ext_modules=extensions
)
