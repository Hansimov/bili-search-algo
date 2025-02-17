from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    name="freq_counter",
    version="0.1.0",
    rust_extensions=[RustExtension("freq_counter.freq_counter", binding=Binding.PyO3)],
    packages=["freq_counter"],
    zip_safe=False,
)
