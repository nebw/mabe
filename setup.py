try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


setup(
    name="mabe",
    version="0.1",
    description="",
    author="Benjamin Wild",
    author_email="b.w@fu-berlin.de",
    packages=["mabe"],
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "sklearn",
        "torch",
        "numba",
        "joblib",
        "h5py",
        "matplotlib",
        "torchtyping",
        "fastprogress",
        "madgrad",
        "bayes_opt",
    ],
)
