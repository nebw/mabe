try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


setup(
    name="template",
    version="0.1",
    description="",
    author="Benjamin Wild",
    author_email="b.w@fu-berlin.de",
    packages=["template"],
)
