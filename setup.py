import codecs
import os
import re

from setuptools import setup, find_packages


###################################################################

NAME = "strobopy"
PACKAGES = find_packages(where="src")
META_PATH = os.path.join("src", "strobopy", "__init__.py")
KEYWORDS = ["TEM", "stroboscopic", "microscopy"]
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: End Users/Desktop",
    "Natural Language :: English",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: 3.6",
]
INSTALL_REQUIRES = ['scikit-learn', 'numpy', 'pandas']

###################################################################

HERE = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    """
    Build an absolute path from *parts* and and return the contents of the
    resulting file.  Assume UTF-8 encoding.
    """
    with codecs.open(os.path.join(HERE, *parts), "rb", "utf-8") as f:
        return f.read()


META_FILE = read(META_PATH)


def find_meta(meta):
    """
    Extract __*meta*__ from META_FILE.
    """
    meta_match = re.search(
        r"^__{meta}__ = ['\"]([^'\"]*)['\"]".format(meta=meta),
        META_FILE, re.M
    )
    if meta_match:
        return meta_match.group(1)
    raise RuntimeError("Unable to find __{meta}__ string.".format(meta=meta))


if __name__ == "__main__":
    setup(
        name=NAME,
        description=find_meta("description"),
        license=find_meta("license"),
        # url=find_meta("uri"),
        version=find_meta("version"),
        author=find_meta("author"),
        author_email=find_meta("email"),
        maintainer=find_meta("author"),
        maintainer_email=find_meta("email"),
        keywords=KEYWORDS,
        # long_description=read("README.rst"),
        # long_description_content_type="text/x-rst",
        packages=PACKAGES,
        package_dir={"": "src"},
        zip_safe=False,
        classifiers=CLASSIFIERS,
        _requires=INSTALL_REQUIRES,
    )
