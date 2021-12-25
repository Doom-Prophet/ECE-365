from nose.tools import eq_, ok_
from distutils.version import LooseVersion

import nose
import nltk
import sys

def setup_module():
    pass

def test_library_versions():

    # We use Python's distutils to verify versions. It seems sufficient for our purpose.
    # Alternatively, we could use pkg_resources.parse_version for a more robust parse.

    min_python = '3.6'
    min_nltk = '3.4'

    ok_(LooseVersion(sys.version) > LooseVersion(min_python))
    ok_(LooseVersion(nltk.__version__) > LooseVersion(min_nltk))