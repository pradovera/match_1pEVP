import os
from setuptools import find_packages, setup

package_directory = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

setup(name="match_1pEVP",
      description="Match-based 1-parameter eigensolver",
      version="1.1.0",
      license="GNU Library or Lesser General Public License (LGPL)",
      packages=find_packages(package_directory),
      zip_safe=False
      )
