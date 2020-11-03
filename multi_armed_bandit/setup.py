from setuptools import Extension, setup
import os 

os.environ["CC"] = "g++"

module = Extension("bandit",
                languange="c++",
                  sources=[
                    'bandit_python.cpp',
                  ])
setup(name='bandit',
     version='1.0',
     description='Python wrapper for custom C extension',
     ext_modules=[module])