from setuptools import setup, find_packages

with open('README.md') as readme:
	long_description = readme.read()

setup(name='big-holes-in-big-data',
      version=0.2,
      description='find empty hyperrectangles in high-dimensional point clouds',
	  long_description=long_description,
      url='https://github.com/pavelkomarov/big-holes-in-big-data',
      author='Pavel Komarov',
      license='BSD',
      packages=[x for x in find_packages() if 'tests' not in x],
      install_requires=['numpy', 'matplotlib'],
      author_email='pvlkmrv@gmail.com')
