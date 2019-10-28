from setuptools import setup

with open('README.md') as readme:
	long_description = readme.read()

setup(name='big-holes-in-big-data',
      version=0.1,
      description='find empty hyperrectangles in high-dimensional point clouds',
	  long_description=long_description,
      url='https://github.com/pavelkomarov/big-holes-in-big-data',
      author='Pavel Komarov',
      license='BSD',
      packages=['bigholes'],
      install_requires=['numpy', 'matplotlib'],
      author_email='pvlkmrv@gmail.com')
