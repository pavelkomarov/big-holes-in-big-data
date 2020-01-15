import sys; sys.path.append('..') # otherwise I can't import HyperRectangle
import unittest
from HyperRectangle import HyperRectangle
from itertools import product
from matplotlib import pyplot
import numpy

class TestHyperRectangle(unittest.TestCase):

	@classmethod
	def setUpClass(cls):
		cls.rect1 = HyperRectangle(1)
		cls.rect1.U[0] = 5
		cls.rect1.L[0] = 0
		cls.rect3 = HyperRectangle(3)
		cls.rect3.U[:] = [1, 0.8, 0.5]
		cls.rect3.L[:] = [0, 0.5, -0.1]
		cls.data = list(product([0, 1], repeat=3))

	## A test to ensure volume returns accurate results
	def testVolume(self):
		self.assertEqual(self.rect1.volume(), 5)
		self.assertTrue(self.rect3.volume() - 0.18 < 1e-9)

	## A test to ensure contains can tell when points are inside or outisde the rectangle
	def testContains(self):
		self.assertFalse(self.rect1.contains([-1]))
		self.assertTrue(self.rect1.contains([1]))
		self.assertFalse(self.rect3.contains([0.5, 0.6, 0.7]))
		self.assertTrue(self.rect3.contains([0.5, 0.6, 0.3]))

	## A test to ensure intersection finds the correct intersection between two rectangles
	def testIntersect(self):
		other = HyperRectangle(1)
		other.U[0] = 8
		other.L[0] = 4
		intersection = self.rect1.intersect(other)
		self.assertTrue(intersection.L[0] == 4 and intersection.U[0] == 5)
		other.L[0] = 6
		self.assertIsNone(self.rect1.intersect(other))
		other = HyperRectangle(3)
		other.U[:] = [1.5, 0.7, 0.4]
		other.L[:] = [0.5, -0.3, 0.1]
		intersection = self.rect3.intersect(other)
		self.assertTrue(numpy.array_equal(intersection.U, [1, 0.7, 0.4]) and numpy.array_equal(intersection.L, [0.5, 0.5, 0.1]))
		other.U[1] = 0
		self.assertIsNone(self.rect3.intersect(other))

	## test that inWay correctly identifies when points are in the way
	def testInWay(self):
		self.assertTrue(self.rect1.inWay([6], 0)) # things are always in the way in 1D
		self.assertTrue(self.rect3.inWay([0.5, 0.6, 0.7], 2))
		self.assertFalse(self.rect3.inWay([0.5, 0.6, 0.7], 0))

	## test isEmpty returns accurate results
	def testIsEmpty(self):
		self.assertTrue(self.rect3.isEmpty(self.data))
		self.data.append((0.5, 0.6, 0.3))
		self.assertFalse(self.rect3.isEmpty(self.data))

	## show plotting to demonstrate it works
	def testPlot(self):
		k = len(self.data[0])
		self.rect3.plot(numpy.array(self.data), [str(i) for i in range(k*(k-1)//2)])

if __name__ == '__main__':
	unittest.main()
