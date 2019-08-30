import numpy
from matplotlib import pyplot, patches
from itertools import combinations

## A convenience class to represent a hyper-rectangle, which is really internally a list of upper and a list of lower bounds
class HyperRectangle:

	## Constructor
	# @param dimension Integer dimension of the space where they hyper-rectangle lives
	def __init__(self, dimension):
		self.U = numpy.zeros(dimension) # holds upper bounds for all dimensions
		self.L = numpy.zeros(dimension) # holds lower bounds

	## Makes print() work gracefully with this class
	# @return A string representation of the HyperRectangle
	def __repr__(self):
		return str(numpy.stack((self.U, self.L)))

	## Equality operator (==)
	# @param other An object to compare this HyperRectangle against
	# @return A boolean, whether the other object is equal to this HyperRectangle
	def __eq__(self, other):
		return isinstance(other, HyperRectangle) and numpy.array_equal(self.U, other.U) and numpy.array_equal(self.L, other.L)

	## Inequality operator (!=)
	# @param other An object to compare this HyperRectangle against
	# @return A boolean, whether the other object is not equal to this HyperRectangle
	def __ne__(self, other):
		return not self.__eq__(other)

	## Function to make this class hashable so it can be used as dictionary keys and in sets
	# @return A positive integer, the hash value
	def __hash__(self):
		return hash((self.U.tostring(), self.L.tostring()))

	## Find the space inside this rectangle. In 1D, volume is the difference between the single upper and lower bound;
	# in 2D, volume is the area; in 3D and higher this property is usually called volume.
	# @return numerical volume
	def volume(self):
		return numpy.prod(self.U - self.L)

	## Figure out whether this HyperRectangle contains a point on its interior
	# @param point A k-dimensional vector describing a point, a row of the data matrix
	# @return boolean True if point inside, False if on surface or outside
	def contains(self, point):
		return numpy.all(self.L < point) and numpy.all(self.U > point) # point is in rectangle iff all point_i in [rect_i_min, rect_i_max]

	## Find the rectangle that is the intersection of this rectangle with some other
	# @param other A HyperRectangle against which to compare this one
	# @return A HyperRectangle, the intersection of the two, or None if there is no intersection
	def intersect(self, other):
		if self.L.shape != other.L.shape:
			raise ValueError('Rectangles must be the same dimension to find an intersection.')
		r = HyperRectangle(self.L.shape[0])
		r.L = numpy.maximum(self.L, other.L) # pointwise operation
		r.U = numpy.minimum(self.U, other.U)
		if numpy.any(r.U - r.L < 0): return None
		return r

	## Figure out whether a point lies in the way of expanding this rectangle in the dth dimension
	# @param point A k-dimensional vector describing a point
	# @param d An integer denoting the dimension of expansion
	# @return boolean True if the point lies in the way of expansion, False if off to the side
	def inWay(self, point, d): # O(2k) -> O(k)
		# This is very similar to the contains condition, except we only consider the dimensions aside from d
		aboveL = self.L < point; aboveL[d] = True # set dth entries to True so they can't turn the all condiiton Falsezor
		belowU = self.U > point; belowU[d] = True
		return numpy.all(aboveL) and numpy.all(belowU)

	## A handy auxilliary method to make sure no point in some collection of k-dimensional data is inside the rectangle
	# @param data An n x k array, where each row is a k-dimensional vector representing a point
	# @return boolean True if no points in data are contained in the HyperRectangle, False if any are
	def isEmpty(self, data): # O(kn)
		for p in data:
			if self.contains(p): return False
		return True

	## Create subplots to show what this rectangle looks like among data. Note this only works in somewhat low dimension.
	# @param data An array of k-dimensional data
	# @param featureNames A list of strings denoting what each dimension of the data represents so plots can be labeled
	def plot(self, data, featureNames):
		k = self.L.shape[0]
		if data.shape[1] != k:
			raise ValueError('data must have the same dimension as the HyperRectangle.')

		# Plot 2D representations for all pairs of features. The total number of subplots will be k choose 2 =
		# k*(k-1)/2. Find closest integer factors. Find closest integer factors.
		numPlots = k*(k-1)//2 # numerator is always an even whole number
		height = int(numpy.sqrt(numPlots)) # the number of subplots vertically
		while (numPlots % height != 0): height -= 1
		width = numPlots // height # the number of subplots horizontally

		# iterate through all pairs of dimensions, letting the pair be (x, y) on the plot
		pyplot.figure()
		for i, pair in enumerate(combinations(range(k), 2)):
			axes = pyplot.subplot(height, width, i+1)

			# Some points are "behind" the rectangle, and some are "in front", where the relative behindness or
			# in-frontness is defined as the magnitude of the point in all dimensions which are not plotted.
			otherDimensions = [d for d in range(k) if d not in pair]
			rectangleHeight = numpy.linalg.norm((self.U[otherDimensions] + self.L[otherDimensions])/2.0) # average upper and lower

			pointHeights = numpy.linalg.norm(data[:, otherDimensions], axis=1)
			behind = pointHeights <= rectangleHeight

			# plot the behind and in-front points
			axes.scatter(data[behind, pair[0]], data[behind, pair[1]], zorder=0) # blue
			axes.scatter(data[~behind, pair[0]], data[~behind, pair[1]], zorder=2) # orange

			# plot this rectangle
			axes.add_patch(patches.Rectangle((self.L[pair[0]], self.L[pair[1]]), # corner
												self.U[pair[0]] - self.L[pair[0]], # side length
												self.U[pair[1]] - self.L[pair[1]], # other side length
												edgecolor='r', facecolor='r', alpha=0.5, zorder=1))

			# label the plots
			pyplot.xlabel(featureNames[pair[0]])
			pyplot.ylabel(featureNames[pair[1]])
 
		pyplot.subplots_adjust(left=0.05, right=0.98, bottom=0.05, top=0.9, hspace=0.33, wspace=0.33)
		pyplot.suptitle('Rectangle with volume ' + str(self.volume()))
		pyplot.show()
		
