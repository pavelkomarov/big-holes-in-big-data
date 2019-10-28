import numpy
from itertools import product
from pickle import dump
from types import MethodType
from datetime import datetime
from multiprocessing import cpu_count
from joblib import Parallel, delayed
from .HyperRectangle import HyperRectangle

## This class implements a monte-carlo-based polynomial-time algorithm for finding Big Holes in Big Data, which is also
# incidentally the title of the paper on which it is based. https://arxiv.org/pdf/1704.00683.pdf The C++ code to go with
# the paper is at https://github.com/joelemley/holesindata
class HoleFinder:

	## Constructor
	# @param data A 2D array, n k-dimensional points stacked together
	# @param strategy A string {sequential, even, random} that controls how rectangle expansion is conducted
	# @param interiorOnly A boolean denoting whether to only consider rectangles that fall completely inside the data,
	#	that are bounded on no side by the edges of the query space
	def __init__(self, data, strategy, interiorOnly):
		self.strategy = {'sequential': self._sequentialExpand, 'even': self._evenExpand, 'random': self._randomExpand}[strategy]
		self.data = data
		self.interiorOnly = interiorOnly
		self.lows = numpy.min(data, axis=0) # find the boundaries of the data
		self.highs = numpy.max(data, axis=0)
		self.n, self.k = data.shape # because I'll use these often
		# projections are a view of hte data along a single dimension. The values along these number lines are sorted
		# and deduplicated to provide a quick way to access the locations of next points in any direction. When a
		# rectangle's edge is expanded, it is always up to some entry in the corresponding projection.
		self.projections = [numpy.unique(data[:,i]) for i in range(self.k)] # unique returns sorted results
		# maps: indicies in each projection -> the datapoints with corresponding values. This datastructure allows quick
		# access to the points that might be blocking a rectangle's expansion in some dimension. It is a memory-for-time
		# tradeoff: takes O(kn) to store, but saves from having to check against all datapoints every time.
		self.maps = [{ndx: numpy.where(data[:,i] == self.projections[i][ndx])[0]
						for ndx in range(len(self.projections[i]))} for i in range(self.k)]
		self.time = str(datetime.now()).replace(' ', '_')

	## Find maximal empty hyper-rectangles, now with parallelization
	# @param maxitr The algorithm stops when no new significant hyper-rectangle is found for this many iterations, where
	#	"significant" means having a volume above the threshold (if one is given) or a new largest volume (when
	#	threshold is not given)
	# @param threshold If given, then this algorithm looks for all rectangles with a volume over this value. If not
	#	given, this algorithm attempts to find the largest few hyper-rectangles.
	# @param whether to print status messages (highly desirable for long runs)
	# @return largest A list of large maximal hyper-rectangles, sorted largest last
	def findLargestMEHRs(self, maxitr, threshold=None, verbose=True):
		c = 0
		maxFound = 0
		hallOfFame = [] if threshold is None else {}
		hofSizes = [] # keep track of how many rectangles are in the hall of fame over time

		while c < maxitr:
			# parallelize finding batches of new HyperRectangles, where batch size is 10*num processes at most
			mehrs = Parallel(n_jobs=cpu_count(), verbose=verbose)(delayed(self._findRandomMEHR)()
				for x in range(min(maxitr-c, cpu_count()*10)))

			# Handle all the rectangles found.
			exterior = 0
			for rect, interior in mehrs:
				volume = rect.volume()
				if not interior: exterior += 1

				# If using a threshold, then collect together all unique rectanges with volume over that threshold
				if (interior or not self.interiorOnly) and threshold and volume > threshold:
					if rect not in hallOfFame: # `in` checked with hash (fast)
						if verbose: print('found new significant rectangle with volume', volume)
						hallOfFame[rect] = volume
						c = 0
					else:
						if verbose: print('found already-discovered rectangle with volume', volume) # very unlikely, but can happen
						c += 1 # count as a failed query
				# If not using threshold, just keep track of new largest rectangles found
				elif (interior or not self.interiorOnly) and not threshold and volume > maxFound:
					if verbose: print('found new largest with volume', volume)
					maxFound = volume
					hallOfFame.append(rect)
					c = 0
				# If the query fails to find a new best or any new (interior) rectangle over the threshold, count up
				else:
					c += 1 # count unsuccessful queries

			hofSizes.append(len(hallOfFame))
			if verbose: print('c=', c, ', maxitr=', maxitr, '%exterior=', exterior*100.0/len(mehrs),
				'last 10 hallOfFame sizes=', hofSizes[-10:], 'total loops=', len(hofSizes))
			dump(hallOfFame, open('MEHRS_' + self.time, 'wb')) # save the largest holes found

		return hallOfFame

	## Function to randomly find a new MEHR, a helper function for findLargestMEHRs
	# @return A maximally expanded HyperRectangle
	def _findRandomMEHR(self):
		numpy.random.seed() # important so that parallel processes don't use the same random numbers
		# Create a random, guaranteed-empty rectangle by choosing a random point in each dimension and letting the upper
		# and lower limits of the rectangle in that dimension equal the values of the closest projected points.
		ehr = HyperRectangle(self.k)
		ndxs = numpy.zeros(self.k, dtype=int) # keep track of where ehr is initially along each projection
		for i in range(self.k): # O(k log n)
			r = numpy.random.uniform(self.lows[i], self.highs[i]) # pick random point in the range of the data
			# With binary search find where in projctions[i] r, if inserted, would keep the array sorted. Note that
			# because lows and highs are defined as the min and max values of projected points, r should never be
			# outside the range of the projections. It is possible (but vanishingly unlikely) r is exactly equal to the
			# low, in which case searchsorted returns the index 0, which causes indexing with ndx-1 to fail. So to be
			# absolutely sure all works, constrain ndx to [1, len(projections[i])-1] rather than [0, len(projections[i])]
			ndxs[i] = numpy.clip(numpy.searchsorted(self.projections[i], r), 1, len(self.projections[i])-1)
			ehr.U[i] = self.projections[i][ndxs[i]]
			ehr.L[i] = self.projections[i][ndxs[i] - 1] # r is between the ndx and ndx-1th items

		# Perform the expansion step to make the ehr into a maximal ehr.
		return self.strategy(ehr, ndxs, ndxs-1) # passing ndxs and ndxs-1 allocates a second array, which is convenient later

	## The sequential strategy involves expanding in one dimension as far as possible, then expanding in the next as
	# far as possible, and so on. Because rectangles start out small and narrow, the expansions typically do not run in
	# to any points for the first few dimensions. As a consequence, they grow long, which makes them more likely to run
	# in to points during later expansions, keeping them skinny.
	# @param ehr A empty HyperRectangle
	# @param undxs An array of ints, the index positions in the projections vectors of the starting rectangle's upper bounds
	# @param lndxs An array of ints, the index positions in the projections vectors of the starting rectangle's lower bounds
	# @return (Rectangle, boolean), the hole found and whether it's bounded completely by points
	def _sequentialExpand(self, ehr, undxs, lndxs): # O(k^2 n)
		interior = True

		for d in numpy.random.permutation(range(self.k)): # expand dimensions in random order to avoid bias
			while True: # try expanding the upper boundary
				upnts = self.data[self.maps[d][undxs[d]]] # Find the points that border the expanded rectangle on the upper side.
				# If any of the points are in the way of expansion, or if we hit the edge of the space, then we have
				# found the upper boundary.
				if numpy.any([ehr.inWay(p, d) for p in upnts]) or undxs[d] >= len(self.projections[d])-1:
					ehr.U[d] = self.projections[d][undxs[d]] # pull out the value of the upper boundary
					interior &= not undxs[d] >= len(self.projections[d]) - 1 # set interior = False if hit the boundary
					break
				else:
					undxs[d] += 1 # consider the next batch, the points with the next-highest value in the dth dimension

			while True: # try expanding the lower boundary
				lpnts = self.data[self.maps[d][lndxs[d]]] # the points that border the expanded rectangle on the lower side
				if numpy.any([ehr.inWay(p, d) for p in lpnts]) or lndxs[d] <= 0:
					ehr.L[d] = self.projections[d][lndxs[d]]
					interior &= not lndxs[d] <= 0
					break
				else:
					lndxs[d] -= 1 # consider the next batch, the points with the next-lowest value in the dth dimension

		return ehr, interior # interior should always be False when using this expansion procedure

	## The even strategy cycles through dimensions in fixed random order, expanding each one randomly up or down by a
	# little, i.e. up to the next set of projected points in that dimension, until the rectangle is bordered on all
	# sides by points. This strategy has the effect of yielding rectangles with fairly even widths in all dimensions.
	# @params and @return See _sequentialExpand
	def _evenExpand(self, ehr, undxs, lndxs): # O(k^2 n)
		order = numpy.random.permutation(range(self.k)) # fixed random order
		ubound = numpy.zeros(self.k, dtype=bool) # keep track of which sides the rectangle is bound on
		lbound = numpy.zeros(self.k, dtype=bool)
		interior = True

		# Loop until the rectangle is bounded on all sides. If interiorOnly, then stop as soon as a rectangle is found
		# not to be interior.
		while not (numpy.all(lbound) and numpy.all(ubound)) and (interior or not self.interiorOnly):
			for d in order: # cycle through dimensions
				coin = numpy.random.randint(2) # randomly decide whether to try going up or down
				
				if coin and not ubound[d]:
					upnts = self.data[self.maps[d][undxs[d]]] # the points that maybe border the rectangle
					# See whether points or the edge of the space actually are in the way of expansion.
					if numpy.any([ehr.inWay(p, d) for p in upnts]) or undxs[d] >= len(self.projections[d])-1:
						ubound[d] = True # HyperRectangle edge is already at this distance, so no need to move it.
						interior &= not undxs[d] >= len(self.projections[d])-1
					else:
						undxs[d] += 1 # Nothing in way, so increment counter and move the rectangle edge a little.
						ehr.U[d] = self.projections[d][undxs[d]] # pull out the value of the upper boundary

				elif not coin and not lbound[d]:
					lpnts = self.data[self.maps[d][lndxs[d]]] # points that maybe border the rectangle
					if numpy.any([ehr.inWay(p, d) for p in lpnts]) or lndxs[d] <= 0:
						lbound[d] = True
						interior &= not lndxs[d] <= 0 # keep interior = True if not hitting boundary
					else:
						lndxs[d] -= 1
						ehr.L[d] = self.projections[d][lndxs[d]] # pull out the value of the lower boundary

				# Notice we could fall through here without really doing anything, but the cycle of dimensions is fixed,
				# so we are guaranteed to hit each one each cycle, so the probability we keep choosing to try to expand
				# a dimension in the direction it's bounded and make no progress is (1/2)^|cycles|.
		return ehr, interior
	
	## The random strategy expands random dimensions randomly up or down by a random number of steps. 
	# @params and @return See _sequentialExpand
	def _randomExpand(self, ehr, undxs, lndxs):
		directions = [x for x in product(range(self.k), [0, 1])] # k dimensions Cartesian-producted with 0, 1 = down, up
		interior = True

		# (dimension, direction) tupes are removed from the directions list when the procedure encounters a boundary.
		while directions and (interior or not self.interiorOnly):
			r = numpy.random.randint(len(directions))
			d, coin = directions[r]
			# Take a small number of steps. If I try to step over the bounary, the length conditions will catch me.
			steps = int(numpy.abs(numpy.random.normal(scale=1))) + 1 # always try to step at least once

			if coin: # going up
				for i in range(steps):
					upnts = self.data[self.maps[d][undxs[d]]]
					# If we read a limit in this direction, then no longer consider it.
					if numpy.any([ehr.inWay(p, d) for p in upnts]) or undxs[d] >= len(self.projections[d])-1:
						directions = directions[:r] + directions[r+1:] # remove the rth entry from the list
						interior &= not undxs[d] >= len(self.projections[d])-1
						break # step no further
					else:
						undxs[d] += 1
						ehr.U[d] = self.projections[d][undxs[d]]

			else: # going down
				for i in range(steps):
					lpnts = self.data[self.maps[d][lndxs[d]]]
					if numpy.any([ehr.inWay(p, d) for p in lpnts]) or lndxs[d] <= 0:
						directions = directions[:r] + directions[r+1:] # slice out this direction
						interior &= not lndxs[d] <= 0
						break
					else:
						lndxs[d] -= 1
						ehr.L[d] = self.projections[d][lndxs[d]]

		return ehr, interior

