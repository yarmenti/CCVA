import numpy


class EmirCoverTwo(object):
    def __call__(self, exposures_at_default):
        summed = numpy.sum(exposures_at_default, axis=1)
        
        length = len(summed)
        if length == 0:
            raise ValueError("The exposures_at_default is empty")
        elif length <= 2:
            return numpy.max(summed)
        elif length >= 3:
            summed.sort()
            return max(summed[-1], summed[-2] + summed[-3])


class LCHEmirCoverTwo(object):
    def __call__(self, exposures_at_default):
        summed = numpy.sum(exposures_at_default, axis=1)
        
        length = len(summed)
        if length == 0:
            raise ValueError("The exposures_at_default is empty")
        elif length == 1:
            return summed[0]
        else:
            summed.sort()
            return summed[-1] + summed[-2]