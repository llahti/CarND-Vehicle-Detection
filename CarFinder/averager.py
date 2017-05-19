import numpy as np

class Averager:
    """Averager for scalars and 1D arrays"""
    def __init__(self, length, datatype=float(), with_value=True):
        """
        Initializes Averager
        :param length: number of elements on array which are used for calculations
        :param datatype: Data type of array
        :param with_value: If true then initialize all elements with given data
        """
        if with_value:
            self.data = np.full_like([datatype]*length, datatype)
        else:
            self.data = np.empty_like([datatype]*length)
        # Initialize weights used to calculate exponential moving average
        self.weights = np.array([1 / ((x+1) ** 2) for x in range(length)])

    def put(self, data):
        """
        Put new element into array
        :param data: Element
        :return: None
        """
        self.data = np.roll(self.data, shift=1, axis=0)
        self.data[0] = data

    def get(self, index=0):
        """Get element in array on given index. Default 0 --> newest element."""
        return self.data[index]

    def get_all(self):
        """Get all data"""
        return self.data

    def mean(self):
        """Calculate arithmetic mean"""
        avg = np.mean(self.data, axis=0)
        return avg

    def ema(self):
        """Calculate exponential moving average"""
        return np.ma.average(self.data, axis=0, weights=self.weights)