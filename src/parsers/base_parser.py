"""
Definition of the Abstract Base Class that will define the structure of a parser regardless of the dataset which
will be used as source
"""
from abc import ABC, abstractmethod


class BaseCANParser(ABC):
    """
    Base interface for CAN datasets parsers
    """

    @abstractmethod
    def parse(self, filepath):
        """
        Reads the file passed as argument in order to return a structured DataFrame to be used for analytical purposes.
        The DataFrame returned will contain at least the following columns: ::
            [
                'id', -> the CAN message ID
                'variable', -> represents the variable (useful for type grouping, especially in ReCAN)
                'time', -> represents the timestamp of the CAN message
                'value' -> represents the actual value of the CAN message
            ]
        :param filepath: the path to the file to be parsed
        :return: a structured DataFrame built over the source file
        """
        pass

    @abstractmethod
    def set_experiment_params(self, filename):
        """
        Given the filename, returns the parameters that need to be used in the experiment according to the model
        :param filename: a string representing the filename for the current experiment
        :return: the list of the correct CAN message IDs, Motion Model implementation and correct analytical functions to
        use within the EKF implementation
        """
        pass
