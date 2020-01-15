import numpy as np
import abc

class Instanton(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def position(self, m, t):
        raise NotImplementedError( "Not implemented." )

    @abc.abstractmethod
    def velocity(self, m, t):
        raise NotImplementedError( "Not implemented." )

    @abc.abstractmethod
    def action(self, m):
        raise NotImplementedError( "Not implemented." )
