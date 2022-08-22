class State(object):
    """
    We define a state object which provides some utility functions for the
    individual states within the state machine.
    """

    def on_event(self, event, statemachine):
        """
        Handle events that are delegated to this State with the 
        statemachine (or subclass) tht called this function
        """
        raise NotImplementedError()

    def __repr__(self):
        """
        Leverages the __str__ method to describe the State.
        """
        return self.__str__()

    def __str__(self):
        """
        Returns the name of the State.
        """
        return self.__class__.__name__