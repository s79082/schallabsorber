from enum import Enum
from statemachine.statemachine import Statemachine
from statemachine.state import State

class Client(Statemachine):

    IDLE = ()

    def __init__(self, init_state=None):
        super().__init__(init_state)



class ListeningState(State):
    def on_event(self, event, statemachine):
        if event is Event.START:
            pass
        
        
class Event(Enum):
    START = 0
    STOP = 1