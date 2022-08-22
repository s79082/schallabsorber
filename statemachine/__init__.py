from statemachine.statemachine import Statemachine

class Client(Statemachine):

    listening = 

    def __init__(self, init_state=None):
        super().__init__(init_state)