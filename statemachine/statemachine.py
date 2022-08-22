class Statemachine:

    def __init__(self, init_state=None):
        if not init_state:
            print("warning: no initial state defined for statemachine")

        self.current_state = init_state
        
    def on_event(self, event):
        """
        Delegates an event
        """
        self.current_state = self.current_state.on_event(event, self)
            