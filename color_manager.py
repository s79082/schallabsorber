import matplotlib.colors as c

class ColorManager:

    def __init__(self) -> None:
        self.colors: list = list(map(lambda c: c.replace("tab:", ""), c.TABLEAU_COLORS.keys()))
        self.counter: int = -1


    def get_color(self) -> str:
        self.counter += 1
        return self.colors[self.counter]

manager = ColorManager()

def get_color() -> str:
    return manager.get_color()
