def rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def rgbNorm(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


class Color:
    def __init__(self, name, hexVal):
        self.name = name
        self.hex = hexVal
        self.rgb = rgb(hexVal)
        self.norm = rgbNorm(hexVal)
        self.value = self.rgb  # Emulate enum

    def packet(self):
        return self.hex


class Neon:
    RED = Color('RED', '#800000')
    ORANGE = Color('ORANGE', '#f58231')
    YELLOW = Color('YELLOW', '#ffff00')

    GREEN = Color('GREEN', '#006400')
    MINT = Color('MINT', '#00ff80')
    CYAN = Color('CYAN', '#4169E1')

    BLUE = Color('BLUE', '#000080')
    PURPLE = Color('PURPLE', '#8000ff')
    MAGENTA = Color('MAGENTA', '#ff00ff')
    SALAD = Color('SALAD', '6fce6b')

    FUCHSIA = Color('FUCHSIA', '#ff0080')
    SPRING = Color('SPRING', '#80ff80')
    SKY = Color('SKY', '#0080ff')

    WHITE = Color('WHITE', '#ffffff')
    GRAY = Color('GRAY', '#666666')
    BLACK = Color('BLACK', '#000000')

    BLOOD = Color('BLOOD', '#bb0000')
    BROWN = Color('BROWN', '#8B4513')
    GOLD = Color('GOLD', '#eec600')
    SILVER = Color('SILVER', '#b8b8b8')

    TERM = Color('TERM', '#41ff00')
    MASK = Color('MASK', '#d67fff')

    DARKGREEN = Color('DARKGREEN', '#006400')

    def color12():
        return (Neon.RED, Neon.PURPLE, Neon.BROWN, Neon.GREEN,
                Neon.BLUE, Neon.CYAN,
                Neon.GOLD, Neon.PURPLE, Neon.MAGENTA,
                Neon.FUCHSIA, Neon.SPRING, Neon.SKY)
    
    def contrast_colors():
        return (Neon.RED, Neon.BLUE, Neon.GREEN, Neon.CYAN, Neon.GOLD, Neon.BLOOD, Neon.SALAD)


