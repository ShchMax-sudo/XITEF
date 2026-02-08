class FenwickTree:
    # Main tree array
    t: dict = dict()
    # Max possible x coordinate
    maxx: int = 0
    # Max possible y coordinate
    maxy: int = 0

    def __init__(self, photons, maxx: int, maxy: int):
        self.t = dict()
        self.maxx = maxx
        self.maxy = maxy
        for photon in photons:
            x, y = self.event_to_xy(photon)
            self.add(x, y, 1)

    def event_to_xy(self, event):
        return round(event[0]), round(event[1])

    def add(self, x: int, y: int, val: int):
        if x < 0 or y < 0 or x >= self.maxx or y >= self.maxy:
            return
        i = x
        while i <= self.maxx:
            j = y
            while j <= self.maxy:

                if (i, j) in self.t:
                    self.t[(i, j)] += val
                else:
                    self.t[(i, j)] = val
                j += j & -j
            i += i & -i

    def sum(self, x: int, y: int):
        result = 0
        i = x
        while i > 0:
            j = y
            while j > 0:

                if (i, j) in self.t:
                    result += self.t[(i, j)]
                j -= j & -j
            i -= i & -i
        return result

    def count(self, x1: int, x2: int, y1: int, y2: int):
        return self.sum(x2, y2) - self.sum(x2, y1 - 1) - self.sum(x1 - 1, y2) + self.sum(x1 - 1, y1 - 1)
    