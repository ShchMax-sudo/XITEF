import numpy as np
from random import random
from typing import Tuple


class Node:
    # Borders of the node
    left: float = 0
    right: float = 0
    up: float = 0
    down: float = 0
    # Is self node is active flag
    active: bool = True
    # Sons of the node
    ul: "Node" = None
    ur: "Node" = None
    dl: "Node" = None
    dr: "Node" = None
    # Position of the node
    x: float = 0
    y: float = 0
    # Values of the node
    cnt: int = 0  # Number of sons
    mx: float = 0  # "Mass center" x multiplied by cnt
    my: float = 0  # "Mass center" y multiplied by cnt

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def update_nodes(self):
        self.cnt = int(self.active)
        self.mx = self.x * int(self.active)
        self.my = self.y * int(self.active)
        for son in [self.ul, self.ur, self.dl, self.dr]:
            if son is not None:
                self.cnt += son.cnt
                self.mx += son.mx
                self.my += son.my

    def activate(self):
        self.active = True
        self.update_nodes()

    def deactivate(self):
        self.active = False
        self.update_nodes()

    # Returns "sum" of information in nodes such as number of son nodes etc
    def __and__(self, other: "Node"):
        if other is None:
            return self
        if self is None:
            return other
        result = Node(0, 0)
        result.cnt = self.cnt + other.cnt
        result.mx = self.mx + other.mx
        result.my = self.my + other.my
        return result

    def down_left(self, x: float, y: float):
        return x < self.x and y < self.y

    def up_left(self, x: float, y: float):
        return x < self.x and y >= self.y

    def down_right(self, x: float, y: float):
        return x >= self.x and y < self.y

    def up_right(self, x: float, y: float):
        return x >= self.x and y >= self.y

    # Adds new node into the tree
    def __iadd__(self, other: "Node"):
        if other is None:
            return
        if self.down_left(other.x, other.y):
            if self.dl is None:
                self.dl = other
                other.left = self.left
                other.right = self.x
                other.down = self.down
                other.up = self.y
            else:
                self.dl += other
        elif self.up_left(other.x, other.y):
            if self.ul is None:
                self.ul = other
                other.left = self.left
                other.right = self.x
                other.down = self.y
                other.up = self.up
            else:
                self.ul += other
        elif self.down_right(other.x, other.y):
            if self.dr is None:
                self.dr = other
                other.left = self.x
                other.right = self.right
                other.down = self.down
                other.up = self.y
            else:
                self.dr += other
        else:
            if self.ur is None:
                self.ur = other
                other.left = self.x
                other.right = self.right
                other.down = self.y
                other.up = self.up
            else:
                self.ur += other
        other.update_nodes()
        self.update_nodes()
        return self

    def in_bounds(self, x: float, y: float, dist: float, circle_mode: bool):
        if circle_mode:
            raise RuntimeError("There are no circle mode supported")
        else:
            if abs(x - self.left) <= dist and abs(self.right - 1 - x) <= dist and abs(y - self.down) <= dist and abs(self.up - 1 - y) <= dist:
                return True
            elif abs(x - self.left) > dist and abs(self.right - 1 - x) > dist and abs(y - self.down) > dist and abs(self.up - 1 - y) > dist and not (
                    self.left <= x < self.right and self.down <= y < self.up):
                return False
            return None

    def self_in_bounds(self, x: float, y: float, dist: float, circle_mode: bool):
        if circle_mode:
            raise RuntimeError("There are no circle mode supported")
        else:
            return abs(self.x - x) <= dist and abs(self.y - y) <= dist

    def sum_events(self, x: float, y: float, dist: float, circle_mode: bool):
        if self.x < self.left or self.x >= self.right or self.y < self.down or self.y >= self.up:
            print("What???")
        flag = self.in_bounds(x, y, dist, circle_mode)
        if flag is None:
            result = Node(self.x, self.y)
            result.active = self.self_in_bounds(x, y, dist, circle_mode)
            result.update_nodes()
            if self.dl is not None:
                result = result & self.dl.sum_events(x, y, dist, circle_mode)
            if self.dr is not None:
                result = result & self.dr.sum_events(x, y, dist, circle_mode)
            if self.ul is not None:
                result = result & self.ul.sum_events(x, y, dist, circle_mode)
            if self.ur is not None:
                result = result & self.ur.sum_events(x, y, dist, circle_mode)
            if result.cnt != 0:
                return result
        elif flag:
            if not self.in_bounds(x, y, dist, circle_mode):
                print("Watt?")
            return self
        return None


class CartesianTree:
    # Root of the tree
    root: Node = None
    # Bounds of the tree
    left: float = 0
    right: float = 0
    down: float = 0
    up: float = 0
    # Form of the searching area around the point
    circle_mode: bool = False
    # Size of the searching area around the point
    dist: float = 0

    def __init__(self, photons, right: float, up: float, dist: float, circle_mode: bool = False):
        events = np.zeros((len(photons),), dtype=[("Rank", float), ("X", float), ("Y", float)])
        for i in range(len(events)):
            events[i] = random(), photons[i][0], photons[i][1]
        events = np.sort(events, order="Rank")
        self.right = right + dist + 1
        self.up = up + dist + 1
        self.left = -dist
        self.down = -dist
        self.circle_mode = circle_mode
        self.dist = dist
        self.root = Node(events[0][1], events[0][2])
        self.root.left = self.left
        self.root.right = self.right
        self.root.down = self.down
        self.root.up = self.up
        for i in range(1, len(events)):
            self.root += Node(events[i][1], events[i][2])

    # Adds a new node into the tree by its coordinates
    def __iadd__(self, xy: Tuple[float, float]):
        x = xy[0]
        y = xy[1]
        if self.root is None:
            self.root = Node(x, y)
            self.root.left = self.left
            self.root.right = self.right
            self.root.down = self.down
            self.root.left = self.left
        else:
            self.root += Node(x, y)

    # Deactivates or activates the node
    def activate(self, x: float, y: float, activate: bool = True):
        node = self.root
        while node.x != x or node.y != y:  # There should never be an error!!!
            if node.down_left(x, y):
                node = node.dl
            elif node.up_left(x, y):
                node = node.ul
            elif node.down_right(x, y):
                node = node.dr
            else:
                node = node.ur

        if activate:
            node.activate()
        else:
            node.deactivate()

    # Finds sum of all events near to xy
    def __mul__(self, xy: Tuple[float, float]):
        result = self.root.sum_events(xy[0], xy[1], self.dist, self.circle_mode)
        return result.cnt, result.mx / result.cnt, result.my / result.cnt

    def deactivate_tree(self, node: Node):
        if type(node) == bool:
            node = self.root
        if node is None:
            return
        node.deactivate()
        self.deactivate_tree(node.dl)
        self.deactivate_tree(node.dr)
        self.deactivate_tree(node.ul)
        self.deactivate_tree(node.ur)

    def print(self, node: Node):
        if node is None:
            return
        print(node.left, node.right, node.down, node.up, node.x, node.y)
        print("down-left")
        self.print(node.dl)
        print("down-right")
        self.print(node.dr)
        print("up-left")
        self.print(node.ul)
        print("up-right")
        self.print(node.ur)

