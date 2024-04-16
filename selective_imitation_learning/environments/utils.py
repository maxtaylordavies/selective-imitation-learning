from dataclasses import dataclass
from enum import IntEnum
from typing import Tuple


@dataclass
class Position:
    x: int
    y: int

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Position):
            return False
        return self.x == __value.x and self.y == __value.y

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    # a bit cheeky - we override subtraction operator to return manhattan distance
    def __sub__(self, other: "Position") -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)

    def to_np_idx(self) -> Tuple[int, int]:
        return (self.y, self.x)

    @classmethod
    def copy(cls, other: "Position") -> "Position":
        return cls(other.x, other.y)


class GridActions(IntEnum):
    up: int = 0
    down: int = 1
    left: int = 2
    right: int = 3
    no_op: int = 4


delta_x_actions = {-1: [GridActions.up], 1: [GridActions.down], 0: []}
delta_y_actions = {-1: [GridActions.left], 1: [GridActions.right], 0: []}
