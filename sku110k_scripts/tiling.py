from math import ceil

import numpy as np


class Box:
    def __init__(self, x1=0, y1=0, x2=0, y2=0, data={}):
        if x1 > x2 or y1 > y2:
            raise ValueError("Wrong rectangle coords")
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.data = data

    def area(self):
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def shape(self):
        return int(self.y2 - self.y1), int(self.x2 - self.x1)

    def move(self, dx, dy):
        self.x1, self.x2 = self.x1 + dx, self.x2 + dx
        self.y1, self.y2 = self.y1 + dy, self.y2 + dy

    def to_ints(self):
        self.x1, self.y1, self.x2, self.y2 = (
            int(self.x1),
            int(self.y1),
            int(self.x2),
            int(self.y2),
        )

    def collapse(self, dc=1):
        self.x1, self.x2 = self.x1 + dc, self.x2 - dc
        self.y1, self.y2 = self.y1 + dc, self.y2 - dc

    def as_tuple(self):
        return self.x1, self.y1, self.x2, self.y2

    def is_inside(self, frame):
        return (
            self.x1 >= frame.x1
            and self.y1 >= frame.y1
            and self.x2 <= frame.x2
            and self.y2 <= frame.y2
        )

    def intersection(self, other, data={}):
        a, b = self, other
        x1 = max(min(a.x1, a.x2), min(b.x1, b.x2))
        y1 = max(min(a.y1, a.y2), min(b.y1, b.y2))
        x2 = min(max(a.x1, a.x2), max(b.x1, b.x2))
        y2 = min(max(a.y1, a.y2), max(b.y1, b.y2))
        if x1 < x2 and y1 < y2:
            res = Box(x1, y1, x2, y2, data=data)
            return res
        else:
            return None

    @staticmethod
    def _split(beg, end, side, overlap):
        length = end - beg
        step = side - overlap
        fl_n = float(length) / step
        n = int(fl_n)
        if n <= 1 or (length - side) <= 2.0:
            return [(int(beg), int(end))]
        fl_step = float(length - side) / (n - 1)
        return [
            (int(x), int(x) + side) for x in np.arange(beg, end - side + 0.01, fl_step)
        ]

    def split(self, size=(512, 512), overlap=(100, 100), **kwargs):
        x_side, y_side = size
        x_overlap, y_overlap = overlap

        return [
            Box(x1, y1, x2, y2, data=self.data)
            for x1, x2 in self._split(self.x1, self.x2, x_side, x_overlap)
            for y1, y2 in self._split(self.y1, self.y2, y_side, y_overlap)
        ]


class Tile:
    def __init__(self, box, tile_boxes=[]):
        self.box = box
        self.tile_boxes = tile_boxes

    def check_nested(self):
        for b in self.tile_boxes:
            if not b.is_inside(self.box):
                print(self.box.as_tuple())
                print(b.as_tuple())
                raise ValueError("Box is outside of tile")

    def move_to_origin(self):
        dx = -self.box.x1
        dy = -self.box.y1
        self.box.move(dx, dy)
        for b in self.tile_boxes:
            b.move(dx, dy)

    def to_ints(self):
        self.box.to_ints()
        for x in self.tile_boxes:
            x.to_ints()

    def collapse(self, dc_box=0, dc_nested=1):
        self.box.collapse(dc_box)
        for x in self.tile_boxes:
            x.collapse(dc_nested)

    def intersection(self, other, box_crop_threshold=0.7, greater=True, **kwargs):
        box_common = self.box.intersection(other)
        if box_common is None:
            return None

        nested_intersections = []
        for x in self.tile_boxes:
            isect = box_common.intersection(x, data=x.data)
            if isect is None:
                continue
            isect_area = isect.area()
            x_area = x.area()
            is_area_greater_th = isect_area >= x_area * box_crop_threshold
            if greater == is_area_greater_th:
                nested_intersections.append(isect)

        res = Tile(box_common, nested_intersections)
        res.check_nested()
        return res

    def split(self, **kwargs):
        parts = self.box.split(**kwargs)
        return [self.intersection(r, greater=True, **kwargs) for r in parts]




