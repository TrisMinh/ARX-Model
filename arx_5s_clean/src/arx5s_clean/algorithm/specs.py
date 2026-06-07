from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArxSpec:
    na: int
    nb: int
    nk: int
    alpha: float

    @property
    def name(self) -> str:
        return f"ARX_na{self.na}_nb{self.nb}_nk{self.nk}_alpha{self.alpha:g}"


def default_specs(grid: str) -> list[ArxSpec]:
    if grid == "tiny":
        return [
            ArxSpec(na=72, nb=8, nk=4, alpha=10.0),
            ArxSpec(na=96, nb=16, nk=2, alpha=10.0),
        ]
    if grid == "wide":
        return [
            ArxSpec(na=na, nb=nb, nk=nk, alpha=alpha)
            for na in (72, 96, 120, 144)
            for nb in (8, 12, 16, 20)
            for nk in (1, 2, 4)
            for alpha in (1.0, 10.0, 100.0)
        ]
    if grid != "quick":
        raise ValueError("grid must be one of: tiny, quick, wide")
    return [
        ArxSpec(na=72, nb=8, nk=4, alpha=10.0),
        ArxSpec(na=72, nb=12, nk=4, alpha=10.0),
        ArxSpec(na=96, nb=12, nk=4, alpha=10.0),
        ArxSpec(na=96, nb=16, nk=2, alpha=10.0),
        ArxSpec(na=144, nb=12, nk=4, alpha=10.0),
        ArxSpec(na=144, nb=16, nk=2, alpha=10.0),
        ArxSpec(na=144, nb=16, nk=4, alpha=100.0),
    ]

