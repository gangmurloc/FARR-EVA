from __future__ import annotations

import unittest

from farr_star.datasets import stratified_hop_slice


class DatasetTests(unittest.TestCase):
    def test_musique_stratified_slice_is_balanced_and_disjoint(self) -> None:
        rows = [
            {"id": f"{hop}hop{variant}__{index}"}
            for hop in (2, 3, 4)
            for variant in ("", "x")
            for index in range(10)
        ]
        first = stratified_hop_slice(rows, 12, 0)
        second = stratified_hop_slice(rows, 12, 4)
        for selected in (first, second):
            counts = {
                hop: sum(
                    str(row["id"]).startswith(f"{hop}hop")
                    for row in selected
                )
                for hop in (2, 3, 4)
            }
            self.assertEqual(counts, {2: 4, 3: 4, 4: 4})
        self.assertFalse(
            {row["id"] for row in first}
            & {row["id"] for row in second}
        )
