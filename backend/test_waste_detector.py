"""Run: python -m unittest discover -s backend -p test_waste_detector.py."""
import os
import unittest
from unittest.mock import patch

from waste_detector import filter_candidate, inference_windows


class DetectionGuards(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_whole_scene_paper_is_rejected(self):
        self.assertIsNone(filter_candidate((0, 0, 1000, 700), .47, 'paper', (0, 0, 1000, 700), (1000, 700)))

    def test_large_river_region_is_rejected_even_at_high_confidence(self):
        self.assertIsNone(filter_candidate((260, 0, 760, 690), .95, 'paper', (0, 0, 1000, 700), (1000, 700)))

    def test_valid_plastic_is_remapped_from_crop(self):
        box = filter_candidate((30, 40, 100, 110), .8, 'plastic', (200, 100, 600, 500), (1000, 700))
        self.assertAlmostEqual(box['x1'], .23)
        self.assertAlmostEqual(box['y2'], .30)

    def test_internal_crop_boundary_is_rejected(self):
        self.assertIsNone(filter_candidate((0, 40, 100, 110), .8, 'plastic', (200, 100, 600, 500), (1000, 700)))

    def test_padding_and_invalid_coordinates_are_rejected(self):
        for box in [(20, -100, 80, -20), (float('nan'), 0, 100, 100)]:
            self.assertIsNone(filter_candidate(box, .9, 'plastic', (0, 0, 1000, 700), (1000, 700)))

    def test_tiles_are_bounded_and_can_be_disabled(self):
        self.assertEqual(len(inference_windows(1600, 1200)), 5)
        with patch.dict(os.environ, {'WASTE_TILES': '0'}):
            self.assertEqual(len(inference_windows(1600, 1200)), 1)


if __name__ == '__main__':
    unittest.main()
