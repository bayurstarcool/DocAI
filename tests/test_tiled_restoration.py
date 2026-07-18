import unittest

import numpy as np
from PIL import Image
import torch

from backend.utils.image_utils import run_tiled_restoration


class IdentityRestorer(torch.nn.Module):
    def forward(self, tensor):
        return tensor, tensor.mean(dim=1, keepdim=True)


class TiledRestorationTests(unittest.TestCase):
    def test_preserves_non_multiple_high_resolution_shape(self):
        source = np.random.default_rng(42).integers(0, 256, (517, 773, 3), dtype=np.uint8)
        image = Image.fromarray(source)
        with torch.inference_mode():
            result, mask = run_tiled_restoration(IdentityRestorer(), image, 'cpu', tile_size=256, overlap=32)
        self.assertEqual(result.size, image.size)
        self.assertEqual(mask.size, image.size)
        np.testing.assert_allclose(np.asarray(result), source, atol=1)

    def test_rejects_invalid_tile_parameters(self):
        image = Image.new('RGB', (16, 16))
        with self.assertRaises(ValueError):
            run_tiled_restoration(IdentityRestorer(), image, 'cpu', tile_size=255)
        with self.assertRaises(ValueError):
            run_tiled_restoration(IdentityRestorer(), image, 'cpu', tile_size=16, overlap=16)


if __name__ == '__main__':
    unittest.main()
