import tempfile
import unittest
from pathlib import Path

from backend.workspace import WorkspaceError, resolve_workspace_path


class WorkspacePathTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name) / "image-restoration-tests"
        self.root.mkdir()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_resolve_workspace_path_allows_file_inside_test_root(self):
        image = self.root / "test-002" / "input" / "original.jpg"
        image.parent.mkdir(parents=True)
        image.write_bytes(b"image")

        self.assertEqual(resolve_workspace_path(self.root, "test-002/input/original.jpg"), image)

    def test_resolve_workspace_path_rejects_traversal_outside_root(self):
        with self.assertRaises(WorkspaceError):
            resolve_workspace_path(self.root, "../private.jpg")

    def test_resolve_workspace_path_rejects_missing_file(self):
        with self.assertRaises(WorkspaceError):
            resolve_workspace_path(self.root, "test-002/input/missing.jpg")


if __name__ == '__main__':
    unittest.main()
