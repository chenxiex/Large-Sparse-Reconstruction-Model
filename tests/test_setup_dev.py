"""Small offline checks for the development setup helper."""

from __future__ import annotations

import argparse
import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

import setup_dev


class SetupDevTests(unittest.TestCase):
    def test_http_error_message_includes_url(self) -> None:
        url = "https://huggingface.co/api/models/example/tree/main"
        error = urllib.error.HTTPError(url, 403, "Forbidden", None, None)
        self.addCleanup(error.close)
        self.assertEqual(
            setup_dev.error_message(error),
            f"HTTP Error 403: Forbidden (URL: {url})",
        )

    def test_selection_and_manifest_targets(self) -> None:
        with tempfile.TemporaryDirectory(dir=setup_dev.ROOT) as directory:
            root = Path(directory)
            args = argparse.Namespace(
                profile="custom",
                components="rgb,gso",
                datasets_dir=str(root / "mount" / "data"),
                checkpoints_dir=str(root / "mount" / "weights"),
                dinov3_weights_dir=None,
            )
            entries = {
                "checkpoints/rgb": [
                    {"type": "file", "path": "checkpoints/rgb/sparse.pth", "size": 123},
                ],
                "datasets/rgb/gso_example": [
                    {
                        "type": "file",
                        "path": "datasets/rgb/gso_example/test.txt",
                        "size": 9,
                    },
                ],
            }
            with (
                patch.object(setup_dev, "ROOT", root),
                patch.object(
                    setup_dev, "repo_files", side_effect=lambda prefix: entries[prefix]
                ),
            ):
                items = setup_dev.manifest(args, setup_dev.selected(args))
            self.assertEqual(len(items), 2)
            self.assertEqual(
                Path(items[0]["target"]), root / "mount" / "weights" / "rgb/sparse.pth"
            )
            self.assertEqual(
                Path(items[1]["target"]),
                root / "mount" / "data" / "rgb/gso_example/test.txt",
            )

    def test_links_reuse_and_conflict(self) -> None:
        with tempfile.TemporaryDirectory(dir=setup_dev.ROOT) as directory:
            root = Path(directory)
            first = root / "mount-a"
            second = root / "mount-b"
            first.mkdir()
            second.mkdir()
            link = root / "project" / "datasets"
            setup_dev.safe_link(link, first)
            setup_dev.safe_link(link, first)
            self.assertEqual(link.resolve(), first)
            with self.assertRaisesRegex(ValueError, "conflicting symlink"):
                setup_dev.safe_link(link, second)

    def test_dinov3_external_weight_link(self) -> None:
        with tempfile.TemporaryDirectory(dir=setup_dev.ROOT) as directory:
            fixture = Path(directory)
            project = fixture / "project"
            project.mkdir()
            code = fixture / "dinov3"
            code.mkdir()
            (code / "hubconf.py").write_text("", encoding="utf-8")
            weights = project / "mount" / "weights"
            weights.mkdir(parents=True)
            weight = weights / "dinov3_vith16plus.pth"
            weight.write_bytes(b"fixture")
            args = argparse.Namespace(
                dinov3_weight_file=str(weight), dinov3_weight_url=None
            )
            locations = {"dinov3_code": code, "dinov3_weights": weights}
            with patch.object(setup_dev, "ROOT", project):
                setup_dev.prepare_dinov3(args, locations)
                setup_dev.prepare_dinov3(args, locations)
            self.assertEqual((project / "dinov3").resolve(), code)
            self.assertEqual((code / "dinov3_vith16plus.pth").resolve(), weight)

    def test_hf_endpoint_rewrites_only_hub_urls(self) -> None:
        with patch.dict("os.environ", {"HF_ENDPOINT": "https://hf-mirror.example"}):
            self.assertEqual(
                setup_dev.rewrite_hf_url(
                    "https://huggingface.co/org/repo/resolve/main/a.pth"
                ),
                "https://hf-mirror.example/org/repo/resolve/main/a.pth",
            )
            self.assertEqual(
                setup_dev.rewrite_hf_url("https://ai.meta.com/a.pth"),
                "https://ai.meta.com/a.pth",
            )

    def test_hub_download_places_file_at_external_target(self) -> None:
        with tempfile.TemporaryDirectory(dir=setup_dev.ROOT) as directory:
            root = Path(directory)
            cached = root / "cached.bin"
            cached.write_bytes(b"fixture")
            target = root / "mount" / "checkpoints" / "rgb" / "sparse.pth"
            item = {
                "source": "checkpoints/rgb/sparse.pth",
                "target": str(target),
                "size": 7,
            }
            with (
                patch.object(setup_dev, "ROOT", root),
                patch("sys.stdin", io.StringIO(json.dumps([item]))),
                patch(
                    "huggingface_hub.hf_hub_download", return_value=str(cached)
                ) as download,
            ):
                setup_dev.download_helper()
            self.assertEqual(target.read_bytes(), b"fixture")
            self.assertEqual(download.call_args.kwargs["filename"], item["source"])
            self.assertEqual(
                download.call_args.kwargs["cache_dir"], target.parents[1] / ".hf-cache"
            )


if __name__ == "__main__":
    unittest.main()
