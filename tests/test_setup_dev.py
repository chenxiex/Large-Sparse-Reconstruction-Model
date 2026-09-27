"""Small offline checks for the development setup helper."""

from __future__ import annotations

import argparse
import builtins
import io
import json
import os
import runpy
import tempfile
import types
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import Mock, patch

import setup_dev


class SetupDevTests(unittest.TestCase):
    def test_prepare_environment_uses_yaml_and_inherits_pip_index(self) -> None:
        for existing in (False, True):
            with self.subTest(existing=existing):
                with tempfile.TemporaryDirectory(dir=setup_dev.ROOT) as directory:
                    root = Path(directory)
                    env_dir = root / ".conda"
                    python = env_dir / "bin" / "python"
                    if existing:
                        python.parent.mkdir(parents=True)
                        python.touch()
                    with (
                        patch.object(setup_dev, "ROOT", root),
                        patch.object(setup_dev, "conda_executable", return_value=Path("/fake/conda")),
                        patch.object(setup_dev.platform, "system", return_value="Linux"),
                        patch.object(setup_dev.platform, "machine", return_value="x86_64"),
                        patch.object(setup_dev.shutil, "which", return_value="/fake/nvidia-smi"),
                        patch.dict(os.environ, {"PIP_INDEX_URL": "https://mirror.example/simple"}),
                        patch.object(setup_dev.subprocess, "run") as run,
                    ):
                        result = setup_dev.prepare_environment(
                            argparse.Namespace(skip_deps=False), {"env": env_dir}
                        )
                    self.assertEqual(result, python)
                    command = run.call_args_list[0]
                    self.assertEqual(
                        command.args[0],
                        [
                            "/fake/conda", "env", "update" if existing else "create",
                            *([] if existing else ["-y"]),
                            "-p", str(env_dir), "-f", str(root / "environment.yml"),
                        ],
                    )
                    self.assertEqual(
                        command.kwargs["env"]["PIP_INDEX_URL"],
                        "https://mirror.example/simple",
                    )
                    self.assertIn(
                        "export NVCC_PREPEND_FLAGS=",
                        (env_dir / "etc/conda/activate.d/zz_lsrm_env.sh").read_text(),
                    )
                    self.assertIn("ray_aabb_intersect", run.call_args_list[1].kwargs["input"])

    def test_prepare_environment_skip_deps_does_not_run_conda(self) -> None:
        with tempfile.TemporaryDirectory(dir=setup_dev.ROOT) as directory:
            python = Path(directory) / ".conda/bin/python"
            python.parent.mkdir(parents=True)
            python.touch()
            with patch.object(setup_dev.subprocess, "run") as run:
                self.assertEqual(
                    setup_dev.prepare_environment(
                        argparse.Namespace(skip_deps=True), {"env": python.parent.parent}
                    ),
                    python,
                )
            run.assert_not_called()

    def test_verify_checks_torchmetrics_for_dinov3(self) -> None:
        with tempfile.TemporaryDirectory(dir=setup_dev.ROOT) as directory:
            root = Path(directory)
            python = root / ".conda/bin/python"
            python.parent.mkdir(parents=True)
            python.touch()
            dinov3 = root / "dinov3"
            dinov3.mkdir()
            (dinov3 / "hubconf.py").touch()
            (dinov3 / "dinov3_vith16plus.pth").touch()
            output = io.StringIO()
            with (
                patch.object(setup_dev, "ROOT", root),
                patch.object(
                    setup_dev.subprocess,
                    "run",
                    return_value=types.SimpleNamespace(returncode=1),
                ) as run,
                patch("sys.stderr", output),
            ):
                self.assertFalse(
                    setup_dev.verify(("dinov3",), {"env": python.parent.parent})
                )
            self.assertIn("import torchmetrics", run.call_args.args[0][2])
            self.assertIn("torchmetrics", output.getvalue())

    def test_main_loads_dotenv_and_preserves_existing_environment(self) -> None:
        with tempfile.TemporaryDirectory(dir=setup_dev.ROOT) as directory:
            root = Path(directory)
            (root / ".env").write_text(
                "# Local settings\n"
                "HF_ENDPOINT=https://mirror.example # comment\n"
                "export HF_TOKEN='token with spaces'\n"
                'HF_HUB_CACHE="cache with spaces"\n',
                encoding="utf-8",
            )
            with (
                patch.object(setup_dev, "ROOT", root),
                patch.dict(os.environ, {"HF_TOKEN": "from-shell"}, clear=True),
                patch("sys.argv", ["setup_dev.py", "verify"]),
                patch.object(setup_dev, "verify", return_value=True) as verify,
            ):
                self.assertEqual(setup_dev.main(), 0)
                self.assertEqual(setup_dev.endpoint(), "https://mirror.example")
                self.assertEqual(os.environ["HF_TOKEN"], "from-shell")
                self.assertEqual(os.environ["HF_HUB_CACHE"], "cache with spaces")
                verify.assert_called_once()

    def test_download_helper_does_not_import_dotenv(self) -> None:
        original_import = builtins.__import__
        hub = types.ModuleType("huggingface_hub")
        hub.hf_hub_download = Mock()

        def import_without_dotenv(name: str, *args: object, **kwargs: object) -> object:
            if name == "dotenv":
                raise ModuleNotFoundError("No module named 'dotenv'")
            return original_import(name, *args, **kwargs)

        with (
            patch("sys.argv", ["setup_dev.py", "_download"]),
            patch("sys.stdin", io.StringIO("[]")),
            patch.dict("sys.modules", {"huggingface_hub": hub}),
            patch("builtins.__import__", side_effect=import_without_dotenv),
        ):
            with self.assertRaises(SystemExit) as result:
                runpy.run_path(str(setup_dev.ROOT / "setup_dev.py"), run_name="__main__")
        self.assertEqual(result.exception.code, 0)
        hub.hf_hub_download.assert_not_called()

    def test_missing_env_file_is_optional(self) -> None:
        with tempfile.TemporaryDirectory(dir=setup_dev.ROOT) as directory:
            with (
                patch.object(setup_dev, "ROOT", Path(directory)),
                patch("sys.argv", ["setup_dev.py", "verify"]),
                patch.object(setup_dev, "verify", return_value=True) as verify,
            ):
                self.assertEqual(setup_dev.main(), 0)
                verify.assert_called_once()

    def test_hub_download_passes_parent_environment_to_helper(self) -> None:
        item = {"source": "checkpoints/rgb/sparse.pth", "target": "/tmp/sparse.pth", "size": 7}
        with (
            patch.dict(
                os.environ,
                {"HF_TOKEN": "from-dotenv", "HF_ENDPOINT": "https://mirror.example"},
            ),
            patch.object(setup_dev.subprocess, "run") as run,
        ):
            setup_dev.hub_download(
                [item], {"checkpoints": setup_dev.ROOT / "checkpoints"}, Path("python")
            )
        helper_env = run.call_args_list[1].kwargs["env"]
        self.assertEqual(helper_env["HF_TOKEN"], "from-dotenv")
        self.assertEqual(helper_env["HF_ENDPOINT"], "https://mirror.example")

    def test_download_skips_gpu_environment_and_verification(self) -> None:
        with tempfile.TemporaryDirectory(dir=setup_dev.ROOT) as directory:
            root = Path(directory)
            args = [
                "setup_dev.py",
                "download",
                "--profile",
                "custom",
                "--components",
                "rgb",
                "--allow-large-downloads",
            ]
            item = {
                "source": "checkpoints/rgb/sparse.pth",
                "target": str(root / "checkpoints" / "rgb" / "sparse.pth"),
                "size": 7,
            }
            with (
                patch.object(setup_dev, "ROOT", root),
                patch("sys.argv", args),
                patch.object(setup_dev, "manifest", return_value=[item]),
                patch.object(setup_dev, "download_python", return_value=root / "python"),
                patch.object(setup_dev, "hub_download") as download,
                patch.object(setup_dev, "prepare_environment") as environment,
                patch.object(setup_dev, "verify") as verify,
            ):
                self.assertEqual(setup_dev.main(), 0)
                download.assert_called_once_with(
                    [item], setup_dev.paths(setup_dev.parser().parse_args(args[1:])), root / "python"
                )
                environment.assert_not_called()
                verify.assert_not_called()

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
            download = Mock(return_value=str(cached))
            hub = types.ModuleType("huggingface_hub")
            hub.hf_hub_download = download
            with (
                patch.object(setup_dev, "ROOT", root),
                patch("sys.stdin", io.StringIO(json.dumps([item]))),
                patch.dict("sys.modules", {"huggingface_hub": hub}),
            ):
                setup_dev.download_helper()
            self.assertEqual(target.read_bytes(), b"fixture")
            self.assertEqual(download.call_args.kwargs["filename"], item["source"])
            self.assertEqual(
                download.call_args.kwargs["cache_dir"], target.parents[1] / ".hf-cache"
            )

    def test_direct_download_reports_progress_and_replaces_target(self) -> None:
        with tempfile.TemporaryDirectory(dir=setup_dev.ROOT) as directory:
            target = Path(directory) / "asset.bin"
            target.write_bytes(b"old")
            response = io.BytesIO(b"new content")
            response.headers = {"Content-Length": "11", "Content-Type": "application/octet-stream"}
            output = io.StringIO()
            with (
                patch("urllib.request.urlopen", return_value=response),
                patch("time.monotonic", side_effect=[0, 11]),
                patch("sys.stderr", output),
            ):
                setup_dev.download_url("https://example.com/asset.bin", target)
            self.assertEqual(target.read_bytes(), b"new content")
            self.assertIn("asset.bin: 11.0 B / 11.0 B (100%)", output.getvalue())
            self.assertIn("Downloaded asset.bin: 11.0 B", output.getvalue())


if __name__ == "__main__":
    unittest.main()
