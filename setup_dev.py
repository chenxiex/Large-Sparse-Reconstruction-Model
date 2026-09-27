"""Prepare the LSRM development environment and its optional assets."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
REPO = "facebook/Large-Sparse-Reconstruction-Model"
COMPONENTS = ("rgb", "brdf", "gso", "orb", "dtc", "dinov3", "blender", "env")
SMOKE = ("rgb", "gso", "dinov3", "blender")
BLENDER_URL = (
    "https://mirrors.aliyun.com/blender/release/Blender4.5/blender-4.5.3-linux-x64.tar.xz"
)
BLENDER_BYTES = 377_397_328
LARGE_LIMIT = 1_000_000_000
ACTIVATE_HOOK = """\
export NVCC_PREPEND_FLAGS="-ccbin=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
export NVCC_CCBIN="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
export TORCH_EXTENSIONS_DIR="$CONDA_PREFIX/var/torch_extensions"
mkdir -p "$TORCH_EXTENSIONS_DIR"

__cap="$(python -c 'import torch; m,n=torch.cuda.get_device_capability(0); print(f"{m}.{n}")' 2>/dev/null || true)"
export TORCH_CUDA_ARCH_LIST="${__cap:-8.0;8.6;8.9;9.0+PTX}"
unset __cap

__nvidia_pkg_dir="$(python -c 'import nvidia,os; print(os.path.dirname(nvidia.__file__))' 2>/dev/null || true)"
if [ -n "$__nvidia_pkg_dir" ]; then
    for __d in cublas cusparse cusolver curand cufft cuda_runtime cuda_cccl cuda_nvcc nvjitlink nvtx; do
        [ -d "$__nvidia_pkg_dir/$__d/include" ] && \\
            CPATH="$__nvidia_pkg_dir/$__d/include:${CPATH:-}"
    done
    export CPATH
fi
unset __nvidia_pkg_dir __d
"""
CUDA_CHECK = """\
import torch
import torchvision
import torchaudio
import nerfacc
import flash_attn
from pytorch3d import _C
from nerfacc import ray_aabb_intersect

assert torch.__version__.split("+")[0] == "2.4.0", torch.__version__
assert torchvision.__version__.split("+")[0] == "0.19.0", torchvision.__version__
assert torchaudio.__version__.split("+")[0] == "2.4.0", torchaudio.__version__
assert torch.version.cuda == "12.1", torch.version.cuda
assert torch.cuda.is_available(), "CUDA GPU is unavailable"
rays_o = torch.zeros(1, 3, device="cuda")
rays_d = torch.tensor([[0.0, 0.0, 1.0]], device="cuda")
aabbs = torch.tensor([[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]], device="cuda")
ray_aabb_intersect(rays_o, rays_d, aabbs)
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("nerfacc:", nerfacc.__version__)
print("flash_attn:", flash_attn.__version__)
print("pytorch3d _C and nerfacc CUDA extension OK")
"""


def endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def request_json(url: str) -> tuple[list[dict[str, Any]], str | None]:
    headers = {"Accept": "application/json"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with urllib.request.urlopen(
        urllib.request.Request(url, headers=headers), timeout=30
    ) as response:
        return json.load(response), response.headers.get("Link")


def repo_files(prefix: str) -> list[dict[str, Any]]:
    """List a subtree, following Hub pagination without downloading its files."""
    lookup = prefix if "/" in prefix else ""
    encoded = urllib.parse.quote(lookup, safe="")
    suffix = f"/{encoded}" if lookup else ""
    url: str | None = (
        f"{endpoint()}/api/models/{REPO}/tree/main{suffix}?recursive=true&expand=true&limit=100"
    )
    files: list[dict[str, Any]] = []
    while url:
        page, links = request_json(url)
        files.extend(item for item in page if item.get("type") == "file")
        match = re.search(r'<([^>]+)>;\s*rel="next"', links or "")
        url = rewrite_hf_url(match.group(1)) if match else None
    return files


def absolute_path(value: str | None, default: Path) -> Path:
    return Path(value).expanduser().resolve() if value else default.resolve()


def paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "datasets": absolute_path(args.datasets_dir, ROOT / "datasets"),
        "checkpoints": absolute_path(args.checkpoints_dir, ROOT / "checkpoints"),
        "dinov3_weights": absolute_path(
            args.dinov3_weights_dir, ROOT.parent / "dinov3"
        ),
        "dinov3_code": ROOT.parent / "dinov3",
        "blender": ROOT.parent / "blender" / "blender-4.5.3-linux-x64",
        "env": ROOT / ".conda",
    }


def selected(args: argparse.Namespace) -> tuple[str, ...]:
    if args.profile == "smoke":
        if args.components:
            raise ValueError("--components requires --profile custom")
        return SMOKE
    if not args.components:
        raise ValueError("--profile custom requires --components")
    items = tuple(dict.fromkeys(part.strip() for part in args.components.split(",")))
    unknown = set(items) - set(COMPONENTS)
    if unknown:
        raise ValueError(f"unknown components: {', '.join(sorted(unknown))}")
    return items


def manifest(
    args: argparse.Namespace, components: tuple[str, ...]
) -> list[dict[str, Any]]:
    locations = paths(args)
    items: list[dict[str, Any]] = []
    prefixes: list[str] = []
    for component in components:
        if component in ("rgb", "brdf"):
            prefixes.append(f"checkpoints/{component}")
        elif component in ("gso", "orb", "dtc"):
            family = "rgb" if component == "gso" else "brdf"
            prefixes.append(f"datasets/{family}/{component}_example")
        elif component == "env":
            prefixes.append("env.exr")
    for prefix in prefixes:
        for entry in repo_files(prefix):
            name = str(entry["path"])
            if not (name == prefix or name.startswith(prefix + "/")):
                continue
            category, _, remainder = name.partition("/")
            target = (
                ROOT / name
                if category == "env.exr"
                else locations[category] / remainder
            )
            items.append(
                {
                    "source": name,
                    "target": str(target),
                    "size": int(entry.get("size") or 0),
                }
            )
    return sorted(items, key=lambda item: item["source"])


def human_size(size: int) -> str:
    return f"{size / 1_000_000_000:.2f} GB"


def human_bytes(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1000 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1000
    raise AssertionError("unreachable")


def external_weight_size(args: argparse.Namespace) -> int | None:
    if args.dinov3_weight_file:
        return Path(args.dinov3_weight_file).expanduser().stat().st_size
    if not args.dinov3_weight_url:
        return None
    weight_url = rewrite_hf_url(args.dinov3_weight_url)
    headers: dict[str, str] = {}
    if (
        urllib.parse.urlsplit(weight_url).netloc
        == urllib.parse.urlsplit(endpoint()).netloc
    ):
        token = os.environ.get("HF_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(weight_url, headers=headers, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            length = response.headers.get("Content-Length")
    except urllib.error.URLError:
        return None
    return int(length) if length else None


def show_plan(
    args: argparse.Namespace, components: tuple[str, ...], items: list[dict[str, Any]]
) -> int:
    locations = paths(args)
    print(f"Hugging Face endpoint: {endpoint()}")
    print(f"Components: {', '.join(components)}")
    total = 0
    for item in items:
        target = Path(item["target"])
        pending = not (target.is_file() and target.stat().st_size == item["size"])
        if pending:
            total += item["size"]
        print(
            f"{'download' if pending else 'present ':8} {human_size(item['size']):>9}  {item['source']} -> {target}"
        )
    if "dinov3" in components:
        weight = locations["dinov3_weights"] / "dinov3_vith16plus.pth"
        if args.dinov3_weight_file:
            weight = Path(args.dinov3_weight_file).expanduser().resolve()
        if not weight.is_file():
            size = external_weight_size(args)
            if args.dinov3_weight_url:
                print(
                    f"download {human_size(size) if size is not None else 'unknown':>9}  DINOv3 weight -> {weight}"
                )
                if size is not None:
                    total += size
            else:
                print(
                    "required DINOv3 weight: provide --dinov3-weight-file or --dinov3-weight-url"
                )
        print(f"DINOv3 code: {locations['dinov3_code']} (git clone if absent)")
    if "blender" in components and not (locations["blender"] / "blender").is_file():
        total += BLENDER_BYTES
        print(
            f"download {human_size(BLENDER_BYTES):>9}  Blender 4.5.3 -> {locations['blender']}"
        )
    print(
        f"Known pending download: {human_size(total)}; dependency and unknown-size downloads are additional"
    )
    print(
        f"Hugging Face cache: {os.environ.get('HF_HUB_CACHE', 'next to each selected asset directory')}"
    )
    return total


def safe_link(link: Path, target: Path) -> None:
    """Create a link only when an existing path does not contain user data."""
    target = target.resolve()
    if link.is_symlink():
        if link.resolve() == target:
            return
        raise ValueError(
            f"conflicting symlink: {link} -> {link.resolve()}; expected {target}"
        )
    if link.exists():
        if link.resolve() == target:
            return
        raise ValueError(f"existing path blocks symlink: {link}; expected {target}")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=target.is_dir())


def rewrite_hf_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    if parsed.netloc.lower() != "huggingface.co":
        return url
    base = urllib.parse.urlsplit(endpoint())
    return urllib.parse.urlunsplit(
        (
            base.scheme,
            base.netloc,
            base.path.rstrip("/") + parsed.path,
            parsed.query,
            parsed.fragment,
        )
    )


def download_url(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    url = rewrite_hf_url(url)
    print(f"Downloading {target}", file=sys.stderr, flush=True)
    headers: dict[str, str] = {}
    if urllib.parse.urlsplit(url).netloc == urllib.parse.urlsplit(endpoint()).netloc:
        token = os.environ.get("HF_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
    with tempfile.NamedTemporaryFile(
        dir=target.parent, prefix=target.name + ".", delete=False
    ) as temp:
        temporary = Path(temp.name)
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers=headers), timeout=120
            ) as source:
                if "text/html" in source.headers.get("Content-Type", ""):
                    raise ValueError(
                        "download returned an HTML page instead of a file; check access credentials"
                    )
                length = source.headers.get("Content-Length")
                total = int(length) if length and length.isdigit() else None
                received = 0
                last_report = time.monotonic()
                interactive = sys.stderr.isatty()
                while chunk := source.read(1024 * 1024):
                    temp.write(chunk)
                    received += len(chunk)
                    now = time.monotonic()
                    if now - last_report >= (1 if interactive else 10):
                        progress = f"{human_bytes(received)}"
                        if total:
                            progress += f" / {human_bytes(total)} ({received / total:.0%})"
                        prefix = "\r" if interactive else ""
                        print(
                            f"{prefix}{target.name}: {progress}",
                            end="" if interactive else "\n",
                            file=sys.stderr,
                            flush=True,
                        )
                        last_report = now
                if interactive:
                    print(file=sys.stderr)
            temporary.replace(target)
            print(
                f"Downloaded {target.name}: {human_bytes(received)}",
                file=sys.stderr,
                flush=True,
            )
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise


def prepare_dinov3(args: argparse.Namespace, locations: dict[str, Path]) -> None:
    code = locations["dinov3_code"]
    if not code.exists():
        subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/facebookresearch/dinov3.git",
                str(code),
            ],
            check=True,
        )
    if not (code / "hubconf.py").is_file():
        raise ValueError(f"DINOv3 repository is missing hubconf.py: {code}")
    safe_link(ROOT / "dinov3", code)
    source = (
        Path(args.dinov3_weight_file).expanduser().resolve()
        if args.dinov3_weight_file
        else locations["dinov3_weights"] / "dinov3_vith16plus.pth"
    )
    if not source.is_file():
        if not args.dinov3_weight_url:
            raise ValueError(
                "DINOv3 weight missing; provide --dinov3-weight-file or --dinov3-weight-url after obtaining access"
            )
        download_url(args.dinov3_weight_url, source)
    safe_link(code / "dinov3_vith16plus.pth", source)


def prepare_blender(locations: dict[str, Path]) -> None:
    target = locations["blender"]
    if (target / "blender").is_file():
        return
    if target.exists() or target.is_symlink():
        raise ValueError(f"existing path blocks Blender extraction: {target}")
    archive = target.parent / "blender-4.5.3-linux-x64.tar.xz"
    if not archive.is_file() or archive.stat().st_size != BLENDER_BYTES:
        download_url(BLENDER_URL, archive)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=target.parent, prefix="blender-extract-"
    ) as directory:
        staging = Path(directory)
        with tarfile.open(archive, "r:xz") as tar:
            for member in tar.getmembers():
                destination = (staging / member.name).resolve()
                if not destination.is_relative_to(staging.resolve()):
                    raise ValueError(f"unsafe Blender archive member: {member.name}")
            tar.extractall(staging, filter="data")
        extracted = staging / "blender-4.5.3-linux-x64"
        if not (extracted / "blender").is_file():
            raise ValueError(
                "Blender extraction did not produce the expected executable"
            )
        extracted.rename(target)


def conda_executable() -> Path:
    existing = shutil.which("conda")
    if existing:
        return Path(existing)
    installer_root = ROOT / ".setup-tools" / "miniconda"
    conda = installer_root / "bin" / "conda"
    if conda.is_file():
        return conda
    installer = ROOT / ".setup-tools" / "Miniconda3-latest-Linux-x86_64.sh"
    installer.parent.mkdir(parents=True, exist_ok=True)
    download_url(
        "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh",
        installer,
    )
    subprocess.run(
        ["bash", str(installer), "-b", "-p", str(installer_root)], check=True
    )
    return conda


def prepare_environment(args: argparse.Namespace, locations: dict[str, Path]) -> Path:
    env_dir = locations["env"]
    python = env_dir / "bin" / "python"
    if args.skip_deps:
        if not python.is_file():
            raise ValueError("--skip-deps requires an existing .conda environment")
        return python
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
        raise ValueError("environment.yml requires Linux x86_64")
    if shutil.which("nvidia-smi") is None:
        raise ValueError(
            "environment setup requires an NVIDIA GPU and nvidia-smi; use --skip-deps only with a prepared environment"
        )
    conda = conda_executable()
    run_env = os.environ.copy()
    run_env.setdefault("CONDA_PKGS_DIRS", str(ROOT / ".setup-cache" / "conda"))
    run_env.setdefault("PIP_CACHE_DIR", str(ROOT / ".setup-cache" / "pip"))
    operation = "update" if python.is_file() else "create"
    command = [str(conda), "env", operation]
    if operation == "create":
        command.append("-y")
    command.extend(["-p", str(env_dir), "-f", str(ROOT / "environment.yml")])
    subprocess.run(
        command,
        check=True,
        env=run_env,
        cwd=ROOT,
    )
    hook = env_dir / "etc" / "conda" / "activate.d" / "zz_lsrm_env.sh"
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text(ACTIVATE_HOOK, encoding="utf-8")
    install_env = run_env | {
        "CONDA_PREFIX": str(env_dir),
        "PATH": f"{env_dir / 'bin'}:{conda.parent}:{run_env.get('PATH', '')}",
    }
    subprocess.run(
        [
            "bash", "-c",
            'set -e; source "$CONDA_PREFIX/etc/conda/activate.d/zz_lsrm_env.sh"; '
            '"$CONDA_PREFIX/bin/python" -',
        ],
        input=CUDA_CHECK,
        text=True,
        cwd=ROOT,
        check=True,
        env=install_env,
    )
    return python


def download_python() -> Path:
    """Keep download-only dependencies separate from the CUDA environment."""
    python = ROOT / ".setup-tools" / "download-venv" / "bin" / "python"
    if not python.is_file():
        subprocess.run(
            [sys.executable, "-m", "venv", str(python.parent.parent)], check=True
        )
    return python


def hub_download(
    items: list[dict[str, Any]], locations: dict[str, Path], python: Path
) -> None:
    if not items:
        return
    subprocess.run(
        [str(python), "-m", "pip", "install", "--no-cache-dir", "huggingface_hub"],
        check=True,
    )
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        category = str(item["source"]).split("/", 1)[0]
        groups.setdefault(category, []).append(item)
    for category, group in groups.items():
        asset_root = locations.get(category, ROOT)
        print(f"Downloading {category} assets ({len(group)} files)...", flush=True)
        env = os.environ.copy()
        env.setdefault("HF_HUB_CACHE", str(asset_root / ".hf-cache"))
        env.setdefault("HF_XET_CACHE", str(asset_root / ".hf-xet"))
        subprocess.run(
            [str(python), str(ROOT / "setup_dev.py"), "_download"],
            input=json.dumps(group),
            text=True,
            check=True,
            env=env,
            cwd=ROOT,
        )


def download_helper() -> None:
    from huggingface_hub import hf_hub_download

    items: list[dict[str, Any]] = json.load(sys.stdin)
    for index, item in enumerate(items, 1):
        target = Path(item["target"])
        if target.is_file() and target.stat().st_size == item["size"]:
            continue
        source = str(item["source"])
        category, _, remainder = source.partition("/")
        asset_root = (
            ROOT
            if category == "env.exr"
            else target.parents[len(Path(remainder).parts) - 1]
        )
        cache = Path(os.environ.get("HF_HUB_CACHE", str(asset_root / ".hf-cache")))
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{index}/{len(items)}] Downloading {source}", file=sys.stderr, flush=True)
        stop = threading.Event()

        def report_wait() -> None:
            started = time.monotonic()
            while not stop.wait(15):
                elapsed = int(time.monotonic() - started)
                print(
                    f"Still downloading {source} ({elapsed}s elapsed)...",
                    file=sys.stderr,
                    flush=True,
                )

        reporter = threading.Thread(target=report_wait, daemon=True)
        reporter.start()
        try:
            cached = Path(
                hf_hub_download(
                    repo_id=REPO,
                    filename=source,
                    cache_dir=cache,
                    endpoint=endpoint(),
                    token=os.environ.get("HF_TOKEN") or True,
                )
            )
        except Exception as error:
            raise ValueError(
                f"cannot download {source}; check repository access, HF_TOKEN, and HF_ENDPOINT: {error}"
            ) from error
        finally:
            stop.set()
            reporter.join()
        temporary = target.with_name(target.name + ".partial")
        temporary.unlink(missing_ok=True)
        try:
            try:
                os.link(cached.resolve(), temporary)
            except OSError:
                shutil.copyfile(cached, temporary)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        if not target.is_file() or target.stat().st_size != item["size"]:
            raise ValueError(f"download incomplete: {target}")
        print(f"[{index}/{len(items)}] Ready: {source}", file=sys.stderr, flush=True)


def verify(components: tuple[str, ...], locations: dict[str, Path]) -> bool:
    problems: list[str] = []
    python = locations["env"] / "bin" / "python"
    if not python.is_file():
        problems.append(str(python))
    for component in components:
        if component in ("rgb", "brdf"):
            base = ROOT / "checkpoints" / component
            for relative in (
                "dense/args.txt",
                "dense/checkpoints/last.pth",
                "sparse.pth",
            ):
                if not (base / relative).is_file():
                    problems.append(str(base / relative))
        elif component in ("gso", "orb", "dtc"):
            family = "rgb" if component == "gso" else "brdf"
            base = ROOT / "datasets" / family / f"{component}_example"
            scene_list = base / "test.txt"
            if not scene_list.is_file():
                problems.append(str(scene_list))
                continue
            scenes = [
                line.strip()
                for line in scene_list.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if not scenes:
                problems.append(f"empty scene list: {scene_list}")
            for scene in scenes:
                scene_root = base / scene
                if component == "gso":
                    required = [
                        scene_root / "images" / "CameraRig.json",
                        scene_root / "images" / "image_process_info.json",
                        scene_root / "scene" / "aria_trajectory.csv",
                        scene_root / "scene" / "scene_info.json",
                    ]
                    required += [
                        scene_root / "images" / "rgb" / f"{kind}{index:07d}.png"
                        for kind in ("rgb", "mask")
                        for index in range(28)
                    ]
                else:
                    required = [
                        scene_root / "transforms_input.json",
                        scene_root / "transforms_output.json",
                        scene_root / "scale_center.txt",
                    ]
                    required += [
                        scene_root / folder
                        for folder in ("input", "mask", "output", "mask_output", "env")
                    ]
                problems.extend(str(path) for path in required if not path.exists())
        elif component == "dinov3":
            if not (ROOT / "dinov3" / "hubconf.py").is_file():
                problems.append("dinov3/hubconf.py")
            if not (ROOT / "dinov3" / "dinov3_vith16plus.pth").is_file():
                problems.append("dinov3/dinov3_vith16plus.pth")
        elif (
            component == "blender" and not (locations["blender"] / "blender").is_file()
        ):
            problems.append(str(locations["blender"] / "blender"))
        elif component == "env" and not (ROOT / "env.exr").is_file():
            problems.append("env.exr")
    used_categories = set()
    if any(x in components for x in ("gso", "orb", "dtc")):
        used_categories.add("datasets")
    if any(x in components for x in ("rgb", "brdf")):
        used_categories.add("checkpoints")
    for name in used_categories:
        if (ROOT / name).resolve() != locations[name].resolve():
            problems.append(f"{name} points to the wrong location")
    if python.is_file():
        check = subprocess.run(
            [
                str(python),
                "-c",
                "import torch, nerfacc, flash_attn; from pytorch3d import _C",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if check.returncode:
            problems.append(
                "Python dependencies failed to import (torch, nerfacc, flash_attn, pytorch3d)"
            )
    if problems:
        print(
            "Missing or conflicting resources:\n  " + "\n  ".join(problems),
            file=sys.stderr,
        )
        return False
    print("Selected resources verified.")
    return True


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("command", choices=("plan", "download", "setup", "verify", "smoke"))
    cli.add_argument("--profile", choices=("smoke", "custom"), default="smoke")
    cli.add_argument("--components", help="Comma-separated: " + ",".join(COMPONENTS))
    cli.add_argument("--datasets-dir")
    cli.add_argument("--checkpoints-dir")
    cli.add_argument("--dinov3-weights-dir")
    cli.add_argument("--dinov3-weight-file")
    cli.add_argument("--dinov3-weight-url")
    cli.add_argument("--allow-large-downloads", action="store_true")
    cli.add_argument("--skip-deps", action="store_true")
    return cli


def main() -> int:
    # The download subprocess inherits .env values from its parent.
    # Its Python environment may not have dotenv installed.
    if len(sys.argv) > 1 and sys.argv[1] == "_download":
        download_helper()
        return 0
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    args = parser().parse_args()
    if args.dinov3_weight_file and args.dinov3_weight_url:
        raise ValueError("choose either --dinov3-weight-file or --dinov3-weight-url")
    components = selected(args)
    locations = paths(args)
    if args.command in ("plan", "download", "setup"):
        items = manifest(args, components)
        pending = show_plan(args, components, items)
        if args.command == "plan":
            return 0
        unknown_download = (
            "dinov3" in components
            and bool(args.dinov3_weight_url)
            and external_weight_size(args) is None
        )
        if (
            pending >= LARGE_LIMIT or unknown_download
        ) and not args.allow_large_downloads:
            raise ValueError(
                "more than 1 GB will be downloaded; rerun with --allow-large-downloads"
            )
        if (
            "dinov3" in components
            and not (locations["dinov3_weights"] / "dinov3_vith16plus.pth").is_file()
            and not args.dinov3_weight_file
            and not args.dinov3_weight_url
        ):
            raise ValueError(
                "DINOv3 weight unavailable; provide --dinov3-weight-file or --dinov3-weight-url"
            )
        if args.command == "setup" and not args.allow_large_downloads and not args.skip_deps:
            raise ValueError(
                "dependency download size is unknown; rerun with --allow-large-downloads or --skip-deps"
            )
        if args.command == "setup":
            print("Preparing Python environment...", flush=True)
            python = prepare_environment(args, locations)
        else:
            python = download_python() if items else Path(sys.executable)
        for category in ("datasets", "checkpoints"):
            if any(
                x in components
                for x in (
                    ("gso", "orb", "dtc") if category == "datasets" else ("rgb", "brdf")
                )
            ):
                locations[category].mkdir(parents=True, exist_ok=True)
                safe_link(ROOT / category, locations[category])
        print("Preparing Hugging Face assets...", flush=True)
        hub_download(items, locations, python)
        if "dinov3" in components:
            print("Preparing DINOv3...", flush=True)
            prepare_dinov3(args, locations)
        if "blender" in components:
            print("Preparing Blender...", flush=True)
            prepare_blender(locations)
        return 0 if args.command == "download" or verify(components, locations) else 1
    if not verify(components, locations):
        return 1
    if args.command == "smoke":
        if components != SMOKE:
            raise ValueError("smoke command requires the default smoke profile")
        gpu = subprocess.run(
            [
                str(locations["env"] / "bin" / "python"),
                "-c",
                "import torch; assert torch.cuda.is_available()",
            ],
            capture_output=True,
            check=False,
        )
        if gpu.returncode:
            raise ValueError("smoke inference requires an available NVIDIA CUDA GPU")
        env = os.environ.copy()
        env["CONDA_PREFIX"] = str(locations["env"])
        env["PATH"] = f"{locations['env'] / 'bin'}:{env.get('PATH', '')}"
        subprocess.run(
            ["bash", str(ROOT / "test_rgb.sh")], cwd=ROOT, env=env, check=True
        )
    return 0


def error_message(error: Exception) -> str:
    if isinstance(error, urllib.error.HTTPError):
        return f"{error} (URL: {error.geturl()})"
    return str(error)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        ValueError,
        OSError,
        urllib.error.HTTPError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"setup_dev: {error_message(error)}", file=sys.stderr)
        raise SystemExit(1) from None
