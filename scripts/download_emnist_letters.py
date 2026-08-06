import argparse
import gzip
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from app.utils.config import PROJECT_ROOT, emnist_training_settings

FILENAMES = (
    "emnist-letters-train-images-idx3-ubyte.gz",
    "emnist-letters-train-labels-idx1-ubyte.gz",
    "emnist-letters-test-images-idx3-ubyte.gz",
    "emnist-letters-test-labels-idx1-ubyte.gz",
)
OFFICIAL_ARCHIVE_URL = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"


def is_gzip_file(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 2:
        return False
    with path.open("rb") as input_file:
        return input_file.read(2) == b"\x1f\x8b"


def copy_as_gzip(source: Path, target: Path) -> None:
    if is_gzip_file(source):
        shutil.copy2(source, target)
        return
    with source.open("rb") as input_file, gzip.open(target, "wb") as output_file:
        shutil.copyfileobj(input_file, output_file)


def find_cached_file(filename: str) -> Path | None:
    search_root = PROJECT_ROOT / "data" / "external" / "emnist"
    candidates = [
        path
        for path in search_root.rglob(f"*{filename}")
        if path.parent != emnist_training_settings.data_root and path.is_file()
    ]
    return max(candidates, key=lambda path: path.stat().st_size) if candidates else None


def provision_files(*, allow_download: bool = True) -> list[Path]:
    destination = emnist_training_settings.data_root
    destination.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    resolved: list[Path] = []
    for filename in FILENAMES:
        target = destination / filename
        if is_gzip_file(target):
            resolved.append(target)
            continue
        cached = find_cached_file(filename)
        if cached is None:
            missing.append(filename)
            continue
        copy_as_gzip(cached, target)
        resolved.append(target)
        print(f"Copied cached official file: {cached} -> {target}")
    if not missing:
        return sorted(resolved)
    if not allow_download:
        raise FileNotFoundError(f"Missing official EMNIST files: {', '.join(missing)}")

    with tempfile.TemporaryDirectory() as temporary_directory:
        archive_path = Path(temporary_directory) / "gzip.zip"
        print(f"Downloading official EMNIST archive: {OFFICIAL_ARCHIVE_URL}")
        urllib.request.urlretrieve(OFFICIAL_ARCHIVE_URL, archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            members = {Path(name).name: name for name in archive.namelist()}
            for filename in missing:
                member = members.get(filename)
                if member is None:
                    raise FileNotFoundError(f"Official archive does not contain {filename}")
                target = destination / filename
                with archive.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                resolved.append(target)
    return sorted(resolved)


def main() -> None:
    parser = argparse.ArgumentParser(description="Provision official EMNIST Letters IDX/GZIP files")
    parser.add_argument("--offline", action="store_true", help="Use local cache only")
    args = parser.parse_args()
    paths = provision_files(allow_download=not args.offline)
    for path in paths:
        print(f"Ready: {path} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
