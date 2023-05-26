from pathlib import Path
from urllib.parse import quote

import cbor2
import lz4.frame
import orjson

from .downloader import download_file

base_url = "https://ranxhub.s3.eu-central-1.amazonaws.com"


def home_path():
    p = Path(Path.home() / ".ranx")
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
    return p


def tmp_path():
    p = Path(Path.home(), ".ranx_tmp")
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)
    return p


def get_path(id: str):
    file_path = home_path() / f"{id}.rh"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    return file_path


def get_tmp_path(id: str):
    file_path = tmp_path() / f"{id}.rh"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    return file_path


def get_url(id: str):
    return f"{base_url}/{quote(id)}.rh"


def download(id: str):
    url = get_url(id)
    path = get_path(id)
    tmp_path = get_tmp_path(id)

    if not path.exists():
        download_file(url, tmp_path)
        tmp_path.rename(path)

    return load_lz4(path)


def save_lz4(content: dict, path: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with lz4.frame.open(path, mode="wb") as f:
        f.write(lz4.frame.compress(cbor2.dumps(content), compression_level=16))


def load_lz4(path: str) -> dict:
    with lz4.frame.open(path, mode="r") as f:
        content = cbor2.loads(lz4.frame.decompress(f.read()))

    return content


def save_json(x: dict, path: str) -> None:
    with open(path, "wb") as f:
        f.write(orjson.dumps(x, option=orjson.OPT_INDENT_2))


def load_json(path: str) -> None:
    return orjson.loads(open(path, "rb").read())
