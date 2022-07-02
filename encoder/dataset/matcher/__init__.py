import re
import os
import sys
import cmake
import shutil
import subprocess
import importlib.util
from checksumdir import dirhash

# Set this if gcc is not the default compiler
# os.environ["CC"] = "/usr/bin/gcc-7"
# os.environ["CXX"] = "/usr/bin/g++-7"

_dir_path = str(os.path.dirname(os.path.abspath(__file__)))
_src_path = str(os.path.join(_dir_path, "matcher_src"))
_build_path = str(os.path.join(_dir_path, "build"))
sys.path.append(_src_path)

md5hash = dirhash(
    _src_path,
    "md5",
    excluded_extensions=["txt", "so"],
    excluded_files=["cmake-build-debug", "idea"],
)
build = True
if os.path.exists(os.path.join(_build_path, "hash.txt")):
    with open(os.path.join(_build_path, "hash.txt"), "r") as file:
        build = file.read() != md5hash
if build:
    shutil.rmtree(_build_path, ignore_errors=True)
    os.makedirs(_build_path)
    subprocess.call(
        [
            os.path.join(cmake.CMAKE_BIN_DIR, "cmake"),
            "-S",
            _src_path,
            "-B",
            _build_path,
        ]
    )
    subprocess.call(["make", "-C", _build_path, "clean"])
    if subprocess.call(["make", "-C", _build_path, "-j4"]) != 0:
        raise RuntimeError("Make failed")
    subprocess.call(["make", "-C", _build_path, "install"])
    with open(os.path.join(_build_path, "hash.txt"), "w") as file:
        file.write(md5hash)

_matcher = importlib.import_module("matcher")
KnowledgeBase = _matcher.KnowledgeBase
KnowledgeMatcher = _matcher.KnowledgeMatcher
ConceptNetReader = _matcher.ConceptNetReader
