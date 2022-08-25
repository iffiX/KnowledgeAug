import os
import json
import pickle
import functools
import pathlib
import gzip
import zipfile
import shutil
import logging
import requests
import torch
from tqdm.auto import tqdm


def open_file_with_create_directories(path, mode):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, mode=mode)


def download_to(url, path):
    # https://stackoverflow.com/questions/
    # 37573483/progress-bar-while-download-file-over-http-with-requests
    r = requests.get(url, stream=True, allow_redirects=True)
    if r.status_code != 200:
        r.raise_for_status()  # Will only raise for 4xx codes, so...
        raise RuntimeError(f"Request to {url} returned status code {r.status_code}")
    file_size = int(r.headers.get("Content-Length", 0))

    path = pathlib.Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    desc = "(Unknown total file size)" if file_size == 0 else ""
    r.raw.read = functools.partial(
        r.raw.read, decode_content=True
    )  # Decompress if needed
    with tqdm.wrapattr(r.raw, "read", total=file_size, desc=desc) as r_raw:
        with path.open("wb") as f:
            shutil.copyfileobj(r_raw, f)


def decompress_gz(path_from, path_to):
    with gzip.open(path_from, "rb") as f_in:
        with open(path_to, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def decompress_zip(path_from, dir_path_to):
    with zipfile.ZipFile(path_from, "r") as zip_ref:
        zip_ref.extractall(dir_path_to)


def decompress_tar_gz(path_from, dir_path_to):
    shutil.unpack_archive(path_from, dir_path_to, "gztar")


class SafeCacheBase:
    def __init__(
        self,
        path,
        generate_func,
        generate_args=(),
        generate_kwargs=None,
        validate_func=None,
    ):
        self.path = path
        self.generate_func = generate_func
        self.generate_kwargs = generate_kwargs or {}
        self.generate_args = generate_args
        self.validate_func = validate_func or (lambda x: True)
        self._is_data_updated = False
        self._data = None

    def __enter__(self):
        data = self.load(self.path, self.validate_func)
        if data is None:
            data = self.generate_func(*self.generate_args, **self.generate_kwargs)
            self.save(data, self.path)
        self._data = data
        return self

    def __exit__(self, type, value, traceback):
        if self._is_data_updated:
            self.save(self._data, self.path)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, new_data):
        self._data = new_data
        self._is_data_updated = True

    def load(self, path, validate_func):
        """
        Subclass should override this function, and return None
        if any exception happens while loading the cache.

        Args:
            path: Cache file path
            validate_func: A function used to validate the loaded data

        Returns:
            Data
        """
        pass

    def save(self, data, path):
        """
        Subclass should override this function, save data to the cache file
        using the given path

        Args:
            data: Saved data
            path: Cache file path
        """
        pass


class JSONCache(SafeCacheBase):
    def load(self, path, validate_func):
        try:
            with open(path, "r") as file:
                data = json.load(file)
        except:
            return None
        if not validate_func(data):
            return None
        return data

    def save(self, data, path):
        with open(path, "w") as file:
            json.dump(data, file, indent=2)


class JSONStreamCache(SafeCacheBase):
    def __init__(self, path, generate_ids, generate_from_id_func, validate_func=None):
        self.path = path
        self.generate_ids = set(generate_ids)
        self.generate_from_id_func = generate_from_id_func
        self.generated_partial_data = {}
        super(JSONStreamCache, self).__init__(
            path=path, generate_func=self.stream_generate, validate_func=validate_func
        )

    def stream_generate(self):
        with open(self.path, "a", buffering=1 << 20, encoding="utf-8") as f:
            for id_ in tqdm(
                list(
                    self.generate_ids.difference(
                        set(self.generated_partial_data.keys())
                    )
                )
            ):
                generated_data = self.generate_from_id_func(id_)
                f.write(
                    json.dumps({"id": id_, "data": generated_data}, ensure_ascii=False,)
                    + "\n"
                )
                self.generated_partial_data[id_] = generated_data
        full_data, self.generated_partial_data = self.generated_partial_data, {}
        return full_data

    def load(self, path, validate_func):
        # if all data are generated, return full data,
        # otherwise set partially generated data and return None
        if os.path.exists(path):
            with open(path, "r") as f:
                generated_raw_data = f.read()
            results = [
                json.loads(line)
                for line in generated_raw_data.splitlines()
                if line.strip()
            ]
            generated_data = {r["id"]: r["data"] for r in results}
            processed_ids = {r["id"] for r in results}

            remain_num = len(self.generate_ids.difference(set(generated_data.keys())))
            logging.info(
                f"Loaded {len(generated_data)} generated data, "
                f"remaining {remain_num}/{len(self.generate_ids)} to generate."
            )

            if self.generate_ids == processed_ids:
                if validate_func(generated_data):
                    return generated_data
            else:
                self.generated_partial_data = generated_data

        return None


class PickleCache(SafeCacheBase):
    def load(self, path, validate_func):
        try:
            with open(path, "rb") as file:
                data = pickle.load(file)
        except:
            return None
        if not validate_func(data):
            return None
        return data

    def save(self, data, path):
        with open(path, "wb") as file:
            pickle.dump(data, file)


class TorchCache(SafeCacheBase):
    def load(self, path, validate_func):
        try:
            with open(path, "rb") as file:
                data = torch.load(file)
        except:
            return None
        if not validate_func(data):
            return None
        return data

    def save(self, data, path):
        with open(path, "wb") as file:
            torch.save(data, file)
