import os
import json

import h5py
import numpy as np

DATA_ROOT = "data"


def make_dirs_if_needed(dir_path: str) -> None:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_dir_files(dir_path: str) -> list:
    """
    Gets the names of the files that are direct children of `dir_path`.
    """
    return [
        fname
        for fname in os.listdir(dir_path)
        if os.path.isfile(os.path.join(dir_path, fname))
    ]


def ndarray_to_h5(data: np.ndarray, fname: str, overwrite: bool = False) -> None:
    if not overwrite and os.path.isfile(fname):
        return

    h5f = h5py.File(fname, "w")
    h5f.create_dataset("data", data=data)
    h5f.close()


def ndarray_from_h5(fname: str) -> np.ndarray:
    print(f"Loading ndarray at HDF5 path '{fname}'...")
    h5f = h5py.File(fname, "r")
    data = h5f["data"][:]
    h5f.close()
    return data


def to_h5(d: dict, fname: str, overwrite: bool = False) -> None:
    if not overwrite and os.path.isfile(fname):
        return

    # Make the directories along the file path if they don't already
    # exist.
    make_dirs_if_needed(os.path.dirname(fname))
    h5f = h5py.File(fname, "w")

    numpy_fields = []
    json_fields = []
    for field, data in d.items():
        if isinstance(data, np.ndarray):
            numpy_fields.append(field)
            h5f.create_dataset(field, data=data)
        else:
            json_fields.append(field)
            h5f.attrs[field] = json.dumps(data)

    h5f.attrs["numpy_fields"] = json.dumps(numpy_fields)
    h5f.attrs["json_fields"] = json.dumps(json_fields)
    h5f.close()


def from_h5(fname: str) -> dict:
    h5f = h5py.File(fname, "r")
    d = {}

    # Load any numpy arrays.
    numpy_fields = json.loads(h5f.attrs["numpy_fields"])
    for field in numpy_fields:
        d[field] = h5f[field][:]

    # Load any json serialized data.
    json_fields = json.loads(h5f.attrs["json_fields"])
    for field in json_fields:
        d[field] = json.loads(h5f.attrs[field])

    h5f.close()
    return d


def object_to_h5(
    obj: object, fname: str, fields: list, overwrite: bool = False,
) -> None:
    """
    Use to persist any of an object's numpy or json serializable data to disk.
    Persists all class members whose names are listed in `fields`. Saves
    data at path `fname`.
    """
    to_h5({field: getattr(obj, field) for field in fields}, fname, overwrite)


def object_from_h5(obj: object, fname: str) -> None:
    """
    Loads and sets data onto `obj` that was previously saved to file using
    `to_h5`.
    """
    obj_data = from_h5(fname)
    for field, data in obj_data.items():
        setattr(obj, field, data)


def get_data_path(path: str):
    return os.path.join(DATA_ROOT, path)
