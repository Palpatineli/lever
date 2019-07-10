from typing import Dict, List, Sequence
from dataclasses import dataclass
from pathlib import Path
import toml

@dataclass
class IterBase:
    def __iter__(self):
        return (v for k, v in self.__dict__.items() if not k.startswith("__"))

@dataclass
class Case(IterBase):
    id: str
    fov: int
    session: int
    name: str

@dataclass
class Item(IterBase):
    id: str
    session: int

def read_group(proj_folder: Path, name: str) -> Dict[str, List[Item]]:
    group_file = proj_folder.joinpath("data", "index", f"{name}.toml")
    return {k: [Item(**x) for x in items] for k, items in toml.loads(group_file.read_text()).items()}

def read_index(proj_folder: Path) -> List[Case]:
    index_file = proj_folder.joinpath("data", "index", "index.toml")
    return [Case(**x) for x in toml.loads(index_file.read_text())['recordings']]

def group(data: list, item_list: List[Case], group_dict: Dict[str, List[Item]]) -> List[tuple]:
    group_lookup = {(value.id, value.session): key for key, values in group_dict.items() for value in values}
    result = list()
    is_seq = isinstance(data[0], Sequence) and not isinstance(data[0], str)
    for datum, item in zip(data, item_list):
        group_name = group_lookup.get((item.id, item.session), None)
        if group_name is not None:
            if is_seq:
                result.append((*datum, f"{item.id}-{item.session}", group_name))
            else:
                result.append((datum, f"{item.id}-{item.session}", group_name))
    return result

def group_nested(data_list: List[list], item_list: List[Case], group_dict: Dict[str, List[Item]]) -> List[tuple]:
    group_lookup = {(value.id, value.session): key for key, values in group_dict.items() for value in values}
    result = list()
    is_seq = isinstance(data_list[0][0], Sequence) and not isinstance(data_list[0], str)
    for data, item in zip(data_list, item_list):
        group_name = group_lookup.get((item.id, item.session), None)
        if group_name is not None:
            for datum in data:
                if is_seq:
                    result.append((*datum, f"{item.id}", group_name))
                else:
                    result.append((datum, f"{item.id}", group_name))
    return result
