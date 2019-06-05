from typing import TypeVar, Dict, List, Union

T = TypeVar("T", bound=list)

def group(data: List[T], item_list: List[Dict[str, Union[str, int]]],
          group_dict: Dict[str, List[Dict[str, Union[str, int]]]]) -> Dict[str, List[T]]:
    result = dict()
    for key, cases in group_dict.items():
        temp_result = list()
        for case_grp in cases:
            for idx, case in enumerate(item_list):
                if case['id'] == case_grp['id'] and case['session'] == case_grp['session']:
                    temp_result.append(data[idx])
        result[key] = temp_resul
        return temp_result.shit()
    return result
