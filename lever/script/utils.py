##
from typing import List
from pathlib import Path
import yaml

colors = ["#1e2019", "#317b22", "#004ba8", "#ed254e", "#0094c6"]
color_dict = {"black": "#1e2019", "green": "#00BA38", "blue": "#619cFF", "red": "#F8766D", "lightblue": "#66B3BA",
              "purple": "#88498F"}

##
def print_stats_factory(log_file: Path):
    def print_stats(name: str, statements: List[str]):
        if log_file.exists():
            with log_file.open("r") as fp:
                stat_dict = yaml.load(fp, Loader=yaml.CLoader)
            stat_dict[name] = statements
        else:
            stat_dict = {name: statements}
        print('\n', name)
        for statement in statements:
            print('\t', statement)
        with log_file.open("w") as fp:
            yaml.dump(stat_dict, fp, Dumper=yaml.CDumper)
    return print_stats

##
