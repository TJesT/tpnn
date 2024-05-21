import re
from pathlib import Path

src_folder = Path("./data/mushroom")
dst_folder = Path("./data")

data = "agaricus-lepiota.data"
names_desc = "agaricus-lepiota.names"

result_name = "mushrooms.csv"

label_pattern = r"[0-9][0-9]*\. ([a-z]+(\-?[a-z]*)*)"

with open(src_folder / names_desc) as names:
    labels = [match[0] for match in re.findall(label_pattern, names.read())]

print(f"Found labels:", end="")
print("", *labels, sep="\n - ")

with open(src_folder / data) as data_file:
    print(f"Writing labels with data to {dst_folder / result_name!s}...")
    with open(dst_folder / result_name, "w") as out_file:
        out_file.write(",".join(["poisonous(target)"] + labels))
        out_file.write("\n")
        out_file.write(data_file.read())

print("Done!")
