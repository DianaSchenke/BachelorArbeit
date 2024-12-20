from datasets import load_dataset, Dataset
from utils.dataset_utils import create_base_dataset, clean_and_reformat_ultrachat


ds_base = Dataset.from_json("ultrachat_existent_material_release_230420.json")
data = ds_base["data"]
ds = clean_and_reformat_ultrachat(data)

create_base_dataset(ds, "assistance_on_existent_materials")

