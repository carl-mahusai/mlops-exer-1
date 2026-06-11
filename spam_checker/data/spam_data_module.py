import toml
import os
import zipfile
import spam_checker.metadata.spam as metadata

from spam_checker import util
from spam_checker.data.base_data_module import _download_raw_dataset, BaseDataModule, load_and_print_info

RAW_DATA_DIRNAME = metadata.RAW_DATA_DIRNAME
METADATA_FILENAME = metadata.METADATA_FILENAME
DL_DATA_DIRNAME = metadata.DL_DATA_DIRNAME
PROCESSED_DATA_DIRNAME = metadata.PROCESSED_DATA_DIRNAME
PROCESSED_DATA_FILENAME = metadata.PROCESSED_DATA_FILENAME



class SPAMDataModule(BaseDataModule):
    """SPAM data module"""

    def __init__(self, args=None):
        super().__init__(args)

    def prepare_data(self, *args, **kwargs) -> None:
        if not os.path.exists(PROCESSED_DATA_FILENAME):
            _download_and_process_data()

def _download_and_process_data():
    metadata = toml.load(METADATA_FILENAME)
    print(metadata)
    _download_raw_dataset(metadata, DL_DATA_DIRNAME)
    _process_raw_dataset(metadata["filename"], DL_DATA_DIRNAME)

def _process_raw_dataset(filename: str, dirname: Path):
    print("Unzipping spam...")
    with util.temporary_working_directory(dirname):
        with zipfile.ZipFile(filename, "r") as zip_file:
            zip_file.extractall()