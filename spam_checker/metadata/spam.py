from pathlib import Path

import spam_checker.metadata.shared as shared

RAW_DATA_DIRNAME = shared.DATA_DIRNAME / "raw" / "spam"
METADATA_FILENAME = RAW_DATA_DIRNAME / "metadata.toml"
DL_DATA_DIRNAME = shared.DATA_DIRNAME / "downloaded" / "spam"
PROCESSED_DATA_DIRNAME = shared.DATA_DIRNAME / "processed" / "spam"
PROCESSED_DATA_FILENAME = PROCESSED_DATA_DIRNAME / "byclass.h5"
