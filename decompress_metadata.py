import gzip
import json
import pydicom
from pydicom.dataset import Dataset
from typing import Any, Dict


def decompress_metadata(input_file: str) -> Dict:
    with gzip.open(input_file, "rt", encoding="utf-8") as gz_file:
        metadata_json = gz_file.read()
    return json.loads(metadata_json)


def convert_to_dicom_value(vr: str, value: Any) -> Any:
    if vr == "OW" or vr == "OB":
        if isinstance(value, str) and value.startswith("<Binary Data"):
            return b"\x00" * 10
        raise ValueError("Invalid binary data representation.")
    elif vr == "PN":
        return pydicom.valuerep.PersonName(value)
    elif vr in ["DS", "IS"]:
        if isinstance(value, list):
            return [float(v) if vr == "DS" else int(v) for v in value]
        return float(value) if vr == "DS" else int(value)
    elif vr == "SQ":
        if isinstance(value, list):
            return [Dataset(**item) for item in value]
        raise ValueError("Sequence data must be a list.")
    else:
        return value


def validate_metadata_with_dicom(dicom_file: str, metadata: Dict) -> Dict:
    dicom_data = pydicom.dcmread(dicom_file)
    filtered_metadata = {}

    for tag, value in metadata.items():
        tag_tuple = tuple(int(t, 16) for t in tag.strip("()").split(","))
        if tag_tuple in dicom_data:
            filtered_metadata[tag] = value
        else:
            print(f"Tag {tag} not found in DICOM file, skipping.")

    return filtered_metadata


def merge_metadata_to_dicom(dicom_file: str, metadata: Dict, output_file: str):
    try:
        dicom_data = pydicom.dcmread(dicom_file)

        for tag, meta in metadata.items():
            try:
                tag_tuple = tuple(int(t, 16) for t in tag.strip("()").split(","))
                if tag_tuple not in dicom_data:
                    print(f"Tag {tag} not found in DICOM file, skipping.")
                    continue

                vr = dicom_data[tag_tuple].VR
                value = meta["value"] if isinstance(meta, dict) else meta
                dicom_data[tag_tuple].value = convert_to_dicom_value(vr, value)
            except Exception as e:
                print(f"Failed to update tag {tag}: {e}")

        dicom_data.save_as(output_file)
        print(f"Updated DICOM file saved to {output_file}")

    except Exception as e:
        print(f"Error: Unable to merge metadata into DICOM file. Details: {e}")


if __name__ == "__main__":
    compressed_metadata_file = "./compressed_metadata.json.gz"

    original_dicom_file = "/home/juneha/data_compression/dicom/002_S_5018/ADNI_002_S_5018_MR_MPRAGE_br_raw_20131118125857454_29_S206236_I398680.dcm"

    updated_dicom_file = "./updated.dcm"

    metadata = decompress_metadata(compressed_metadata_file)

    filtered_metadata = validate_metadata_with_dicom(original_dicom_file, metadata)

    merge_metadata_to_dicom(original_dicom_file, filtered_metadata, updated_dicom_file)
