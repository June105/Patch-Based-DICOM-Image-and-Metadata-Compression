import pydicom
import gzip
import json
from typing import Any

def convert_to_serializable(value: Any) -> Any:
    if isinstance(value, pydicom.multival.MultiValue):
        return list(value)
    elif isinstance(value, pydicom.valuerep.PersonName):
        return str(value)
    elif isinstance(value, pydicom.sequence.Sequence):
        return [convert_to_serializable(item) for item in value]
    elif isinstance(value, pydicom.dataset.Dataset):
        return {str(elem.tag): {"name": elem.name, "value": convert_to_serializable(elem.value)} for elem in value}
    elif isinstance(value, (bytes, bytearray)):
        return f"<Binary Data of length {len(value)}>"
    elif isinstance(value, str):
        return value.strip()
    else:
        return value


def compress_dicom_metadata(input_file: str, output_file: str):
    try:
        # DICOM 파일 읽기
        dicom_data = pydicom.dcmread(input_file)

        # 메타데이터를 딕셔너리로 변환
        metadata = {}
        for elem in dicom_data.iterall():
            try:
                tag = str(elem.tag)
                name = elem.name
                value = convert_to_serializable(elem.value)

                metadata[tag] = {"name": name, "value": value}
            except Exception as e:
                metadata[tag] = {"name": "Unknown", "value": f"Error: {e}"}

        metadata_json = json.dumps(metadata, indent=2)

        with gzip.open(output_file, "wt", encoding="utf-8") as gz_file:
            gz_file.write(metadata_json)

        print(f"Compressed metadata saved to {output_file}")

    except Exception as e:
        print(f"Error: Unable to compress DICOM metadata. Details: {e}")

if __name__ == "__main__":
    dicom_file_path = "/home/juneha/data_compression/dicom/002_S_5018/ADNI_002_S_5018_MR_MPRAGE_br_raw_20131118125857454_29_S206236_I398680.dcm"
    compressed_output_path = "./compressed_metadata.json.gz"

    compress_dicom_metadata(dicom_file_path, compressed_output_path)

    file_path = "/home/juneha/data_compression/compressed_metadata.json.gz"
