import argparse
import io
import os
import sys
import urllib
import pydicom
from absl import app
from absl.flags import argparse_flags
import numpy as np
import tensorflow as tf
import tensorflow_compression as tfc
import struct
from scipy.ndimage import gaussian_filter
import cv2
import zlib

from devide_patches import divide_image_with_bb_in_center, calculate_overall_bounding_box, perform_segmentation, load_sam_model, filter_masks_by_tissue

URL_PREFIX = "https://storage.googleapis.com/tensorflow_compression/metagraphs"
METAGRAPH_CACHE = "/tmp/tfc_metagraphs"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_cached(filename):
    """Downloads and caches files from web storage."""
    pathname = os.path.join(METAGRAPH_CACHE, filename)
    try:
        with tf.io.gfile.GFile(pathname, "rb") as f:
            string = f.read()
    except tf.errors.NotFoundError:
        url = f"{URL_PREFIX}/{filename}"
        request = urllib.request.urlopen(url)
        try:
            string = request.read()
        finally:
            request.close()
        tf.io.gfile.makedirs(os.path.dirname(pathname))
        with tf.io.gfile.GFile(pathname, "wb") as f:
            f.write(string)
    return string

def instantiate_model_signature(model, signature, inputs=None, outputs=None):
  string = load_cached(model + ".metagraph")
  metagraph = tf.compat.v1.MetaGraphDef()
  metagraph.ParseFromString(string)
  wrapped_import = tf.compat.v1.wrap_function(
      lambda: tf.compat.v1.train.import_meta_graph(metagraph), [])
  graph = wrapped_import.graph
  if inputs is None:
    inputs = metagraph.signature_def[signature].inputs
    inputs = [graph.as_graph_element(inputs[k].name) for k in sorted(inputs)]
  else:
    inputs = [graph.as_graph_element(t) for t in inputs]
  if outputs is None:
    outputs = metagraph.signature_def[signature].outputs
    outputs = [graph.as_graph_element(outputs[k].name) for k in sorted(outputs)]
  else:
    outputs = [graph.as_graph_element(t) for t in outputs]
  return wrapped_import.prune(inputs, outputs)

# Compress

def read_dicom(filename):
    ds = pydicom.dcmread(filename)
    pixel_array = ds.pixel_array

    pixel_array = (pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array))
    pixel_array = (pixel_array * 255).astype(np.uint8)

    if pixel_array.ndim == 2:
        pixel_array = np.expand_dims(pixel_array, axis=-1)

    if pixel_array.shape[-1] == 1:
        pixel_array = np.repeat(pixel_array, 3, axis=-1)

    sam_model = load_sam_model()
    masks = perform_segmentation(sam_model, pixel_array)
    filtered_masks = filter_masks_by_tissue(masks)
    bounding_box = calculate_overall_bounding_box(filtered_masks)
    patches = divide_image_with_bb_in_center(pixel_array, bounding_box)
    del sam_model

    return patches, pixel_array.shape[:2]

def compress_bbox_and_shape(bbox, shape):
    """Compress bbox and shape into a compact binary format."""
    data = np.array(list(bbox) + list(shape), dtype=np.uint16).tobytes()
    return zlib.compress(data)

def compress(model, input_file, output_file, rd_parameter=None, rd_parameter_tolerance=None, target_bpp=None, bpp_strict=False):
    if not output_file:
        output_file = input_file + ".tfci"

    patches, image_shape = read_dicom(input_file)

    compressed_bitstream = b''

    for i, (patch, bbox) in enumerate(patches):
        if i == 4:
            patch = tf.expand_dims(patch, 0)
            compressed_bbox_and_shape = compress_bbox_and_shape(bbox, image_shape)
            compressed_bitstream += struct.pack("I", len(compressed_bbox_and_shape))
            compressed_bitstream += compressed_bbox_and_shape
            bitstring = compress_patch(model, patch, rd_parameter, rd_parameter_tolerance, target_bpp, bpp_strict)
            compressed_bitstream += struct.pack("I", len(bitstring))
            compressed_bitstream += bitstring

    compressed_bitstream = zlib.compress(compressed_bitstream)

    with tf.io.gfile.GFile(output_file, "wb") as f:
        f.write(compressed_bitstream)

def compress_image(model, input_image, rd_parameter=None):
    sender = instantiate_model_signature(model, "sender")
    if len(sender.inputs) == 1:
        if rd_parameter is not None:
            raise ValueError("This model doesn't expect an RD parameter.")

        tensors = sender(input_image)
    elif len(sender.inputs) == 2:
        if rd_parameter is None:
            raise ValueError("This model expects an RD parameter.")
        rd_parameter = tf.constant(rd_parameter, dtype=sender.inputs[1].dtype)

        tensors = sender(input_image, rd_parameter)

        for i, t in enumerate(tensors):
            if t.dtype.is_floating and t.shape.rank == 0:
                tensors[i] = tf.expand_dims(t, 0)
    else:
        raise RuntimeError("Unexpected model signature.")

    packed = tfc.PackedTensors()
    packed.model = model
    packed.pack(tensors)
    return packed.string

def compress_patch(model, input_image, rd_parameter=None, rd_parameter_tolerance=None, target_bpp=None, bpp_strict=False):
    num_pixels = input_image.shape[-2] * input_image.shape[-3]

    if not target_bpp:
        bitstring = compress_image(model, input_image, rd_parameter=rd_parameter)
    else:
        models = load_cached(model + ".models")
        models = models.decode("ascii").split()

        try:
            lower, upper = [float(m) for m in models]
            use_rd_parameter = True
        except ValueError:
            lower = -1
            upper = len(models)
            use_rd_parameter = False

        bpp = None
        best_bitstring = None
        best_bpp = None
        while bpp != target_bpp:
            if use_rd_parameter:
                if upper - lower <= rd_parameter_tolerance:
                    break
                i = (upper + lower) / 2
                bitstring = compress_image(model, input_image, rd_parameter=i)
            else:
                if upper - lower < 2:
                    break
                i = (upper + lower) // 2
                bitstring = compress_image(models[i], input_image)
            bpp = 8 * len(bitstring) / num_pixels
            is_admissible = bpp <= target_bpp or not bpp_strict
            is_better = (best_bpp is None or
                         abs(bpp - target_bpp) < abs(best_bpp - target_bpp))
            if is_admissible and is_better:
                best_bitstring = bitstring
                best_bpp = bpp
            if bpp < target_bpp:
                lower = i
            if bpp > target_bpp:
                upper = i
        if best_bpp is None:
            assert bpp_strict
            raise RuntimeError(
                "Could not compress image to less than {} bpp.".format(target_bpp))
        bitstring = best_bitstring

    return bitstring

def compress_folder(model, input_folder, output_folder,
                    rd_parameter=None, rd_parameter_tolerance=None,
                    target_bpp=None, bpp_strict=False):
    """Compresses all DICOM files in a folder to a specified output folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        input_file = os.path.join(input_folder, filename)
        if not input_file.endswith(".dcm"):
            continue

        output_file = os.path.join(output_folder)
        try:
            print(f"Compressing {input_file} to {output_file}")
            compress(model, input_file, output_file, rd_parameter, rd_parameter_tolerance, target_bpp, bpp_strict)
        except Exception as e:
            print(f"Error compressing {input_file}: {e}")
            continue

# Decompress

def write_dicom(filename, image, original_dicom):
    original_dicom.PixelData = image.tobytes()
    original_dicom.Rows, original_dicom.Columns = image.shape

    original_dicom.save_as(filename)

def decompress_bbox_and_shape(compressed_data):
    """Decompress bbox and shape from binary format."""
    data = zlib.decompress(compressed_data)
    array = np.frombuffer(data, dtype=np.uint16)
    bbox = tuple(array[:4])
    shape = tuple(array[4:])
    return bbox, shape

def decompress_patch(bitstream):
    packed = tfc.PackedTensors(bitstream)
    receiver = instantiate_model_signature(packed.model, "receiver")
    tensors = packed.unpack([t.dtype for t in receiver.inputs])
    for i, t in enumerate(tensors):
        if t.dtype.is_floating and t.shape == (1,):
            tensors[i] = tf.squeeze(t, 0)
    output_image, = receiver(*tensors)

    return output_image

def decompress(input_file, output_file, original_dicom_file):
    if not output_file:
        output_file = input_file + ".dcm"

    with tf.io.gfile.GFile(input_file, "rb") as f:
        compressed_bitstream = f.read()

    compressed_bitstream = zlib.decompress(compressed_bitstream)

    offset = 0

    bbox_size = struct.calcsize("I")
    compressed_bbox_and_shape_size = struct.unpack("I", compressed_bitstream[offset:offset + bbox_size])[0]
    offset += bbox_size

    compressed_bbox_and_shape = compressed_bitstream[offset:offset + compressed_bbox_and_shape_size]
    offset += compressed_bbox_and_shape_size

    bbox, shape = decompress_bbox_and_shape(compressed_bbox_and_shape)

    original_dicom = pydicom.dcmread(original_dicom_file)
    original_min, original_max = np.min(original_dicom.pixel_array), np.max(original_dicom.pixel_array)
    reconstructed_image = np.zeros(shape, dtype=np.float32)

    while offset < len(compressed_bitstream):
        bitstream_length = struct.unpack("I", compressed_bitstream[offset:offset + 4])[0]
        offset += 4
        patch_bitstream = compressed_bitstream[offset:offset + bitstream_length]
        offset += bitstream_length

        decompressed_patch = decompress_patch(patch_bitstream)
        image_patch = tf.squeeze(decompressed_patch, 0).numpy()
        image_patch = np.mean(image_patch, axis=-1)

        x_min, y_min, x_max, y_max = bbox
        reconstructed_image[y_min:y_max, x_min:x_max] = image_patch

    reconstructed_image = (reconstructed_image / 255.0) * (original_max - original_min) + original_min
    reconstructed_image = gaussian_filter(reconstructed_image, sigma=0.5)
    reconstructed_image = np.clip(reconstructed_image, original_min, original_max).astype(original_dicom.pixel_array.dtype)

    write_dicom(output_file, reconstructed_image, original_dicom)

def decompress_folder(input_folder, output_folder, original_dicom_folder):
    """Decompresses all TFCI files in a folder to the specified output folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        input_file = os.path.join(input_folder, filename)
        if not input_file.endswith(".tfci"):
            continue

        original_dicom_file = os.path.join(original_dicom_folder, filename.replace(".tfci", ".dcm"))
        output_file = os.path.join(output_folder, filename.replace(".tfci", ".dcm"))
        try:
            print(f"Decompressing {input_file} to {output_file}")
            decompress(input_file, output_file, original_dicom_file)
        except Exception as e:
            print(f"Error compressing {input_file}: {e}")
            continue

def list_models():
  """Lists available models in web storage with a description."""
  url = URL_PREFIX + "/models.txt"
  request = urllib.request.urlopen(url)
  try:
    print(request.read().decode("utf-8"))
  finally:
    request.close()

def list_tensors(model):
  """Lists all internal tensors of a given model."""
  def get_names_dtypes_shapes(function):
    for op in function.graph.get_operations():
      for tensor in op.outputs:
        yield tensor.name, tensor.dtype.name, tensor.shape

  sender = instantiate_model_signature(model, "sender")
  tensors = sorted(get_names_dtypes_shapes(sender))
  print("Sender-side tensors:")
  for name, dtype, shape in tensors:
    print(f"{name} (dtype={dtype}, shape={shape})")
  print()

  receiver = instantiate_model_signature(model, "receiver")
  tensors = sorted(get_names_dtypes_shapes(receiver))
  print("Receiver-side tensors:")
  for name, dtype, shape in tensors:
    print(f"{name} (dtype={dtype}, shape={shape})")

def dump_tensor(model, tensors, input_file, output_file):
  """Dumps the given tensors of a model in .npz format."""
  if not output_file:
    output_file = input_file + ".npz"
  sender = instantiate_model_signature(model, "sender", outputs=tensors)
  input_image = read_png(input_file)
  table = str.maketrans(r"^./-:", r"_____")
  tensors = [t.translate(table) for t in tensors]
  values = [t.numpy() for t in sender(input_image)]
  assert len(tensors) == len(values)
  with io.BytesIO() as buf:
    np.savez(buf, **dict(zip(tensors, values)))
    with tf.io.gfile.GFile(output_file, mode="wb") as f:
      f.write(buf.getvalue())

def parse_args(argv):
    """Parses command line arguments."""
    parser = argparse_flags.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--url_prefix",
        default=URL_PREFIX,
        help="URL prefix for downloading model metagraphs.")
    parser.add_argument(
        "--metagraph_cache",
        default=METAGRAPH_CACHE,
        help="Directory where to cache model metagraphs.")
    subparsers = parser.add_subparsers(
        title="commands", dest="command",
        help="Invoke '<command> -h' for more information.")


    compress_cmd = subparsers.add_parser(
        "compress",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Reads a DICOM file or folder, compresses it using the given model, and "
                    "writes a TFCI file or folder.")
    compress_cmd.add_argument(
        "model",
        help="Unique model identifier.")
    compress_cmd.add_argument(
        "input_path",
        help="Input file or folder.")
    compress_cmd.add_argument(
        "output_path",
        help="Output file or folder.")
    compress_cmd.add_argument(
        "--rd_parameter", "-r", type=float,
        help="Rate-distortion parameter (for some models). Ignored if "
             "'target_bpp' is set.")
    compress_cmd.add_argument(
        "--rd_parameter_tolerance", type=float,
        default=2 ** -4,
        help="Tolerance for rate-distortion parameter.")
    compress_cmd.add_argument(
        "--target_bpp", "-b", type=float,
        help="Target bits per pixel.")
    compress_cmd.add_argument(
        "--bpp_strict", action="store_true",
        help="Try never to exceed 'target_bpp'. Ignored if 'target_bpp' is not set.")

    decompress_cmd = subparsers.add_parser(
        "decompress",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Reads a TFCI file or folder, reconstructs the image and writes back a DICOM file or folder.")
    decompress_cmd.add_argument(
        "input_path",
        help="Input file or folder.")
    decompress_cmd.add_argument(
        "output_path",
        help="Output file or folder.")
    decompress_cmd.add_argument(
        "--original_dicom_folder", required=True,
        help="Folder containing the original DICOM files to retain the metadata.")

    args = parser.parse_args(argv[1:])
    if args.command is None:
        parser.print_usage()
        sys.exit(2)
    return args

def main(args):
    global URL_PREFIX, METAGRAPH_CACHE
    URL_PREFIX = args.url_prefix
    METAGRAPH_CACHE = args.metagraph_cache

    if args.command == "compress":
        if os.path.isdir(args.input_path):
            compress_folder(args.model, args.input_path, args.output_path,
                            args.rd_parameter, args.rd_parameter_tolerance,
                            args.target_bpp, args.bpp_strict)
        else:
            compress(args.model, args.input_path, args.output_path,
                     args.rd_parameter, args.rd_parameter_tolerance,
                     args.target_bpp, args.bpp_strict)
    elif args.command == "decompress":
        if os.path.isdir(args.input_path):
            decompress_folder(args.input_path, args.output_path, args.original_dicom_folder)
        else:
            decompress(args.input_path, args.output_path, args.original_dicom_folder)


if __name__ == "__main__":
    app.run(main, flags_parser=parse_args)
