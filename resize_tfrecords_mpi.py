#!/usr/bin/env python3
#
# Copyright (c) 2018 Dell Inc., or its subsidiaries. All Rights Reserved.
#
# Written by Claudio Fahey <claudio.fahey@dell.com>
#


"""
This script reads images from TFRecord files, resizes all images, and writes them to new TFRecord files.
"""

import os
import argparse
from os.path import join, basename, splitext
import tensorflow as tf
import six
from glob import glob


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  if six.PY3 and isinstance(value, six.text_type):
    value = six.binary_type(value, encoding='utf-8')
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
  """Wrapper for inserting float features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _convert_to_example(filename, image_buffer, label, synset, human, xmin, ymin, xmax, ymax,
                        height, width, output_format):
  """Build an Example proto for an example.

  Args:
    filename: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
    label: integer, identifier for the ground truth for the network
    synset: string, unique WordNet ID specifying the label, e.g., 'n02323233'
    human: string, human-readable label, e.g., 'red fox, Vulpes vulpes'
    xmin, ymin, xmax, ymax: list of bounding boxes
    height: integer, image height in pixels
    width: integer, image width in pixels
  Returns:
    Example proto
  """
  colorspace = 'RGB'
  channels = 3
  image_format = output_format

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/colorspace': _bytes_feature(colorspace),
      'image/channels': _int64_feature(channels),
      'image/class/label': _int64_feature(label),
      'image/class/synset': _bytes_feature(synset),
      'image/class/text': _bytes_feature(human),
      'image/object/bbox/xmin': _float_feature(xmin),
      'image/object/bbox/xmax': _float_feature(xmax),
      'image/object/bbox/ymin': _float_feature(ymin),
      'image/object/bbox/ymax': _float_feature(ymax),
      'image/object/bbox/label': _int64_feature([label] * len(xmin)),
      'image/format': _bytes_feature(image_format),
      'image/filename': _bytes_feature(os.path.basename(filename)),
      'image/encoded': _bytes_feature(image_buffer)}))
  return example


def process_tf_record_file(input_tf_record_filename, output_tf_record_filename, args):
    """Read single TFRecord file, resize images, and write a new TFRecord file.

    Note that bounding box values are floats between 0 and 1 and do not need to be scaled.
    """
    tf_record_iterator = tf.python_io.tf_record_iterator(path=input_tf_record_filename)
    original_len_total = 0
    resized_len_total = 0
    image_count = 0
    with tf.python_io.TFRecordWriter(output_tf_record_filename) as writer:
        for record_string in tf_record_iterator:
            image_count += 1

            # Parse input record.
            example = tf.train.Example()
            example.ParseFromString(record_string)
            filename = example.features.feature['image/filename'].bytes_list.value[0]
            input_format = example.features.feature['image/format'].bytes_list.value[0].decode()
            label = int(example.features.feature['image/class/label'].int64_list.value[0])
            synset = example.features.feature['image/class/synset'].bytes_list.value[0]
            human = example.features.feature['image/class/text'].bytes_list.value[0]
            xmin = list(example.features.feature['image/object/bbox/xmin'].float_list.value)
            ymin = list(example.features.feature['image/object/bbox/ymin'].float_list.value)
            xmax = list(example.features.feature['image/object/bbox/xmax'].float_list.value)
            ymax = list(example.features.feature['image/object/bbox/ymax'].float_list.value)
            original_height = int(example.features.feature['image/height'].int64_list.value[0])
            original_width = int(example.features.feature['image/width'].int64_list.value[0])
            original_encoded = example.features.feature['image/encoded'].bytes_list.value[0]
            original_len = len(original_encoded)
            original_len_total += original_len

            # with open("/imagenet-scratch/in.jpg", "wb") as output_jpeg_file:
            #     output_jpeg_file.write(encoded)

            # Decode JPEG.
            assert input_format == 'JPEG'
            num_components = 3
            image = tf.image.decode_jpeg(original_encoded, channels=num_components)

            # Validate JPEG dimensions.
            original_shape = tf.shape(image).eval()
            assert original_shape[1] == original_width
            assert original_shape[0] == original_height
            assert original_shape[2] == num_components

            resize_factor = args.resize_factor
            if resize_factor == 1.0:
                new_height = original_height
                new_width = original_width
                resized_image = image
            else:
                # Calculate new image size.
                # We must avoid making the image too large.
                # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/lib/jpeg/jpeg_mem.cc#L181
                total_size_original = original_height * original_width * num_components
                # print("input_tf_record_filename=%s, original_shape=%s, total_size_original=%g" % (input_tf_record_filename, str(original_shape), total_size_original))
                total_size_max = 2**29 - 1  # 536 MB uncompressed
                max_resize_factor = (total_size_max / total_size_original)**0.5

                if resize_factor > max_resize_factor:
                    print('New image increased by factor of only %0.3f to avoid max image size; original is %dx%dx%d' %
                          (max_resize_factor, original_height, original_width, num_components))
                    resize_factor = max_resize_factor
                new_height = int(original_height * resize_factor)
                new_width = int(original_width * resize_factor)
                total_size = new_height * new_width * num_components
                # print("input_tf_record_filename=%s, total_size=%g" % (input_tf_record_filename, total_size))
                assert total_size < total_size_max, "total_size=%d, original_shape=%s, resize_factor=%f, input_tf_record_filename=%s" % (
                    total_size, str(original_shape), resize_factor, input_tf_record_filename)
                # Resize image.
                resized_image = tf.image.resize_images(image, [new_height, new_width], align_corners=True)

            # Encode image.
            if args.output_format == 'JPEG':
                resized_encoded = tf.image.encode_jpeg(
                    tf.cast(resized_image, tf.uint8),
                    quality=100,
                    chroma_downsampling=False,
                )
            elif args.output_format == 'PNG':
                resized_encoded = tf.image.encode_png(
                    tf.cast(resized_image, tf.uint8),
                    compression=0,
                )
            else:
                raise Exception('Unsupported output_format')

            resized_encoded = resized_encoded.eval()
            resized_len = len(resized_encoded)
            resized_len_total += resized_len

            print('%(input_tf_record_filename)s: %(filename)s %(original_KB)0.0f KB => %(resized_KB)0.0f KB' % dict(
                input_tf_record_filename=input_tf_record_filename,
                xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax,
                filename=filename.decode(), original_height=original_height, original_width=original_width,
                original_KB=original_len/1000.0, resized_KB=resized_len/1000.0))

            # with open("/imagenet-scratch/out.jpg", "wb") as output_jpeg_file:
            #     output_jpeg_file.write(resized_encoded)

            # Write to TFRecord.
            example = _convert_to_example(
                filename, resized_encoded, label,
                synset, human, xmin, ymin, xmax, ymax,
                new_height, new_width, args.output_format)
            writer.write(example.SerializeToString())

            if 0 < args.max_image_count <= image_count:
                break

    original_len_mean = original_len_total / image_count
    resized_len_mean = resized_len_total / image_count
    print('%(input_tf_record_filename)s: %(image_count)d images, mean size %(original_KB_mean)0.0f KB => %(resized_KB_mean)0.0f KB' % dict(
        input_tf_record_filename=input_tf_record_filename,
        image_count=image_count,
        original_KB_mean=original_len_mean/1000,
        resized_KB_mean=resized_len_mean / 1000))


def worker(rank, size, input_files, output_dir, args):
    with tf.Session():
        input_tf_record_filenames = sorted(glob(input_files))
        num_files = len(input_tf_record_filenames)
        i = rank
        while i < num_files:
            input_tf_record_filename = input_tf_record_filenames[i]
            output_tf_record_filename = join(output_dir, basename(input_tf_record_filename))
            print(rank, input_tf_record_filename, output_tf_record_filename)
            process_tf_record_file(input_tf_record_filename, output_tf_record_filename, args)
            i += size


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', '--input_files', help='Input files', required=True)
    parser.add_argument('-o', '--output_dir', help='Output directory', required=True)
    parser.add_argument('--output_format', default='JPEG', help='JPEG or PNG')
    parser.add_argument('--max_image_count', type=int, default=0)
    parser.add_argument('--resize_factor', type=float, default=3.0, help='Resize factor for each image dimension.')
    args = parser.parse_args()
    rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', '0'))
    size = int(os.environ.get('OMPI_COMM_WORLD_SIZE', '1'))
    worker(rank, size, args.input_files, args.output_dir, args)


if __name__ == '__main__':
    main()
