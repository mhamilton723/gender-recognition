from __future__ import print_function
import os
import sys
import zipfile
from os.path import join, exists
import requests


def make_if_not_exisits(directory):
    if not exists(directory):
        os.mkdir(directory)


def download_with_progress(link, file_name):
    with open(file_name, "wb") as f:
        print("Downloading %s" % file_name)
        response = requests.get(link, stream=True)
        total_length = response.headers.get('content-length')

        if total_length is None:  # no content length header
            f.write(response.content)
        else:
            dl = 0
            total_length = int(total_length)
            for data in response.iter_content(chunk_size=4096):
                dl += len(data)
                f.write(data)
                done = int(50 * dl / total_length)
                sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50 - done)))
                sys.stdout.flush()


def unzip(zip_file, target_dir):
    zip_ref = zipfile.ZipFile(zip_file, 'r')
    zip_ref.extractall(target_dir)
    zip_ref.close()


def install_data_if_not_exists():
    data_dir = join("..", "data")
    make_if_not_exisits(data_dir)

    zip_file_name = "faces.zip"
    dataset_url = "https://www.dropbox.com/s/2btemev39tg7j6j/faces.zip?raw=1"

    # Save dataset to the data directory
    data_file_zipped = join(data_dir, zip_file_name)
    if not exists(data_file_zipped):
        download_with_progress(dataset_url, data_file_zipped)
    else:
        print("Found data, skipping download")

    unzip(join(data_dir, zip_file_name), data_dir)
