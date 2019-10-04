import urllib.request
import zipfile
import os
from glob import glob
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    print("[!] download data file")
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def unzip_zip_file(zip_path, data_path):
    zip_ref = zipfile.ZipFile(zip_path, 'r')
    zip_ref.extractall(data_path)
    zip_ref.close()
    os.remove(zip_path)


if __name__ == "__main__":
    download_url('https://data.vision.ee.ethz.ch/cvl/clic/professional_train.zip', 'datasets/train/1.zip')
    download_url('https://data.vision.ee.ethz.ch/cvl/clic/mobile_train.zip', 'datasets/train/2.zip')
    zip_files = glob('datasets/train/*.zip')
    for zip_path in zip_files:
        unzip_zip_file(zip_path, 'datasets')
        os.remove(zip_path)
