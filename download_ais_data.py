import requests
import xml.etree.ElementTree as ET
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

YEAR = "2024"
S3_API_URL = f"http://aisdata.ais.dk.s3.eu-central-1.amazonaws.com/?delimiter=/&prefix={YEAR}/"
BASE_DOWNLOAD_URL = "http://aisdata.ais.dk/"
DOWNLOAD_DIR = "data"
MAX_WORKERS = 8


def get_zip_urls(xml_content):
    root = ET.fromstring(xml_content)
    namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

    zip_urls = []
    for contents in root.findall("s3:Contents", namespace):
        key_elem = contents.find("s3:Key", namespace)
        if key_elem is not None and key_elem.text.endswith(".zip"):
            zip_urls.append(BASE_DOWNLOAD_URL + key_elem.text)

    return zip_urls


def download_file(url, dest_path, retries=3):
    filename = os.path.basename(url)

    if dest_path.exists():
        print(f"Skipping {filename} (already exists)")
        return filename, True, "Already exists"

    for attempt in range(retries):
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            with open(dest_path, "wb") as f, tqdm(
                desc=filename, total=total_size, unit="B", unit_scale=True, unit_divisor=1024, leave=False
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            return filename, True, "Downloaded"

        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2**attempt)
                continue
            else:
                if dest_path.exists():
                    dest_path.unlink()
                return filename, False, str(e)

    return filename, False, "Max retries exceeded"


def main():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    print(f"Fetching file list from {S3_API_URL}...")
    response = requests.get(S3_API_URL)
    response.raise_for_status()

    zip_urls = get_zip_urls(response.text)
    print(f"Found {len(zip_urls)} zip files")

    download_dir = Path(DOWNLOAD_DIR)

    successful = []
    failed = []
    skipped = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_file, url, download_dir / os.path.basename(url)): url for url in zip_urls}

        with tqdm(total=len(zip_urls), desc="Overall Progress") as pbar:
            for future in as_completed(futures):
                filename, success, message = future.result()

                if message == "Already exists":
                    skipped.append(filename)
                elif success:
                    successful.append(filename)
                else:
                    failed.append((filename, message))

                pbar.update(1)

    print(f"\n{'='*60}")
    print(f"Download Summary:")
    print(f"  Successfully downloaded: {len(successful)}")
    print(f"  Skipped (already exist): {len(skipped)}")
    print(f"  Failed: {len(failed)}")

    if failed:
        print(f"\nFailed downloads:")
        for filename, error in failed:
            print(f"  - {filename}: {error}")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
