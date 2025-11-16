import requests
import xml.etree.ElementTree as ET
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time


def get_zip_urls(xml_content, base_url="http://aisdata.ais.dk/"):
    root = ET.fromstring(xml_content)
    namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

    zip_urls = []
    for contents in root.findall("s3:Contents", namespace):
        key_elem = contents.find("s3:Key", namespace)
        if key_elem is not None and key_elem.text.endswith(".zip"):
            zip_urls.append(base_url + key_elem.text)

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


def download_ais_data(year, download_dir, max_workers=8):
    s3_api_url = f"http://aisdata.ais.dk.s3.eu-central-1.amazonaws.com/?delimiter=/&prefix={year}/"
    if not isinstance(download_dir, Path):
        download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching file list from {s3_api_url}...")
    response = requests.get(s3_api_url)
    response.raise_for_status()

    zip_urls = get_zip_urls(response.text)
    print(f"Found {len(zip_urls)} zip files")

    successful = []
    failed = []
    skipped = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
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

    return {
        'successful': successful,
        'failed': failed,
        'skipped': skipped
    }
