from bing_image_downloader import downloader
import os

def create_dataset():
    base_dir = "dataset"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # 1. Workshop
    print("Downloading Workshop Images...")
    downloader.download("pottery workshop", limit=20, output_dir=f"{base_dir}/workshop", adult_filter_off=True, force_replace=False, timeout=60, verbose=False)
    
    # 2. Artifacts
    print("Downloading Artifacts Images...")
    downloader.download("traditional wooden or pottery artifacts", limit=25, output_dir=f"{base_dir}/artifacts", adult_filter_off=True, force_replace=False, timeout=60, verbose=False)
    
    # 3. Process
    print("Downloading Process Images...")
    downloader.download("wood carving craft process", limit=20, output_dir=f"{base_dir}/process", adult_filter_off=True, force_replace=False, timeout=60, verbose=False)
    
    # 4. Textures
    print("Downloading Texture Images...")
    downloader.download("wood grain texture high res", limit=10, output_dir=f"{base_dir}/textures", adult_filter_off=True, force_replace=False, timeout=60, verbose=False)
    downloader.download("ceramic glaze texture close up", limit=15, output_dir=f"{base_dir}/textures", adult_filter_off=True, force_replace=False, timeout=60, verbose=False)

if __name__ == "__main__":
    create_dataset()
    print("Dataset downloaded and organized successfully into 'dataset/' directory.")
