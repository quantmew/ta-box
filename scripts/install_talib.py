import sys
import platform
import subprocess
import urllib.request
import os

# Basic information
version_tag = "0.6.4"
base_url = f"https://github.com/quantmew/talib-prebuilt/releases/download/v{version_tag}/"
filename_prefix = f"ta_lib-{version_tag}"

# Get Python version
py_version = f"cp{sys.version_info.major}{sys.version_info.minor}"
py_tag = f"{py_version}-{'cp' + str(sys.version_info.major) + str(sys.version_info.minor)}"

# Get platform information
system = platform.system()
machine = platform.machine().lower()
platform_tag = ""

if system == "Windows":
    if platform.architecture()[0] == "64bit":
        platform_tag = "win_amd64"
    else:
        platform_tag = "win32"
elif system == "Linux":
    # Assume default using manylinux
    platform_tag = "manylinux_2_28_x86_64"
    if "musl" in platform.libc_ver()[0].lower():
        platform_tag = "musllinux_1_2_x86_64"
elif system == "Darwin":
    mac_ver = platform.mac_ver()[0]
    major_version = int(mac_ver.split('.')[0])
    arch = "arm64" if machine == "arm64" or machine == "aarch64" else "x86_64"
    if major_version >= 14:
        platform_tag = f"macosx_14_0_{arch}"
    else:
        platform_tag = f"macosx_13_0_{arch}"
else:
    raise RuntimeError(f"Unsupported OS: {system}")

# Construct file name and URL
wheel_filename = f"{filename_prefix}-{py_tag}-{platform_tag}.whl"
download_url = base_url + wheel_filename

# Download whl file
print(f"[+] Detected Python {sys.version_info.major}.{sys.version_info.minor}, System: {system}, Arch: {platform_tag}")
print(f"[+] Downloading wheel: {download_url}")
urllib.request.urlretrieve(download_url, wheel_filename)

# Install wheel file
print(f"[+] Installing {wheel_filename}...")
subprocess.check_call([sys.executable, "-m", "pip", "install", wheel_filename])

# Optional: delete downloaded file
os.remove(wheel_filename)
print("[+] Installation complete and wheel file removed.")
