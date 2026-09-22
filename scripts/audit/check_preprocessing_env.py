import os
import sys
import subprocess

REPORT_FILE = "/mnt/c/Users/krish/FYP/audit/preprocessing_environment.txt"
BIDS_DIR = "/mnt/c/Users/krish/FYP/BIDS"

def check_cmd(cmd):
    try:
        result = subprocess.run(["which", cmd], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else "Missing"
    except:
        return "Missing"

def check_pkg(pkg_name):
    try:
        if pkg_name == "bids":
            import bids
            return bids.__version__
        elif pkg_name == "nipype":
            import nipype
            return nipype.__version__
        elif pkg_name == "nibabel":
            import nibabel
            return nibabel.__version__
        elif pkg_name == "nilearn":
            import nilearn
            return nilearn.__version__
        elif pkg_name == "numpy":
            import numpy
            return numpy.__version__
        elif pkg_name == "scipy":
            import scipy
            return scipy.__version__
        elif pkg_name == "pandas":
            import pandas
            return pandas.__version__
    except ImportError:
        return "Missing"

def main():
    print("Gathering environment information...")
    
    os_info = subprocess.run(["cat", "/etc/os-release"], capture_output=True, text=True).stdout
    uname_info = subprocess.run(["uname", "-a"], capture_output=True, text=True).stdout
    
    python_ver = sys.version.split()[0]
    
    bids_access = "NO"
    if os.path.exists(BIDS_DIR) and os.path.isdir(BIDS_DIR):
        try:
            os.listdir(BIDS_DIR)
            bids_access = "YES"
        except:
            pass

    env_data = [
        "=== PREPROCESSING ENVIRONMENT VERIFICATION ===",
        "",
        "--- SYSTEM ---",
        f"OS / Kernel: {uname_info.strip()}",
        "",
        "--- PYTHON ---",
        f"Python Version: {python_ver}",
        f"Nipype: {check_pkg('nipype')}",
        f"NiBabel: {check_pkg('nibabel')}",
        f"Nilearn: {check_pkg('nilearn')}",
        f"NumPy: {check_pkg('numpy')}",
        f"SciPy: {check_pkg('scipy')}",
        f"Pandas: {check_pkg('pandas')}",
        f"PyBIDS: {check_pkg('bids')}",
        "",
        "--- EXECUTABLES (FSL) ---",
        f"FSL Availability: {'YES' if check_cmd('fsl') != 'Missing' else 'NO'}",
        f"fsl Path: {check_cmd('fsl')}",
        f"flirt Path: {check_cmd('flirt')}",
        f"mcflirt Path: {check_cmd('mcflirt')}",
        f"fslmaths Path: {check_cmd('fslmaths')}",
        "",
        "--- BIDS ACCESS ---",
        f"BIDS directory accessible: {bids_access}",
        f"Path: {BIDS_DIR}"
    ]
    
    report_text = "\n".join(env_data)
    
    print(report_text)
    
    os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)
    with open(REPORT_FILE, "w") as f:
        f.write(report_text + "\n")
        
    print(f"\nReport written to {REPORT_FILE}")

if __name__ == "__main__":
    main()
