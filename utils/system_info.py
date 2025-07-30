import os
import platform
try:
    import tensorflow as tf
except ImportError:
    tf = None

try:  
    import torch 
except ImportError:
    torch = None  

from datetime import datetime
import subprocess

# For Windows registry access (only on Windows)
try:
    import winreg
except ImportError:
    winreg = None


def get_cpu_name():
    """
    Retrieve the CPU name for the current system.

    Returns:
        str: The CPU model name, or "Unknown" if it cannot be determined.
    """
    system = platform.system()

    try:
        if system == "Windows":
            result = subprocess.run(
                ["wmic", "cpu", "get", "Name"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            output = result.stdout.decode().strip().split('\n')
            return output[1].strip() if len(output) > 1 else "Unknown"
        elif system == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        # Extract the CPU model name from /proc/cpuinfo
                        return line.strip().split(":")[1].strip()
        elif system == "Darwin":  # macOS
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                stdout=subprocess.PIPE
            )
            return result.stdout.decode().strip()
        else:
            return platform.processor()
    except Exception as e:
        return f"Unknown ({e})"


def get_gpu_info_nvidia_smi():
    """
    Print GPU information using nvidia-smi if available.

    The function queries GPU name and memory usage and prints each GPU's:
        - Total memory
        - Used memory
        - Free memory

    Returns:
        None
    """
    try:
        result = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,memory.used,memory.free",
             "--format=csv,nounits,noheader"],
            universal_newlines=True
        )
        lines = result.strip().split('\n')
        for idx, line in enumerate(lines):
            name, total, used, free = [x.strip() for x in line.split(',')]
            print(f"- GPU {idx}                  : {name}")
            print(f"  ↳ Total memory         : {total} MB")
            print(f"  ↳ Used memory          : {used} MB")
            print(f"  ↳ Free memory          : {free} MB")
    except FileNotFoundError:
        print("- Error: `nvidia-smi` is not available on the system.")
    except Exception as e:
        print(f"- Error detecting GPU with `nvidia-smi`: {e}")


def get_windows_edition():
    """
    Read Windows edition details from the registry and format them.

    Reads the following registry values:
        - ProductName (e.g., "Windows 11 Home Single Language")
        - DisplayVersion (e.g., "23H2" on Win-11) or ReleaseId (e.g., "21H2" on older Win-10)
        - CurrentBuild (e.g., "22631")
        - UBR (e.g., "5472")

    Returns:
        str: A human-readable Windows edition string or "Unknown (error)" on failure.
    """
    try:
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Microsoft\Windows NT\CurrentVersion"
        )
        product = winreg.QueryValueEx(key, "ProductName")[0]
        # Try DisplayVersion (Win-11), otherwise ReleaseId
        try:
            version = winreg.QueryValueEx(key, "DisplayVersion")[0]
        except FileNotFoundError:
            version = winreg.QueryValueEx(key, "ReleaseId")[0]
        build = winreg.QueryValueEx(key, "CurrentBuild")[0]
        ubr = winreg.QueryValueEx(key, "UBR")[0]
        return f"{product} ({version}, Build {build}.{ubr})"
    except Exception as e:
        return f"Unknown ({e})"


def get_linux_distribution():
    """
    Retrieve the pretty name of the Linux distribution from /etc/os-release.

    Returns:
        str: The value of PRETTY_NAME or a combination of NAME and VERSION, or "Unknown (error)".
    """
    info = {}
    try:
        with open("/etc/os-release") as f:
            for line in f:
                if "=" in line:
                    key, val = line.rstrip().split("=", 1)
                    info[key] = val.strip().strip('"')
        return info.get("PRETTY_NAME", f"{info.get('NAME','')} {info.get('VERSION','')}".strip())
    except Exception as e:
        return f"Unknown ({e})"


def get_os_info():
    """
    Get operating system kernel version and human-readable name.

    Returns:
        tuple: (kernel_version: str, human_readable: str)
    """
    system = platform.system()
    kernel = platform.release()
    if system == "Windows" and winreg:
        human = get_windows_edition()
    elif system == "Linux":
        human = get_linux_distribution()
    else:
        # macOS or others
        human = system
    return kernel, human


def print_system_info():
    """
    Print comprehensive system and runtime environment information.

    This includes:
        - Current date and time
        - Virtual environment name
        - OS kernel and distribution
        - CPU and GPU details
        - Python, TensorFlow, and Keras versions
        - Current working directory

    Returns:
        None
    """
    print("-" * 60)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("System and runtime environment information:")
    print("=" * 60)
    print(f" Execution date and time : {current_time}")
    print("=" * 60)
    print(f"- Virtual environment    : {os.environ.get('CONDA_DEFAULT_ENV', 'Unknown')}")
    print(f"- OS Kernel              : {platform.system()} - Release {platform.release()} ({platform.version()})")
    print(f"- OS Distribution        : {get_os_info()[1]}")
    print(f"- CPU                    : {get_cpu_name()}")
    get_gpu_info_nvidia_smi()
    print(f"- Python version         : {platform.python_version()}")
    if tf is not None:
        print(f"- TensorFlow             : {tf.__version__}")
        try:
            print(f"- Keras                  : {tf.keras.__version__}")
        except AttributeError:
            print("- Keras                  : Not available in this TensorFlow version")
    if torch is not None:
        print(f"- PyTorch                : {torch.__version__}")
    print("")
    print(f"- Current directory      : {os.getcwd()}")
    print("=" * 60)
