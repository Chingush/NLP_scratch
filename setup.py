import sys
from cx_Freeze import setup, Executable

base = None

if sys.platform == "win32":
    base = "Win32GUI"

executables = [Executable("scratch.py", base=base)]

build_exe_options = {
    "include_files": ["C:\\Users\\Пользователь\\Downloads\\nlp\\nlp"]
}

setup(
    options={"build_exe": build_exe_options},
    name="NLP",
    version="1.0",
    description="Description of your app",
    executables=executables
)