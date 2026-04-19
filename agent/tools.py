from langchain.tools import tool
import os

@tool
def write_file(path: str, content: str) -> str:
    """Write content to a file"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"File written to {path}"


@tool
def read_file(path: str) -> str:
    """Read content from a file"""
    if not os.path.exists(path):
        return "File does not exist"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@tool
def list_files() -> str:
    """List files in current directory"""
    return "\n".join(os.listdir())


@tool
def get_current_directory() -> str:
    """Get current working directory"""
    return os.getcwd()