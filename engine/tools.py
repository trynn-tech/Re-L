# engine/tools.py
import os, subprocess

def write_file(path: str, content: str):
    """Allows the Emperor to crystallize thoughts into code or docs."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"

def exec_python(code: str):
    """A limited REPL for the engine to test its own logic."""
    # Note: In a production environment, use a sandbox like Docker or Modal
    try:
        result = subprocess.run(
            ["python3", "-c", code], 
            capture_output=True, text=True, timeout=5
        )
        return f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    except Exception as e:
        return f"Execution Error: {str(e)}"


