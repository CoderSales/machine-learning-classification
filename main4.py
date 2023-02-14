import subprocess

subprocess.run("start-env.py", check=True)

subprocess.run("script.sh", check=True)
