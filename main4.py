import subprocess

import startenv


# subprocess.run("python.exe", "start-env.py", check=True)

# subprocess.run("script.sh", check=True)
subprocess.call(["sh", "script.sh"])
