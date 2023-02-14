import os

import subprocess

subprocess.call(["sh", "bash_activate.sh"])


# def activate () {
#   . .venv/Scripts/activate
# }

# activate

# part b:
os.system(
    ". .venv/Scripts/activate && pip install notebook && pip install matplotlib && pip install pandas && pip install seaborn && pip install numpy && pip install scipy && pip install statsmodels && pip install -U scikit-learn && pip install ipykernel && pip install nb-black"
)

# part z:
# modules_list to pass modules
modules_list = [
    "notebook",
    "matplotlib",
    "pandas",
    "seaborn",
    "numpy",
    "scipy",
    "statsmodels",
    "-U scikit-learn",
    "ipykernel",
    "nb-black",
]

# part z a:
# look through modules_list
# after check
# check if scikit-learn is present
# if doing this way
# delete first 3 characters
# or do nothing
# if listed post check without
# add 1st 3 characters

# part a:
# add checker

# # TODO: uncomment and code:
# not_installed=[] # to be populated

# for package in packages:
#     # do some or nothing
#     if package <how to check if installed>: # if package is installed next
#         pass
#     # pass out non-installed elements
#     else:
#         not_installed.append(package) # 1. add it to the list
# end of TODO

# 2. Install not_installed

# 3. concatenate prepend as prefix -U to scikit-learn
