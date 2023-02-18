# import sys
# import importlib.util
# import sys

# For illustrative purposes.
# name = "itertools"
# modules_list = [
#     "notebook",
#     "matplotlib",
#     "pandas",
#     "seaborn",
#     "numpy",
#     "scipy",
#     "statsmodels",
#     "-U scikit-learn",
#     "ipykernel",
#     "nb-black",
# ]

# for name in modules_list:  # '-U scikit-learn'
#     string = name  #
#     if string == "-U scikit-learn":
#         string = string[2:15]
#     else:
#         module = string
#     if name in sys.modules:
#         print(f"{name!r} already in sys.modules")
#     elif (spec := importlib.util.find_spec(name)) is not None:
#         # If you choose to perform the actual import ...
#         module = importlib.util.module_from_spec(spec)
#         sys.modules[name] = module
#         spec.loader.exec_module(module)
#         print(f"{name!r} has been imported")
#     else:
#         print(f"can't find the {name!r} module")


import os

# os.system("source .venv/Scripts/activate")
# os.system(
#     ".venv/Scripts/activate && pip install notebook && pip install matplotlib && pip install pandas && pip install seaborn && pip install numpy && pip install scipy && pip install statsmodels && pip install -U scikit-learn && pip install ipykernel && pip install nb-black"
# )

# os.system("ls -l")
# os.system("pip install notebook")
# os.system("pip install matplotlib")
# os.system("pip install pandas")
# os.system("pip install seaborn")
# os.system("pip install numpy")
# os.system("pip install scipy")
# os.system("pip install statsmodels")
# os.system("pip install -U scikit-learn")
# os.system("pip install ipykernel")
# os.system("pip install nb-black")
