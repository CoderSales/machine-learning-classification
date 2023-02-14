import sys
import importlib.util

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

for name in modules_list:  # '-U scikit-learn'
    string = name  #
    if string == "-U scikit-learn":
        string = string[2:15]
    else:
        module = string
    if name in sys.modules:
        print(f"{name!r} already in sys.modules")
    elif (spec := importlib.util.find_spec(name)) is not None:
        # If you choose to perform the actual import ...
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        print(f"{name!r} has been imported")
    else:
        print(f"can't find the {name!r} module")
