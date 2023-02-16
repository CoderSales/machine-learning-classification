import importlib.util
import sys
import os


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
    "importlib.util",
    "sys",
    "os",
]

uninstalled_list = []
modules_list_modified = []
unchecked_formatted_list = []


def stringifier(string):
    string = string[3:15]
    return string


def sort_output(name):
    string = name  #
    # module
    if string == "-U scikit-learn":
        module = stringifier(string)  # string = string[2:15] #
        return module
    # now have holder=scikit-learn
    else:
        module = string
        return string


def loop_unformatted_packages(modules_list):
    for name in modules_list:  # '-U scikit-learn'
        unchecked_formatted_list.append(sort_output(name))
    return unchecked_formatted_list


unchecked_formatted_list = loop_unformatted_packages(modules_list)


def check_not_installed(name):
    """
    Check if a package (formatted) is installed or not.
    """
    if name in sys.modules:
        print(f"{name!r} already in sys.modules")
    elif (spec := importlib.util.find_spec(name)) is not None:
        # If you choose to perform the actual import ..
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        print(f"{name!r} has been imported")
    else:
        print(f"can't find the {name!r} module")
        return name


def loop_formatted_packages(unchecked_formatted_list, uninstalled_list):
    """
    takes:
     - var unchecked_formatted_list
     - var uninstalled_list
    calls:
     - function check_not_installed
    """
    for name in unchecked_formatted_list:  # '-U scikit-learn'
        if check_not_installed(name):
            uninstalled_list.append(name)
    return uninstalled_list


uninstalled_list = loop_formatted_packages(unchecked_formatted_list, uninstalled_list)

print("\nuninstalled modules are:", uninstalled_list)

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


# def installer():


# installer()


# def importer():


# importer()
