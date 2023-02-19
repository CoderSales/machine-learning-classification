import importlib.util
import sys
import os
import subprocess


def installer():
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
        "importlib.util",
        "subprocess",
        "warnings",
        "statsmodels.tools.sm_exceptions",
        "matplotlib.pyplot",
        "sklearn.model_selection.train_test_split",
        "statsmodels.stats.api",
        "statsmodels.stats.outliers_influence",
        "statsmodels.api",
        "statsmodels.tools.tools",
        "sklearn.tree.DecisionTreeClassifier",
        "sklearn.tree",
        "sklearn.model_selection.GridSearchCV",
        "sklearn.metrics.f1_score",
        "sklearn.metrics.accuracy_score",
        "sklearn.metrics.recall_score",
        "sklearn.metrics.precision_score",
        "sklearn.metrics.confusion_matrix",
        "sklearn.metrics.roc_auc_score",
        "sklearn.metrics.ConfusionMatrixDisplay",
        "sklearn.metrics.precision_recall_curve",
        "sklearn.metrics.roc_curve",
        "sklearn.metrics.make_scorer",
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

    uninstalled_list = loop_formatted_packages(
        unchecked_formatted_list, uninstalled_list
    )

    print("\nuninstalled modules are:", uninstalled_list, "\n")

    newly_installed = []

    def installer(uninstalled_list):
        for name in uninstalled_list:
            if name == "scikit-learn":
                name2 = "-U " + name
            os.system("pip install {}".format(name2))
            print(name, "module has been installed")
            newly_installed.append(name)
        print("\n", newly_installed, "modules have just been installed")

    installer(uninstalled_list)
