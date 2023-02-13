import sys
import os

os.system("ls -l")

# stream = os.popen("echo Returned output")
# output = stream.read()
# output


# import subprocess

# process = subprocess.Popen(
#     ["echo", "More output"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
# )
# stdout, stderr = process.communicate()
# stdout, stderr


# with open("test.txt", "w") as f:
#     process = subprocess.Popen(["ls", "-l"], stdout=f)


# process = subprocess.Popen(
#     ["ping", "-c 4", "python.org"], stdout=subprocess.PIPE, universal_newlines=True
# )

# while True:
#     output = process.stdout.readline()
#     print(output.strip())
#     # Do something else
#     return_code = process.poll()
#     if return_code is not None:
#         print("RETURN CODE", return_code)
#         # Process has finished, read rest of the output
#         for output in process.stdout.readlines():
#             print(output.strip())
#         break


# import shlex

# shlex.split('/bin/prog -i data.txt -o "more data.txt"')


# process = subprocess.run(
#     ["echo", "Even more output"], stdout=subprocess.PIPE, universal_newlines=True
# )
# process

# process.stdout


# import subprocess

# ssh = subprocess.Popen(
#     ["ssh", "-i .ssh/id_rsa", "user@host"],
#     stdin=subprocess.PIPE,
#     stdout=subprocess.PIPE,
#     stderr=subprocess.PIPE,
#     universal_newlines=True,
#     bufsize=0,
# )

# # Send ssh commands to stdin
# ssh.stdin.write("uname -a\n")
# ssh.stdin.write("uptime\n")
# ssh.stdin.close()

# # Fetch output
# for line in ssh.stdout:
#     print(line.strip())


# %%sh pip install geocoder
# !{sys.executable} -m pip install -r requirements.txt

# install
# https://discourse.jupyter.org/t/python-in-terminal-finds-module-jupyter-notebook-does-not/2262/7
# **!**pip install module name
# !pip install could install to wrong env use % instead
os.system("ls -l%pip install jup notebook")
# os.system("%pip install matplotlib")
# os.system("%pip install pandas")
# os.system("%pip install seaborn")
# os.system("%pip install numpy")
# os.system("%pip install scipy")
# os.system("%pip install statsmodels")
# os.system("%pip install -U scikit-learn")
# os.system("%pip install ipykernel")
# os.system("%pip install nb-black")

# !pip install nb-black

# --------------------------------------

# this will help in making the Python code more structured automatically (help adhere to good coding practices)
os.system("%load_ext nb_black")

import warnings

warnings.filterwarnings("ignore")
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter("ignore", ConvergenceWarning)

# Libraries to help with reading and manipulating data
import pandas as pd
import numpy as np

# libaries to help with data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Removes the limit for the number of displayed columns
pd.set_option("display.max_columns", None)
# Sets the limit for the number of displayed rows
pd.set_option("display.max_rows", 200)
# setting the precision of floating numbers to 5 decimal points
pd.set_option("display.float_format", lambda x: "%.5f" % x)

# Library to split data
from sklearn.model_selection import train_test_split

# To build model for prediction
import statsmodels.stats.api as sms
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from statsmodels.tools.tools import add_constant
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# To tune different models
from sklearn.model_selection import GridSearchCV


# To get diferent metric scores
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    roc_curve,
    make_scorer,
)
