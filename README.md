# machine-learning-classification

# primary source for this README: jupyter-6-Supervised-Learning
Repository for running jupyter notebooks and keeping relevant files in one place


Updates from 

# ML-logistic-regression-notes
- [CoderSales/ML-logistic-regression-notes](https://github.com/CoderSales/ML-logistic-regression-notes/blob/main/README.md)

# All content below this point from documentation repository:
- [CoderSales/documentation](https://github.com/CoderSales/documentation)

# documentation
documentation for different repositories

# assembling:
# part 1:

[closed], W., Wencel, W. and Agrawal, S. (2016) What is the difference between a feature and a label?, Stack Overflow. Available at: https://stackoverflow.com/questions/40898019/what-is-the-difference-between-a-feature-and-a-label#:~:text=Briefly%2C%20feature%20is%20input%3B%20label,region%2C%20family%20income%2C%20etc. (Accessed: 9 February 2023).

# part 2: Repos used to compile this README.md :
## ML-logistic-regression-notes
## machine-learning-classification


# part 3: README from Repo 1
# ML-logistic-regression-notes

# files
QUICKSTART-WIN-VSC-BASH.md
from 
- [CoderSales/machine-learning-classification](https://github.com/CoderSales/machine-learning-classification)

# primary source for this README: machine-learning-classification
Repository for running jupyter notebooks and keeping relevant files in one place
# secondary source for this README: jupyter-6-Supervised-Learning
Repository for running jupyter notebooks and keeping relevant files in one place


# notes
## notes made for previous plan to remove null values
check how to remove null values from dataframe

### notes
pandas 
.iloc() - locate by row, col indices
.loc() - locate by row index and col NAME

### how to run python files from terminal
- [python3 main.py](https://realpython.com/run-python-scripts/)
 
### Data Cleaning
### 2.13 Lecture
df.drop('Column name', axis=1)
    - where axies = 0 for rows, 1 for columns
    - drops referenced column from data frame
    - inplace=True argument to ensure column stays dropped.
df.drop(1,axis=0).reset_index()
    - new col with old indices
df.drop(1,axis=0).reset_index(drop=True,inplace=True)

df.copy

### 4.1 Lecture Data Sanity Checks - Part 1
df['columnname'].apply(type).value_counts()
    - this looks at and notes the values by type and then counts them

df['colname'] = df['colname'].replace('missing','inf'],np.nan)
    - replaces our specified strings 'missing' and 'inf' 
    -  with np.nan

df['colname'] = df['colname'].astype(float)
    - convert values to float

Review note: when we substitute np.nan in for strings the resulting data type is (if all the other entries are say float) float.

df.info()
    - rerunning this after data cleaning may result in cleaned columns type changing to, say, float.

Check length of each column 
Columns shorter than max col length means missing values as empty cells

#### Alternative approach - clean while loading:
##### using na_values to tell python which values it should consider as NaN
data_new = pd.read_csv('/content/drive/MyDrive/Python Course/Melbourne_Housing.csv',na_values=['missing','inf'])
- on load, above line automatically converts all missing and inf to nan so, running:
data_new['BuildingArea'].dtype
- gives 
dtype('float64')
as only float (and nan which seems to be treated as whatever the rest of the data types are)

#### Review note
data['BuildingArea'].unique()
- above line run before cleaning gives unique values in column as a numpy array
- so can inspect to find out which strings to remove.
# setup steps
python3 -m venv .venv
    - in bash
    - and on Windows
source .venv/bin/activate
    - in bash
source .venv/Scripts/activate
    - on Windows
    - on VSCode Windows bash
/workspace/machine-learning-classification/.venv/bin/python -m pip install --upgrade pip
    - in GitPod
python3 -m pip install --upgrade pip
    - on Windows

.venv/Scripts/python.exe -m pip install --upgrade pip
    - in .venv

pip install --upgrade pip
pip install jupyter notebook
pip install matplotlib
pip install pandas
pip install seaborn
pip install numpy
pip install scipy
pip install statsmodels
pip install -U scikit-learn
pip install ipykernel
pip install nb-black


Ctrl Shift P
Create New Jupyter Notebook
Save and name notebook
Paste in necessary code

Ctrl Shift P
Python: Select Interpreter
use Python version in ./.venv/bin/python

pip freeze > requirements.txt

pip install -r requirements.txt

## Add required files
pima-indians-diabetes.csv
## Extensions
Extension: Excel Viewer
    - for  viewing csv files in VSCode

## Debug

### jupyter cannot find modules

- [install modules from jupyter notebook](https://discourse.jupyter.org/t/python-in-terminal-finds-module-jupyter-notebook-does-not/2262/7)
### prelim
per above
Python:Select Interpreter
3.10.9 (.venv)
### ipykernel bug
after running
pip install ipykernel
on running LinearRegression_HandsOn-1.ipynb
message appears saying:
it is necessary to install ipykernel
OK
installing ipykernel
Rerun
LinearRegression_HandsOn-1.ipynb

### pandas bug
after running
pip install pandas 
pandas not found

### Fix for previous 2 bugs
create new jupyter notebook using 
Ctrl Shift P
Create New Jupyter Notebook

# Files
## summary
- summary-income.md
    - high level summary of steps in income.ipynb notebook
# References
## previous repositories
jupyter-test
jupyter-repo-2
jupyter-3
- [CoderSales/jupyter-5](https://github.com/CoderSales/jupyter-5)
- [CoderSales/jupyter-6-Supervised-Learning](https://github.com/CoderSales/jupyter-6-Supervised-Learning)
- [CoderSales/machine-learning-project](https://github.com/CoderSales/machine-learning-project)


# References Part2 / (MyGreatLearning, Colab, modules)
#### MyGreatLearning
##### pre scikit-learn
- [LMS - Hands_on_Notebook_Week3.ipynb](https://www.mygreatlearning.com/)
- [LMS - ENews_Express_Learner_Notebook%5BLow_Code_Version%5D.ipynb](https://www.mygreatlearning.com/)
- [LMS - abtest.csv](https://www.mygreatlearning.com/)
- [2.13 Pandas - Accessing and Modifying DataFrames (condition-based indexing)](https://www.mygreatlearning.com/)
#### scikit-learn
- [Supervised Learning - Foundations / Week 1 - Lecture Video Materials](https://www.mygreatlearning.com/)
    - [auto-mpg.csv used in 1.9 Linear Regression Hands-on](https://www.mygreatlearning.com/)

#### Colab
- Google Colab [mount drive](https://colab.research.google.com/notebooks/io.ipynb#scrollTo=RWSJpsyKqHjH)

#### modules
##### matplotlib
###### matplotlib figure dimentions
- [Set plot dimensions matplotlib](https://stackoverflow.com/questions/332289/how-do-i-change-the-size-of-figures-drawn-with-matplotlib)

##### scipy
- [scipy - check version](https://blog.finxter.com/how-to-check-scipy-package-version-in-python/)


# References Part3 / (StackOverflow, Git, Tutorials and Repositories)
## StackOverflow
https://stackoverflow.com/questions/46419607/how-to-automatically-install-required-packages-from-a-python-script-as-necessary

## Git
### git
#### gitignore
- [How to stop tracking and ignore changes to a file in Git?](https://stackoverflow.com/questions/936249/how-to-stop-tracking-and-ignore-changes-to-a-file-in-git)
### Gitpod
- [Gitpod docs prebuilds](https://www.gitpod.io/docs/configure/projects/prebuilds)
- [Gitpod docs workspaces](https://www.gitpod.io/docs/configure/workspaces/tasks)
- [Gitpod Prebuild](https://youtu.be/ZtlJ0PakUHQ?t=54)
### Git in VSCode
- search string: [pause git tracking](https://www.google.com/search?q=pause+git+tracking&oq=pause+git+tracking&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTILCAEQABgWGB4Y8QQyCwgCEAAYFhgeGPEEMgsIAxAAGBYYHhjxBDILCAQQABgWGB4Y8QQyCwgFEAAYFhgeGPEEMggIBhAAGBYYHjIICAcQABgWGB4yCwgIEAAYFhgeGPEEMgsICRAAGBYYHhjxBNIBCDM2NjNqMGo3qAIAsAIA&sourceid=chrome&ie=UTF-8)
- [Git source control in VS Code](https://code.visualstudio.com/docs/sourcecontrol/overview)

## Tutorials and Repositories

# References Part4 / (environments, Packages, Statistics, python, ML, Stats for ML)
## environments
### local
- [Getting Full Directory Path in Python](https://www.youtube.com/watch?v=DQRSvg54bhM&ab_channel=Analyst%27sCorner)

Windows
Anaconda
conda create --name .cenv
y
conda activate .cenv

python3

not installed so Windows store opens
install Python 3.10

#### conda
##### virtual environment
- [conda.io](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html)

#### python environment
`python3 -m venv .venv`
command was slow at first  but self-resolved
- search string: stuck on $ python3 -m venv .venv [setting up environment in virtaulenv using python3 stuck on ...](https://discuss.dizzycoding.com/setting-up-environment-in-virtaulenv-using-python3-stuck-on-setuptools-pip-wheel/)
- search string: installing collected packages stuck [why is the pip install process stuck on ''Installing collected packages" step?](https://stackoverflow.com/questions/54699197/why-is-the-pip-install-process-stuck-on-installing-collected-packages-step)

## Packages
### NumPy
- search string: [np.clip](https://www.google.com/search?q=np.clip&oq=np.clip&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIHCAEQABiABDIHCAIQABiABDIHCAMQABiABDIHCAQQABiABDIHCAUQABiABDIHCAYQABiABNIBBzk0N2owajeoAgCwAgA&sourceid=chrome&ie=UTF-8)
- [numpy.clip | numpy.org | Documentation](https://numpy.org/doc/stable/reference/generated/numpy.clip.html)
- search string: [np broadcast against dataframe python](https://www.google.com/search?q=np+broadcast+against+dataframe+python&newwindow=1&sxsrf=AJOqlzVs5XFBfTGYuALitoPd-H-QfsrAUA%3A1676106129272&ei=kVnnY-miEJCP8gKjgaIo&oq=np+broadcast+against+datafra&gs_lcp=Cgxnd3Mtd2l6LXNlcnAQAxgBMgcIIRCgARAKMgcIIRCgARAKMgcIIRCgARAKOgoIABBHENYEELADOgUIABCiBDoFCCEQoAFKBAhBGABKBAhGGABQ2xRYkkBg9FBoAnABeACAAXuIAcwGkgEDNy4ymAEAoAEByAEIwAEB&sclient=gws-wiz-serp)
- [Q/ What does the term "broadcasting" mean in Pandas documentation? | A/ the term broadcasting comes from numpy | stackoverflow](https://stackoverflow.com/questions/29954263/what-does-the-term-broadcasting-mean-in-pandas-documentation)
- [broadcasting examples in pandas documentaton | linked to by previous reference on broadcasting | pandas.org | Documentation](https://pandas.pydata.org/pandas-docs/stable/user_guide/basics.html)
- [Broadcasting | definition: | The term broadcasting describes how NumPy treats arrays with different shapes during arithmetic operations. | NumPy | numpy.org | Documentation](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- [Universal functions (ufunc) | NumPy | numpy.org | Documentation](https://numpy.org/doc/stable/reference/ufuncs.html#ufuncs-kwargs)
### Pandas
- [EDA: from is_categorical def | Check if dataframe column is Categorical | print(is_categorical(data[col])) | stackoverflow](https://stackoverflow.com/questions/26924904/check-if-dataframe-column-is-categorical)
- [pandas.get_dummies | Pandas | pandas.pydata.org | Documentation](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html)
- search string: [pd.get_dummies](https://www.google.com/search?q=pd.get_dummies&oq=pd.get_dummies&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIGCAEQIxgnMgcIAhAAGIAEMgcIAxAAGIAEMgcIBBAAGIAEMgcIBRAAGIAEMgcIBhAAGIAEMgcIBxAAGIAEMgcICBAAGIAEMgcICRAAGIAE0gEHNDUzajBqN6gCALACAA&sourceid=chrome&ie=UTF-8)
### matplotlib
- search string: [plotting fig from subplot returns Figure(1500x1000)](https://www.google.com/search?q=plotting+fig+from+subplot+returns+Figure(1500x1000)&oq=plotting+fig+from+subplot+returns+Figure(1500x1000)&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIHCAEQABiiBDIHCAIQABiiBDIHCAMQABiiBDIHCAQQABiiBDIHCAUQABiiBNIBCTEyMzgzajFqN6gCALACAA&sourceid=chrome&ie=UTF-8)
- [fig, ax = plt.subplots()](https://stackoverflow.com/questions/34162443/why-do-many-examples-use-fig-ax-plt-subplots-in-matplotlib-pyplot-python)
- matplotlib docs [fig, ax = plt.subplots()](https://matplotlib.org/stable/plot_types/basic/plot.html)
- search string: [subplot](https://www.google.com/search?q=subplot&oq=subplot&gs_lcrp=EgZjaHJvbWUqBggAEEUYOzIGCAAQRRg7MgYIARAjGCcyCggCEAAYsQMYgAQyDQgDEAAYgwEYsQMYgAQyBggEEEUYPDIGCAUQRRg8MgYIBhBFGDwyBggHEEUYQdIBBzg2NWowajeoAgCwAgA&sourceid=chrome&ie=UTF-8)
- [matplotlib.pyplot.subplot](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplot.html)
#### subplots
##### colors
- search string: [fig.patch.set_facecolor('xkcd:blue')](https://www.google.com/search?q=fig.patch.set_facecolor(%27xkcd%3Ablue%27)&oq=fig.patch.set_facecolor(%27xkcd%3Ablue%27)&gs_lcrp=EgZjaHJvbWUyBggAEEUYOdIBBzU3NGowajeoAgCwAgA&sourceid=chrome&ie=UTF-8)
- [xkcd.com/color/rgb/](https://xkcd.com/color/rgb/)
- search string: [fig, axs = plt.subplots(2, 2)](https://www.google.com/search?q=fig%2C+axs+%3D+plt.subplots(2%2C+2)&oq=fig%2C+axs+%3D+plt.subplots(2%2C+2)&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIHCAEQABiABDIICAIQABgWGB4yBwgDEAAYhgPSAQc4MjRqMGo3qAIAsAIA&sourceid=chrome&ie=UTF-8)
- [Creating multiple subplots using plt.subplots >> Stacking subplots in two directions](https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html)
#### other matplotlib
##### boxplot
- [matplotlib.pyplot.boxplot](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.boxplot.html)
- [matplotlib.pyplot.boxplot [deprecated]](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.boxplot.html)
- search string [boxplot pyplot](https://www.google.com/search?q=boxplot+pyplot&oq=boxplot+pyplot&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTILCAEQABgWGB4Y8QQyCAgCEAAYFhgeMggIAxAAGBYYHjIICAQQABgWGB4yCwgFEAAYFhgeGPEEMgsIBhAAGBYYHhjxBDILCAcQABgWGB4Y8QQyCwgIEAAYFhgeGPEEMgsICRAAGBYYHhjxBNIBCDY2NDZqMGo3qAIAsAIA&sourceid=chrome&ie=UTF-8)
##### histplot
- [matplotlib.pyplot.hist](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html)
- search string [matplotlib.pyplot histogram](https://www.google.com/search?q=matplotlib.pyplot+histogram&newwindow=1&sxsrf=AJOqlzVGsD20ZAypbaqD47k1A9gAJNR0ug%3A1675939330927&ei=As7kY52cOOW58gKh-Z7oBw&oq=matplotlib.pyplot+&gs_lcp=Cgxnd3Mtd2l6LXNlcnAQARgAMgoIABCABBAUEIcCMgUIABCABDIFCAAQgAQyBQgAEIAEMgUIABCABDIFCAAQgAQyBQgAEIAEMgUIABCABDIFCAAQgAQyBQgAEIAEOg4IABCABBCxAxCDARCwAzoNCAAQgAQQFBCHAhCwAzoICAAQgAQQsAM6CQgAEAcQHhCwA0oECEEYAUoECEYYAFCYB1iYB2DWEWgBcAB4AIABTYgBTZIBATGYAQCgAQHIAQrAAQE&sclient=gws-wiz-serp)
- [Histogram with Boxplot above in Python](https://stackoverflow.com/questions/33381330/histogram-with-boxplot-above-in-python)
- search string [histogram_boxplot matplotlib](https://www.google.com/search?q=histogram_boxplot+matplotlib&newwindow=1&sxsrf=AJOqlzWw29as3Nymo_ZtGfRt-TMyNd9yAA%3A1675938211495&ei=o8nkY5vzHZPD8gK2yb2AAw&ved=0ahUKEwjb0IuunIj9AhWToVwKHbZkDzAQ4dUDCA8&uact=5&oq=histogram_boxplot+matplotlib&gs_lcp=Cgxnd3Mtd2l6LXNlcnAQAzIJCAAQFhAeEPEEMgUIABCGAzIFCAAQhgM6CggAEEcQ1gQQsAM6BwgAEA0QgAQ6BwgAEB4Q8QQ6BggAEB4QDzoJCAAQCBAeEPEEOgYIABAIEB46CAgAEBYQHhAPOgYIABAWEB46CwgAEAgQHhANEPEESgQIQRgASgQIRhgAULYRWKdCYMNEaAJwAXgAgAGgAYgB_AiSAQM5LjOYAQCgAQHIAQjAAQE&sclient=gws-wiz-serp)
### error
- search string [Non-default argument follows default argumentPylance](https://www.google.com/search?q=Non-default+argument+follows+default+argumentPylance&oq=Non-default+argument+follows+default+argumentPylance&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIJCAEQABgKGIAEMgoIAhAAGAoYFhgeMgcIAxAAGIYDMgcIBBAAGIYDMgcIBRAAGIYDMgcIBhAAGIYD0gEHNjcyajBqN6gCALACAA&sourceid=chrome&ie=UTF-8)
- [SyntaxError: non-default argument follows default argument](https://stackoverflow.com/questions/24719368/syntaxerror-non-default-argument-follows-default-argument)
### scipy
### scipy.stats
### statsmodels
- [statsmodels.stats.proportion.proportions_ztest](https://www.statsmodels.org/dev/generated/statsmodels.stats.proportion.proportions_ztest.html)
- search string: [what are model predictors statsmodels](https://www.google.com/search?q=add+color+using+nb+black&oq=add+color+using+nb+black&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIHCAEQABiiBDIHCAIQABiiBDIHCAMQABiiBNIBCDU5NjhqMGo3qAIAsAIA&sourceid=chrome&ie=UTF-8)
- [Prediction (out of sample) | statsmodels.org | statsmodels | Documentation](https://www.statsmodels.org/dev/examples/notebooks/generated/predict.html)
### scikit-learn
#### Documentation
- [search string: sklearn](https://www.google.com/search?q=sklearn&oq=sklearn&gs_lcrp=EgZjaHJvbWUqBggAEEUYOzIGCAAQRRg7MgYIARAjGCcyBggCEAAYQzIGCAMQABhDMgYIBBAAGEMyBggFEAAYQzIGCAYQRRg8MgYIBxBFGDzSAQc3MzVqMGo3qAIAsAIA&sourceid=chrome&ie=UTF-8)
- [scikit-learn | Machine Learning in Python](https://scikit-learn.org/stable/)
- [Getting Started -- skikit-learn](https://scikit-learn.org/stable/getting_started.html)
- [Citing scikit-learn](https://scikit-learn.org/stable/about.html#citing-scikit-learn)
- [User Guide](https://scikit-learn.org/stable/user_guide.html#user-guide)
- [Installing scikit-learn](https://scikit-learn.org/stable/install.html)
- Scikit-learn: Machine Learning in Python [Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.](https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html)
- redirects to https://scikit-learn.org/stable/ (link 2 in this section, above) [Source code, binaries, and documentation](http://scikit-learn.sourceforge.net)
### ipykernel
- [search string: ipykernel](https://www.google.com/search?q=ipykernel&oq=ipykernel&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIHCAEQABiABDIHCAIQABiABDIHCAMQABiABDIHCAQQABiABDIHCAUQABiABDIHCAYQABiABDIHCAcQABiABDIMCAgQABgUGIcCGIAEMgcICRAAGIAE0gEHNDUzajBqN6gCALACAA&sourceid=chrome&ie=UTF-8)
- pip install ipykernel [ipykernel 6.19.2](https://pypi.org/project/ipykernel/)
## colors for jupyter notebook charts
- search string: [pandas plot frame color -matplotlib](https://www.google.com/search?q=pandas+plot+frame+color+-matplotlib&newwindow=1&sxsrf=AJOqlzWPmi_tMpOW7pQfQRSNTlnG2AeQsQ%3A1676033565587&ei=HT7mY6-_I8vzgAa4iKuYDg&ved=0ahUKEwjvwbvK_4r9AhXLOcAKHTjECuMQ4dUDCA8&uact=5&oq=pandas+plot+frame+color+-matplotlib&gs_lcp=Cgxnd3Mtd2l6LXNlcnAQAzoKCAAQRxDWBBCwAzoJCAAQFhAeEPEEOgUIABCGA0oECEEYAEoECEYYAFDIBlj9GWCPG2gBcAF4AIABX4gBvQaSAQIxMpgBAKABAcgBCMABAQ&sclient=gws-wiz-serp)
- [Pandas - Plotting](https://www.w3schools.com/python/pandas/pandas_plotting.asp)
- [pandas.DataFrame.plot](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html)
- search string: [pandas plot](https://www.google.com/search?q=pandas+plot&oq=pandas+plot&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIMCAEQABgUGIcCGIAEMgcIAhAAGIAEMgcIAxAAGIAEMgcIBBAAGIAEMgcIBRAAGIAEMgcIBhAAGIAEMgYIBxBFGEHSAQgyNzU5ajBqN6gCALACAA&sourceid=chrome&ie=UTF-8)
- [pandas.crosstab](https://pandas.pydata.org/docs/reference/api/pandas.crosstab.html)
- search string [pd crosstab](https://www.google.com/search?q=pd+crosstab&oq=pd+crosstab&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIGCAEQIxgnMgYIAhAjGCcyCQgDEAAYChiABDIOCAQQABgKGBQYhwIYgAQyCQgFEAAYChiABDIGCAYQRRg8MgYIBxBFGEHSAQg2ODU2ajBqN6gCALACAA&sourceid=chrome&ie=UTF-8)
- ANSWER to color: [seaborn.set_style()](https://stackoverflow.com/questions/30729473/seaborn-legend-with-background-color)
- search string: [sns seaborn color frame facecolor](https://www.google.com/search?q=sns+seaborn+color+frame+facecolor&newwindow=1&sxsrf=AJOqlzVAs2RY94np7bRieSF4g4kEWQelZw%3A1676027522466&ei=gibmY-iQHISg8gKG-om4Bg&ved=0ahUKEwjo2fCI6Yr9AhUEkFwKHQZ9AmcQ4dUDCA8&uact=5&oq=sns+seaborn+color+frame+facecolor&gs_lcp=Cgxnd3Mtd2l6LXNlcnAQAzIKCAAQ8QQQHhCiBDIFCAAQogQyBQgAEKIEMgUIABCiBDoKCAAQRxDWBBCwAzoFCCEQoAE6BwghEKABEApKBAhBGABKBAhGGABQ6QFY5xZguRpoAXABeACAAYMBiAHfB5IBAzYuNJgBAKABAcgBCMABAQ&sclient=gws-wiz-serp)
- [seaborn.set_theme](https://seaborn.pydata.org/generated/seaborn.set_theme.html)
- search string: [sns.set_theme(style="whitegrid")](https://www.google.com/search?q=sns.set_theme(style%3D%22whitegrid%22)&oq=sns.set_theme(style%3D%22whitegrid%22)&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIHCAEQABiABNIBBzU1OWowajeoAgCwAgA&sourceid=chrome&ie=UTF-8)
- [seaborn.countplot | sns.set_theme(style="whitegrid")](https://seaborn.pydata.org/generated/seaborn.countplot.html)
- search string: [countplot sns perc](https://www.google.com/search?q=countplot+sns+perc&oq=countplot+sns+perc&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTILCAEQABgWGB4Y8QQyCggCEAAYChgWGB4yCwgDEAAYFhgeGPEEMgcIBBAAGIYDMgcIBRAAGIYDMgcIBhAAGIYDMgcIBxAAGIYD0gEIODM0N2owajeoAgCwAgA&sourceid=chrome&ie=UTF-8)
- [Actually, really change all of the background color | fig, ax = plt.subplots(facecolor='lightslategray'); | df.plot(ax=ax, color='white')](https://jonathansoma.com/lede/data-studio/matplotlib/changing-the-background-of-a-pandas-matplotlib-graph/)

- [Change the facecolor of boxplot in pandas | stackoverflow](https://stackoverflow.com/questions/39297093/change-the-facecolor-of-boxplot-in-pandas)
- search string: [pandas facecolor](https://www.google.com/search?q=pandas+facecolor&oq=pandas+facecolor&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTILCAEQABgWGB4Y8QQyCwgCEAAYFhgeGPEEMggIAxAAGBYYHjIHCAQQABiGAzIHCAUQABiGAzIHCAYQABiGA9IBCDUzOTVqMGo3qAIAsAIA&sourceid=chrome&ie=UTF-8)
- [jsfiddle | iterate through object properties](https://jsfiddle.net/tbynA/1/)
- [Recursively looping through an object to build a property list | stackoverflow](https://stackoverflow.com/questions/15690706/recursively-looping-through-an-object-to-build-a-property-list)
- search string: [how to recursively return all levels of an object](https://www.google.com/search?q=how+to+recursively+return+all+levels+of+an+object&oq=how+to+recursively+return+all+levels+of+an+object&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIGCAEQRRhA0gEIODMwM2owajeoAgCwAgA&sourceid=chrome&ie=UTF-8)
- search string: [is matplotlib. pyplot an object?](https://www.google.com/search?q=is+matplotlib.+pyplotan+object%3F&oq=is+matplotlib.+pyplotan+object%3F&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIGCAEQRRhA0gEIMzgxNmowajeoAgCwAgA&sourceid=chrome&ie=UTF-8)
- search string: [matplotlib pyplot plt](https://www.google.com/search?q=matplotlib+pyplot+plt&oq=matplotlib+pyplot+plt&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIICAEQABgWGB4yCAgCEAAYFhgeMggIAxAAGBYYHjIICAQQABgWGB4yCAgFEAAYFhgeMggIBhAAGBYYHjIGCAcQRRg80gEINjg5MGowajeoAgCwAgA&sourceid=chrome&ie=UTF-8)
- [pandas.crosstab | pandas | Documentation](https://pandas.pydata.org/docs/reference/api/pandas.crosstab.html)
- search string: [pd.crosstab color](https://www.google.com/search?q=pd.crosstab+color&newwindow=1&bih=575&biw=1097&hl=en&sxsrf=AJOqlzUEG9wlmJQdqdCH5QwYBcOICwkOEw%3A1676030036392&ei=VDDmY-XQF8mpgQbX3q2QAQ&ved=0ahUKEwjlys638or9AhXJVMAKHVdvCxIQ4dUDCA8&uact=5&oq=pd.crosstab+color&gs_lcp=Cgxnd3Mtd2l6LXNlcnAQAzIFCAAQgAQyBQgAEIYDMgUIABCGAzIFCAAQhgMyBQgAEIYDOgcIIxCwAxAnOgoIABBHENYEELADOgcIIxCwAhAnOgcIABANEIAEOgYIABAHEB46CQgAEAcQHhDxBDoHCCMQsQIQJzoECCMQJzoFCAAQkQI6BggAEBYQHjoJCAAQFhAeEPEESgQIQRgASgQIRhgAUNIdWPlcYPNeaARwAXgAgAGaAYgBowiSAQQxMS4xmAEAoAEByAEKwAEB&sclient=gws-wiz-serp)
- saved search string: (autocomplete) [pd.crosstab df normalize='index').plot(kind="bar", figsize=(6,8),stacked=True)](<a href="https://www.google.com/search?q=pd.crosstab+df+normalize%3D%27index%27).plot(kind%3D%22bar%22%2C+figsize%3D(6%2C8)%2Cstacked%3DTrue)&oq=pd.crosstab+df+normalize%3D%27index%27).plot(kind%3D%22bar%22%2C+figsize%3D(6%2C8)%2Cstacked%3DTrue)&gs_lcrp=EgZjaHJvbWUqBggAEEUYOzIGCAAQRRg70gEIMTYxOGowajeoAgCwAgA&sourceid=chrome&ie=UTF-8">link</a>)
-[Creating Links in Markdown](https://anvilproject.org/guides/content/creating-links)

- [[deprecated] | matplotlib.pyplot.figure | matplotlib | Documentation](https://matplotlib.org/2.1.1/api/_as_gen/matplotlib.pyplot.figure.html)
- [Elegantly changing the color of a plot frame in matplotlib | fig, axes = plt.subplots(nrows=2); | axes[0].plot(range(10), 'r-'); | axes[1].plot(range(10), 'bo-'); | stackoverflow](https://stackoverflow.com/questions/7778954/elegantly-changing-the-color-of-a-plot-frame-in-matplotlib)
- search string: [ply.figure frame color](https://www.google.com/search?q=ply.figure+frame+color&oq=ply.figure+frame+color&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIJCAEQIRgKGKABMgkIAhAhGAoYoAEyCQgDECEYChigAdIBCDg1NDFqMGo3qAIAsAIA&sourceid=chrome&ie=UTF-8)
- [How to change plot background color?](https://stackoverflow.com/questions/14088687/how-to-change-plot-background-color)

- [How do I plot two countplot graphs side by side in seaborn? | fig, ax =plt.subplots(1,2); | sns.countplot(df['batting'], ax=ax[0]); | sns.countplot(df['bowling'], ax=ax[1]); | fig.show() | stackoverflow](https://stackoverflow.com/questions/43131274/how-do-i-plot-two-countplot-graphs-side-by-side-in-seaborn)
- [countplot sns subplot](https://www.google.com/search?q=countplot+sns+subplot&oq=countplot+sns+subplot&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTILCAEQABgWGB4Y8QQyBwgCEAAYhgMyBwgDEAAYhgMyBwgEEAAYhgMyBwgFEAAYhgPSAQg1MTc5ajBqN6gCALACAA&sourceid=chrome&ie=UTF-8)
- [How to prevent overlapping x-axis labels in sns.countplot | code: | plt.figure(figsize=(15,10)) #adjust the size of plot; | ax=sns.countplot(x=df['Location'],data=df,hue='label',palette='mako'); | stackoverflow](https://stackoverflow.com/questions/42528921/how-to-prevent-overlapping-x-axis-labels-in-sns-countplot)
- search string: [countplot | recursively unpacck ax in sns countplot](https://www.google.com/search?q=recursively+unpacck+ax+in+sns+countplot&oq=recursively+unpacck+ax+in+sns+countplot&gs_lcrp=EgZjaHJvbWUyBggAEEUYOdIBCTE2MDAzajBqN6gCALACAA&sourceid=chrome&ie=UTF-8)
- [Countplot using seaborn in Python | geeksforgeeks](https://www.geeksforgeeks.org/countplot-using-seaborn-in-python/)
- search string: [countplot sns ax frame](https://www.google.com/search?q=countplot+sns+ax+frame&oq=countplot+sns++ax+frame&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIHCAEQIRigAdIBCDcwNzBqMGo3qAIAsAIA&sourceid=chrome&ie=UTF-8)
- [seaborn.countplot | content: | kwargs : key, value mappings | Other keyword arguments are passed through to matplotlib.axes.Axes.bar(). | Returns: | axmatplotlib Axes | Returns the Axes object with the plot drawn onto it. | seaborn | Documentation](https://seaborn.pydata.org/generated/seaborn.countplot.html)
- search string: [countplot sns](https://www.google.com/search?q=countplot+sns&oq=&gs_lcrp=EgZjaHJvbWUqBggAEEUYOzIGCAAQRRg7MgYIARBFGDsyBwgCEAAYgAQyBwgDEAAYgAQyBwgEEAAYgAQyBwgFEAAYgAQyBwgGEAAYgAQyBggHEEUYPNIBCDI0NDhqMGo3qAIAsAIA&sourceid=chrome&ie=UTF-8)

## .venv error [Resolved]
- find in page: | your path [PermissionError: [Errno 13] Permission denied | terminal error trying to install preinstalled .venv | stackoverflow](https://stackoverflow.com/questions/36434764/permissionerror-errno-13-permission-denied)
- search string: [Error: [Errno 13] Permission denied: 'C:\\Users\\OneDrive\\Documents\\.venv\\Scripts\\python.exe'](https://www.google.com/search?q=Error%3A+%5BErrno+13%5D+Permission+denied%3A+%27C%3A%5C%5CUsers%5C%5COneDrive%5C%5CDocuments%5C%5C.venv%5C%5CScripts%5C%5Cpython.exe%27&oq=Error%3A+%5BErrno+13%5D+Permission+denied%3A+%27C%3A%5C%5CUsers%5C%5COneDrive%5C%5CDocuments%5C%5C.venv%5C%5CScripts%5C%5Cpython.exe%27&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIGCAEQRRg60gEHNzMwajBqN6gCALACAA&sourceid=chrome&ie=UTF-8)

## 0 Axes error [Resolved]
- to remove with 0 Axes: comment line: | plt.figure(facecolor='blue').set_facecolor('xkcd:cerulean blue')
[I used matplotlib, but the error message '<Figure size 720x576 with 0 Axes>' appeared with graph](https://stackoverflow.com/questions/52834616/i-used-matplotlib-but-the-error-message-figure-size-720x576-with-0-axes-app) 

## save Pandas dataframe/series data to figure then to file
- [fig.savefig('asdf.png')](https://stackoverflow.com/questions/19726663/how-to-save-the-pandas-dataframe-series-data-as-a-figure)
- [How to save the Pandas dataframe from pd.crosstab as a figure (with render_mpl_table)? | stackoverflow](https://stackoverflow.com/questions/72039213/how-to-save-the-pandas-dataframe-from-pd-crosstab-as-a-figure-with-render-mpl-t)
- search string: [pd.crosstab "set_facecolor"](https://www.google.com/search?newwindow=1&sxsrf=AJOqlzW3SDsHl3u-1f9e3ewMZHbuZa-q5Q:1676028618400&q=pd.crosstab+%22set_facecolor%22&sa=X&ved=2ahUKEwjkmbuT7Yr9AhXZHcAKHdikA_gQ5t4CegQIHBAB&biw=1097&bih=575&dpr=1.75)
- [ResidentMario / missingno | Issue | Matplotlib error: 'AxesSubplot' object has no attribute 'set_facecolor' #25 | GitHub](https://github.com/ResidentMario/missingno/issues/25)
- search string: [AttributeError: 'DataFrame' object has no attribute 'set_facecolor'](https://www.google.com/search?q=AttributeError%3A+%27DataFrame%27+object+has+no+attribute+%27set_facecolor%27&oq=AttributeError%3A+%27DataFrame%27+object+has+no+attribute+%27set_facecolor%27&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIGCAEQRRg60gEHNzU5ajBqN6gCALACAA&sourceid=chrome&ie=UTF-8)
- [scikit-learn/scikit-learn | Issue| 'AxesSubplot' object has no attribute 'set_axis_bgcolor' #10762 | GitHub](https://github.com/scikit-learn/scikit-learn/issues/10762)
- search string: [AttributeError: 'DataFrame' object has no attribute 'set_axis_bgcolor'](https://www.google.com/search?q=AttributeError%3A+%27DataFrame%27+object+has+no+attribute+%27set_axis_bgcolor%27&oq=AttributeError%3A+%27DataFrame%27+object+has+no+attribute+%27set_axis_bgcolor%27&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIGCAEQRRg60gEHNzY3ajBqN6gCALACAA&sourceid=chrome&ie=UTF-8)
## Statistics

## pandas print statement
- [turn off automatic pandas data type output on print statment](https://stackoverflow.com/questions/29645153/remove-name-dtype-from-pandas-output-of-dataframe-or-series)

## python
### main.py (files 1 to 4) and script.sh in CoderSales/machine-learning-classification (repository reference below)
- repository reference [CoderSales/machine-learning-classification](https://github.com/CoderSales/machine-learning-classification)
- [slice strings in python](https://www.w3schools.com/python/gloss_python_string_slice.asp)
- [Check if Python Package is installed](https://stackoverflow.com/questions/1051254/check-if-python-package-is-installed)
- [pip install notebook](https://jupyter.org/install)
- [How to Execute Shell Commands with Python](https://janakiev.com/blog/python-shell-commands/)
- [How to print a string literally in Python](https://stackoverflow.com/questions/6903551/how-to-print-a-string-literally-in-python)
- [4 ways to add variables or values into Python strings](https://medium.com/analytics-vidhya/4-ways-to-add-variables-or-values-into-python-strings-860082cf8461)
- search string: [percentage symbol pip bash](https://www.google.com/search?q=percentage+symbol+pip+bash&newwindow=1&sxsrf=AJOqlzWVNEAC2sWl-_Fd1EM9HLo8UPFV4Q%3A1676309809136&ei=MXXqY4WBCJqTgQb0oIrgDQ&ved=0ahUKEwjF3dPVhJP9AhWaScAKHXSQAtwQ4dUDCA8&uact=5&oq=percentage+symbol+pip+bash&gs_lcp=Cgxnd3Mtd2l6LXNlcnAQAzIFCCEQoAEyBQghEKABOgcIIxCwAxAnOgoIABBHENYEELADOgQIIxAnSgQIQRgASgQIRhgAULsHWNgPYLIRaAFwAXgAgAGWAYgB_QOSAQM0LjGYAQCgAQHIAQnAAQE&sclient=gws-wiz-serp)
- search string: [python access "Option -c 4"](https://www.google.com/search?q=python+access+%22Option+-c+4%22&newwindow=1&sxsrf=AJOqlzWfGXa5nLwxEAeDtoihI2XQhYCEow%3A1676309333868&ei=VXPqY_K9NNWM8gK7rocI&ved=0ahUKEwiyvIPzgpP9AhVVhlwKHTvXAQEQ4dUDCA8&uact=5&oq=python+access+%22Option+-c+4%22&gs_lcp=Cgxnd3Mtd2l6LXNlcnAQAzoKCAAQRxDWBBCwA0oECEEYAEoECEYYAFDGCFjGCGC2C2gBcAF4AIABOogBOpIBATGYAQCgAQHIAQjAAQE&sclient=gws-wiz-serp)
- [How to Execute Shell Commands with Python](https://janakiev.com/blog/python-shell-commands/)
- import subprocess | subprocess.run('/path/to/script.sh', check=True) [os.system() | run all shell commands with a single call](https://stackoverflow.com/questions/53151899/python-how-to-script-virtual-environment-building-and-activation)
### storing variables
#### naming arbitrary number of variables
- [used for first attempt at naming arbitrary number of variables](https://stackoverflow.com/questions/48372808/create-an-unknown-number-of-programmatically-defined-variables)
- [second attempt at naming arbitrary number of variables](https://pythonprinciples.com/ask/how-do-you-create-a-variable-number-of-variables/)

#### append
- [.append()](https://realpython.com/python-append/#:~:text=Python%20provides%20a%20method%20called,list%20using%20a%20for%20loop.)

### pass multiple variables into string
- [pass multiple variables into string](https://stackoverflow.com/questions/10112614/how-do-i-create-a-multiline-python-string-with-inline-variables)

### multiline string python
- [Python Multiline Strings](https://www.w3schools.com/python/gloss_python_multi_line_strings.asp)

### How do you add value to a key in Python?
- 'a':'0' [How do you add value to a key in Python?](https://www.mygreatlearning.com/blog/python-dictionary-append/#:~:text=How%20do%20you%20add%20value,new%20values%20to%20the%20keys.)

### pass variable into string variable 
- [pass variable into string variable](https://matthew-brett.github.io/teaching/string_formatting.html)

### turn off pandas index output
- [Remove name, dtype from pandas output of dataframe or series](https://stackoverflow.com/questions/29645153/remove-name-dtype-from-pandas-output-of-dataframe-or-series)
- [2ndary source for turning off index on pandas dataframe print out](https://stackoverflow.com/questions/24644656/how-to-print-pandas-dataframe-without-index)

### concatenate
- [concatenate with +](https://www.digitalocean.com/community/tutorials/python-string-concatenation)

### String into variable
- [String Into Variable Name in Python Using the vars() Function](https://www.pythonforbeginners.com/basics/convert-string-to-variable-name-in-python#:~:text=is%20pythonforbeginners.com-,String%20Into%20Variable%20Name%20in%20Python%20Using%20the%20vars(),like%20the%20globals()%20function.)
- [Convert string to variable name in python [duplicate]](https://stackoverflow.com/questions/19122345/convert-string-to-variable-name-in-python)

- option used [Python Template String Formatting Method](https://towardsdatascience.com/python-template-string-formatting-method-df282510a87a)

### .update() a dictionary
- [Python dictionary append: How to add Key-value Pair?](https://www.mygreatlearning.com/blog/python-dictionary-append/)
- [Python Dictionary update() Method](https://www.w3schools.com/python/ref_dictionary_update.asp)
- [EDA: def is_categorical | Update dictionary items with a for loop | stackoverflow](https://stackoverflow.com/questions/31069955/update-dictionary-items-with-a-for-loop)

### print separate with no spaces
- [Print without space in python 3](https://stackoverflow.com/questions/12700558/print-without-space-in-python-3)

### function
- [anatomy of a function in python](https://www.google.com/search?q=anatomy+of+a+function+in+python&oq=anatomy+of+a+function+in++python&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIHCAEQABiiBDIHCAIQABiiBDIHCAMQABiiBDIJCAQQABgeGKIE0gEJMTQ1MjJqMGo3qAIAsAIA&sourceid=chrome&ie=UTF-8)
- [Functions](https://geo-python.github.io/2017/lessons/L4/functions.html#:~:text=Anatomy%20of%20a%20function&text=The%20function%20definition%20opens%20with,indented%20below%20the%20definition%20line.)
## ML
### Linear Regression

### Logistic Regression

## Statistics for ML (Logistic Regression)
- detailed confusion matrix [Precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall#:~:text=Recall%20in%20this%20context%20is%20also%20referred%20to%20as%20the,rate%20is%20also%20called%20specificity.)
 - used for calculation of F1 score [Harmonic mean](https://en.wikipedia.org/wiki/Harmonic_mean#Harmonic_mean_of_two_numbers)
 - image [Geometric proof without words that max (a,b) > root mean square (RMS) or quadratic mean (QM) > arithmetic mean (AM) > geometric mean (GM) > harmonic mean (HM) > min (a,b) of two distinct positive numbers a and b](https://en.wikipedia.org/wiki/Harmonic_mean#/media/File:QM_AM_GM_HM_inequality_visual_proof.svg)
 - image [QM_AM_GM_HM_inequality_visual_proof.svg/2560px-QM_AM_GM_HM_inequality_visual_proof.svg.png](https://upload.wikimedia.org/wikipedia/commons/thumb/a/a1/QM_AM_GM_HM_inequality_visual_proof.svg/2560px-QM_AM_GM_HM_inequality_visual_proof.svg.png)
 ### F-beta score: sklearn documentation
 - Search string: [F-beta score](https://www.google.com/search?q=F-beta+score&oq=F-beta+score&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIGCAEQRRg7MggIAhAAGBYYHjIICAMQABgWGB4yCAgEEAAYFhgeMggIBRAAGBYYHjIICAYQABgWGB4yBggHEEUYPNIBCDE4ODFqMGo3qAIAsAIA&sourceid=chrome&ie=UTF-8)
 - Search string: [F-beta score is the weighted harmonic mean of precision and recall](https://www.google.com/search?q=F-beta+score+is+the+weighted+harmonic+mean+of+precision+and+recall,&source=lmns&bih=808&biw=1552&hl=en&sa=X&ved=2ahUKEwiOlv2ZqYT9AhU4nCcCHcS_COMQ_AUoAHoECAEQAA)
 - Search string: [f2 ml sklearn](https://www.google.com/search?q=f2+ml+sklearn&newwindow=1&sxsrf=AJOqlzX0pT5Uc4oPuqHgd-hjnAFUNKH-WQ%3A1675622250460&ei=avffY7fnG9WEhbIPrdqIyAw&ved=0ahUKEwi3xJCog__8AhVVQkEAHS0tAskQ4dUDCBA&uact=5&oq=f2+ml+sklearn&gs_lcp=Cgxnd3Mtd2l6LXNlcnAQAzIFCAAQogQyBQgAEKIEMgUIABCiBDoLCAAQsQMQsAMQkQI6BwgAELADEEM6DQgAELEDEIMBELADEEM6CAgAELADEJECOg0IABDkAhDWBBCwAxgBOg8ILhDUAhDIAxCwAxBDGAI6DAguEMgDELADEEMYAjoECAAQQzoHCAAQsQMQQzoFCAAQgAQ6CwguEIAEEMcBEK8BOgYIABAWEB46BQgAEIYDOgUIIRCgAToHCAAQDRCABDoGCAAQHhANOgQIIRAVOgcIIRCgARAKSgQIQRgBSgQIRhgBUJm6Alir_gNgwIIEaAFwAHgAgAFfiAHzBpIBAjExmAEAoAEByAETwAEB2gEGCAEQARgJ2gEGCAIQARgI&sclient=gws-wiz-serp)
 - fbeta_score [sklearn.metrics.fbeta_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html#:~:text=The%20F%2Dbeta%20score%20is,recall%20in%20the%20combined%20score.)
 - fbeta_score [sklearn.metrics.fbeta_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html)
 ### F score
 - [F score](https://en.wikipedia.org/wiki/F-score)
 
# References Part5 / (other, VSCODE workflow window views, HTML, CSS, IMG)
## VSCODE workflow window views
- Keyboard Shortcuts > workbench.action.duplicateWorkspaceInNewWindow Ctrl Shift Alt N (modified from suggested on site) [VSCODE workflow window views](https://stackoverflow.com/questions/43362133/visual-studio-code-open-tab-in-new-window)

## font
## HTML

## CSS
- not used [box-shadow: red](https://stackoverflow.com/questions/61476773/how-to-add-a-background-square-behind-the-image)
- used [change body tag background color behind image](https://stackoverflow.com/questions/7415872/change-color-of-png-image-via-css)
- search string: [css font color](https://www.google.com/search?q=css+font+color&oq=css+font+color&gs_lcrp=EgZjaHJvbWUqCggAEAAYsQMYgAQyCggAEAAYsQMYgAQyBwgBEAAYgAQyBwgCEAAYgAQyBwgDEAAYgAQyBwgEEAAYgAQyBwgFEAAYgAQyBwgGEAAYgAQyBggHEEUYQdIBCDIwODFqMGo3qAIAsAIA&sourceid=chrome&ie=UTF-8)
- [CSS Text](https://www.w3schools.com/css/css_text.asp)

## nb-black / jupyter notebook formatting
- search string: [add color using nb black](https://www.google.com/search?q=add+color+using+nb+black&oq=add+color+using+nb+black&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIHCAEQABiiBDIHCAIQABiiBDIHCAMQABiiBNIBCDU5NjhqMGo3qAIAsAIA&sourceid=chrome&ie=UTF-8)
- <font color='red'>bar</font> [How to change color in markdown cells ipython/jupyter notebook? | stackoverflow](https://stackoverflow.com/questions/19746350/how-to-change-color-in-markdown-cells-ipython-jupyter-notebook)

## Images
## IMG
- not used [to crop images in css](https://www.digitalocean.com/community/tutorials/css-cropping-images-object-fit)
## SVG
- [harmonic mean .svg file](https://upload.wikimedia.org/wikipedia/commons/f/f7/MathematicalMeans.svg)
- [harmonic mean .svg file page 2](https://en.wikipedia.org/wiki/File:MathematicalMeans.svg)
- [means visual proof](https://en.wikipedia.org/wiki/Harmonic_mean#/media/File:QM_AM_GM_HM_inequality_visual_proof.svg)
- [How to edit color via code of svg file with: open svg file in explorer > inspect element > Elements > edit circle tag fill attribute](https://medium.com/@nick.cqx/illustrate-your-project-without-any-graphic-design-software-using-svg-and-your-browser-20e9a73b53a3)

## Repositories
- [ResidentMario/matplotlib](https://github.com/ResidentMario/matplotlib)
# References Part 6 / (bash, shell scripting)
- import subprocess [Python: How to script virtual environment building and activation?](https://stackoverflow.com/questions/53151899/python-how-to-script-virtual-environment-building-and-activation)
- Put this in main.py: | import yoursubfile | Treat it like a module: import file.[How can I make one python file run another? [duplicate] | Get one python file to run another, using python 2.7.3 and Ubuntu 12.10:](https://stackoverflow.com/questions/7974849/how-can-i-make-one-python-file-run-another)
## subprocess file calls
- [How to add images to README.md on GitHub?](https://stackoverflow.com/questions/14494747/how-to-add-images-to-readme-md-on-github)
![wireframe](https://github.com/CoderSales/machine-learning-classification/blob/main/img/wireframe-bash-py-scripts-2.png)
- The error is pretty clear. The file hello.py is not an executable file. You need to specify the executable: subprocess.call(['python.exe', 'hello.py', 'htmlfilename.htm'])
[OSError: [WinError 193] %1 is not a valid Win32 application](https://stackoverflow.com/questions/25651990/oserror-winerror-193-1-is-not-a-valid-win32-application)
- [Python Exception <TypeError>: bufsize must be an integer](https://community.safe.com/s/question/0D54Q00009jkcMHSAY/python-exception-typeerror-bufsize-must-be-an-integer)
- Using the subprocess Module | python 3.11.2 [subprocess — Subprocess management | Using the subprocess Module | python 3.11.2 ](https://docs.python.org/3/library/subprocess.html)
- [How can I make one python file run another? [duplicate]](https://stackoverflow.com/questions/7974849/how-can-i-make-one-python-file-run-another)
- [How to call a shell script from python code?](https://stackoverflow.com/questions/3777301/how-to-call-a-shell-script-from-python-code)
- Your best option would be to do it in a function

- activate () {  . ../.env/bin/activate} [How to source virtualenv activate in a Bash script](https://stackoverflow.com/questions/13122137/how-to-source-virtualenv-activate-in-a-bash-script)
- def my_function(): [Python Functions](https://www.w3schools.com/python/python_functions.asp)
- Main result: If you want to ignore a file that you've committed in the past, you'll need to delete the file from your repository and then add a .gitignore rule for it. | search string: [how to add files to gitignore](https://www.google.com/search?q=how+to+add+files+to+gitignore&oq=how+to+add+files+to+gitignore&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIHCAEQABiABDIHCAIQABiABDIHCAMQABiABDIHCAQQABiABDIHCAUQABiABDIHCAYQABiABDIMCAcQABgUGIcCGIAEMgcICBAAGIAEMgcICRAAGIYD0gEINTA3N2owajeoAgCwAgA&sourceid=chrome&ie=UTF-8)
- [Ignoring a previously committed file](https://www.atlassian.com/git/tutorials/saving-changes/gitignore#:~:text=If%20you%20want%20to%20ignore,directory%20as%20an%20ignored%20file.)
- JavaScript function definition syntax (uses curly brackets like bash syntax)[Function.prototype.apply()](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Function/apply)
- [site to find out which language code is written in](https://dpaste.com/)
- [Is there a website that can recognize and identify what programming language is being input (pasted)?](https://www.quora.com/Is-there-a-website-that-can-recognize-and-identify-what-programming-language-is-being-input-pasted)
- search string: ['.' is not recognized as an internal or external command,](https://www.google.com/search?q=%27.%27+is+not+recognized+as+an+internal+or+external+command%2C&oq=%27.%27+is+not+recognized+as+an+internal+or+external+command%2C&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIHCAEQABiABDIHCAIQABiABDIHCAMQABiABDIHCAQQABiABDIHCAUQABiABDIGCAYQRRhBMgYIBxBFGEHSAQgxMDk0ajBqN6gCALACAA&sourceid=chrome&ie=UTF-8)
- [5 Ways to Fix the "Not Recognized as an Internal or External Command" Error in Windows](https://www.makeuseof.com/windows-not-recognized-as-an-internal-or-external-command-error/#:~:text=You%20can%20resolve%20this%20issue,files%20to%20the%20System32%20folder.)
- search string: [subprocess.Popen() documentation](https://www.google.com/search?q=subprocess.Popen()+documentation&sourceid=chrome&ie=UTF-8)
- [TypeError: got multiple values for argument](https://stackoverflow.com/questions/21764770/typeerror-got-multiple-values-for-argument)
- [Python Exception <TypeError>: bufsize must be an integer](https://community.safe.com/s/question/0D54Q00009jkcMHSAY/python-exception-typeerror-bufsize-must-be-an-integer)
### venv location
- [Python Virtual Environments: A Primer](https://realpython.com/python-virtual-environments-a-primer/#:~:text=You%20can%20find%20the%20base,is%20one%20folder%20level%20up.)
- [How Does a Virtual Environment Work?](https://realpython.com/python-virtual-environments-a-primer/#how-does-a-virtual-environment-work)
- [The VIRTUAL_ENV environment variable is only available if the virtual environment is activated.
For instance:
$ python3 -m venv myapp
$ source myapp/bin/activate
(myapp) $ python  -c "import os; print(os.environ['VIRTUAL_ENV'])"
/path/to/virtualenv/myapp](https://stackoverflow.com/questions/22003769/get-virtualenvs-bin-folder-path-from-script)
- [What is the difference between executing a Bash script vs sourcing it?](https://superuser.com/questions/176783/what-is-the-difference-between-executing-a-bash-script-vs-sourcing-it)
- [How to activate python virtualenv through shell script?](https://superuser.com/questions/1547228/how-to-activate-python-virtualenv-through-shell-script)
- search string: [run python3 -m venv from shell script](https://www.google.com/search?q=run+python3+-m+venv+from+shell+script&oq=run+python3+-m+venv+from+shell+script&gs_lcrp=EgZjaHJvbWUyBggAEEUYOdIBCTUwOTU2ajBqN6gCALACAA&sourceid=chrome&ie=UTF-8)
- search string: [Error: [Errno 13] Permission denied: 'C:\\Users\](https://www.google.com/search?q=Error%3A+%5BErrno+13%5D+Permission+denied%3A+%27C%3A%5C%5CUsers%5C&newwindow=1&sxsrf=AJOqlzWirByzwKUkHddqGQt9p_WYB-8a3Q%3A1676448744989&ei=6JPsY66GPOGx8gK9_6_YCQ&ved=0ahUKEwiumrefipf9AhXhmFwKHb3_C5sQ4dUDCA8&uact=5&oq=Error%3A+%5BErrno+13%5D+Permission+denied%3A+%27C%3A%5C%5CUsers%5C&gs_lcp=Cgxnd3Mtd2l6LXNlcnAQAzoKCAAQRxDWBBCwA0oECEEYAEoECEYYAFDsBVjsBWCTCWgBcAF4AIABN4gBN5IBATGYAQCgAQHIAQjAAQE&sclient=gws-wiz-serp)
### shell
- search string: [chmod executable shell script](https://www.google.com/search?q=chmod+executable+shell+script&oq=chmod+executable+shell+script&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIKCAEQABgPGBYYHjILCAIQABgWGB4Y8QQyCggDEAAYDxgWGB4yCwgEEAAYFhgeGPEEMgcIBRAAGIYDMgcIBhAAGIYDMgcIBxAAGIYDMgcICBAAGIYD0gEINjAxMGowajeoAgCwAgA&sourceid=chrome&ie=UTF-8)
- chmod +x <fileName> [Steps to write and execute a script](https://www.javatpoint.com/steps-to-write-and-execute-a-shell-script)
- search string: [how to start shell script](https://www.google.com/search?q=how+to+start+shell+script&oq=how+tostart+shell&gs_lcrp=EgZjaHJvbWUqCQgBEAAYDRiABDIGCAAQRRg5MgkIARAAGA0YgAQyCQgCEAAYDRiABDIJCAMQABgNGIAEMgkIBBAAGA0YgAQyCQgFEAAYDRiABDIJCAYQABgNGIAEMgkIBxAAGA0YgAQyCQgIEAAYDRiABDIJCAkQABgNGIAE0gEJNTUxMjdqMWo3qAIAsAIA&sourceid=chrome&ie=UTF-8)
- search string: [run shell using source](https://www.google.com/search?q=run+shell+using+source&oq=run+shell+using+source&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIJCAEQABgeGKIEMgkIAhAAGB4YogQyBwgDEAAYogQyBwgEEAAYogTSAQkxMTM0MGowajeoAgCwAgA&sourceid=chrome&ie=UTF-8)
- The first line in Bash scripts is a character sequence known as the "shebang." The shebang is the program loader's first instruction when executing the file, and the characters indicate which interpreter to run when reading the script. | Add the following line to the file to indicate the use of the Bash interpreter: | #!/bin/bash [How to Write a Bash Script with Examples | Writing a Bash Script | Adding the "shebang" | #!/usr/bin/env <interpreter> | Uses the env program to locate the interpreter. Use this shebang for other scripting languages, such as Perl, Python, etc.](https://phoenixnap.com/kb/write-bash-script#:~:text=The%20first%20line%20in%20Bash,run%20when%20reading%20the%20script.)
- search string: [what does comment do at top of shell script](https://www.google.com/search?q=what+does+comment+do+at+top+of+shell+script&oq=what+does+comment+do+at+top+of+shell+script&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIHCAEQIRigATIHCAIQIRigAdIBCTEwMDc2ajBqN6gCALACAA&sourceid=chrome&ie=UTF-8)
- [How to activate a Python virtual environment from a script file](https://www.a2hosting.com/kb/developer-corner/python/activating-a-python-virtual-environment-from-a-script-file)
- search string: [python file to start venv](https://www.google.com/search?q=python+file+to+start+venv&oq=python+file+to+start+venv&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIGCAEQRRhA0gEIODQ2M2owajeoAgCwAgA&sourceid=chrome&ie=UTF-8)
- search string: [pass raw strings from shell or py file to terminal to run command in terminal](https://www.google.com/search?q=pass+raw+strings+from+shell+or+py+file+to+terminal+to+run+command+in+terminal&oq=pass+raw+strings+from+shell+or+py+file+to+terminal+to+run+command+in+terminal&gs_lcrp=EgZjaHJvbWUyBggAEEUYOdIBCTIxNjc3ajBqN6gCALACAA&sourceid=chrome&ie=UTF-8)
- search string: [Taking Linux Command as Raw String in Python](https://stackoverflow.com/questions/22230294/taking-linux-command-as-raw-string-in-python)
- [Taking Linux Command as Raw String in Python](https://stackoverflow.com/questions/22230294/taking-linux-command-as-raw-string-in-python/22230442#comment33755480_22230442)
- search string: [how to pass raw code to terminal](https://www.google.com/search?q=how+to+pass+raw+codee+to+terminal&oq=how+to+pass+raw+codee+to+terminal&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIGCAEQRRhA0gEINjY1N2owajeoAgCwAgA&sourceid=chrome&ie=UTF-8)
- [What are some ways to pass raw bytes to a program via the Linux terminal?](https://reverseengineering.stackexchange.com/questions/24755/what-are-some-ways-to-pass-raw-bytes-to-a-program-via-the-linux-terminal)
- [Pass bash argument to python script](https://stackoverflow.com/questions/14340822/pass-bash-argument-to-python-script)
- search string: [pass arg to function python through bash call](https://www.google.com/search?q=pass+arg+to+function+python+through+bash+call&oq=pass+arg+to+function+python+through+bash+call&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIHCAEQABiiBDIHCAIQABiiBDIHCAMQABiiBDIHCAQQABiiBDIHCAUQABiiBNIBCTE1NjgwajBqN6gCALACAA&sourceid=chrome&ie=UTF-8)
- [venv — Creation of virtual environments | An example of extending EnvBuilder](https://docs.python.org/3/library/venv.html)
- search string: [try catch shell python venv](https://www.google.com/search?q=try+catch+shell+python+venv&oq=try+catch+shell+python+venv&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIHCAEQIRigATIHCAIQIRigAdIBCDU3NzhqMGo3qAIAsAIA&sourceid=chrome&ie=UTF-8)
- [PermissionError: [Errno 13] Permission denied](https://stackoverflow.com/questions/36434764/permissionerror-errno-13-permission-denied)
- [How to assign the output of a Bash command to a variable? [duplicate]](https://stackoverflow.com/questions/2314750/how-to-assign-the-output-of-a-bash-command-to-a-variable)
- search string: [#!/bin/bash -x PWD=`pwd`](https://www.google.com/search?q=%23!%2Fbin%2Fbash+-x+PWD%3D%60pwd%60&oq=%23!%2Fbin%2Fbash+-x+PWD%3D%60pwd%60&gs_lcrp=EgZjaHJvbWUyBggAEEUYOdIBBzU0MmowajeoAgCwAgA&sourceid=chrome&ie=UTF-8)
- search string: [how to activate venv in existing shell](https://www.google.com/search?q=how+to+activate+venv+in+existing+shell&oq=how+to+activate+venv+in+existing+shell&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIGCAEQRRhA0gEJMTE0ODBqMGo3qAIAsAIA&sourceid=chrome&ie=UTF-8)
- search string: [use shell to activate venv](https://www.google.com/search?q=use+shell+to+activate+venv&oq=use+shell+to+activate+venv&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIHCAEQIRigATIHCAIQIRigATIHCAMQIRigATINCAQQIRgWGB0YHhjxBDINCAUQIRgWGB0YHhjxBNIBCTEwMTMyajBqN6gCALACAA&sourceid=chrome&ie=UTF-8)
- [Learn X in Y minutes](https://learnxinyminutes.com/docs/bash/)
- [Writing shell scripts](https://infinum.com/handbook/qa/automation/general/writing-shell-scripts)
- search string: [automate virtual env](https://www.google.com/search?q=automate+virtual+env&oq=automate+virtual+env&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTILCAEQABgWGB4Y8QQyCwgCEAAYFhgeGPEEMgsIAxAAGBYYHhjxBDILCAQQABgWGB4Y8QQyCAgFEAAYFhgeMgsIBhAAGBYYHhjxBDIICAcQABgWGB4yCggIEAAYDxgWGB4yCggJEAAYDxgWGB7SAQg5ODk1ajBqN6gCALACAA&sourceid=chrome&ie=UTF-8)