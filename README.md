# machine-learning-classification

# primary source for this README: jupyter-6-Supervised-Learning
Repository for running jupyter notebooks and keeping relevant files in one place


Updates from 

# ML-logistic-regression-notes
- [CoderSales/ML-logistic-regression-notes](https://github.com/CoderSales/ML-logistic-regression-notes/blob/main/README.md)
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
### Gitpod
- [Gitpod docs prebuilds](https://www.gitpod.io/docs/configure/projects/prebuilds)
- [Gitpod docs workspaces](https://www.gitpod.io/docs/configure/workspaces/tasks)
- [Gitpod Prebuild](https://youtu.be/ZtlJ0PakUHQ?t=54)
### Git in VSCode
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
### Pandas
### matplotlib
### scipy
### scipy.stats
### statsmodels
- [statsmodels.stats.proportion.proportions_ztest](https://www.statsmodels.org/dev/generated/statsmodels.stats.proportion.proportions_ztest.html)
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
## Statistics

## pandas print statement
- [turn off automatic pandas data type output on print statment](https://stackoverflow.com/questions/29645153/remove-name-dtype-from-pandas-output-of-dataframe-or-series)

## python
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

### print separate with no spaces
- [Print without space in python 3](https://stackoverflow.com/questions/12700558/print-without-space-in-python-3)
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
## HTML

## CSS
- not used [box-shadow: red](https://stackoverflow.com/questions/61476773/how-to-add-a-background-square-behind-the-image)
- used [change body tag background color behind image](https://stackoverflow.com/questions/7415872/change-color-of-png-image-via-css)

## Images
## IMG
- not used [to crop images in css](https://www.digitalocean.com/community/tutorials/css-cropping-images-object-fit)
## SVG
- [harmonic mean .svg file](https://upload.wikimedia.org/wikipedia/commons/f/f7/MathematicalMeans.svg)
- [harmonic mean .svg file page 2](https://en.wikipedia.org/wiki/File:MathematicalMeans.svg)
- [means visual proof](https://en.wikipedia.org/wiki/Harmonic_mean#/media/File:QM_AM_GM_HM_inequality_visual_proof.svg)
- [How to edit color via code of svg file with: open svg file in explorer > inspect element > Elements > edit circle tag fill attribute](https://medium.com/@nick.cqx/illustrate-your-project-without-any-graphic-design-software-using-svg-and-your-browser-20e9a73b53a3)