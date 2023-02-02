python3 -m venv .venv
source .venv/Scripts/activate
python3 -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install --upgrade pip
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
