# PROCESS OVERVIEW OF startenv.py
1. imports
2. venv startup
3. modules list to be installed
4. (part z a) modification post check
5. declare not_installed list
6. go through for loop
7. Install packages from not_installed list
7.1 need to install nb-black
7.2 (#3) concatenate prepend as prefix -U to scikit-learn
