Income Group Classification
Install
Import
	- ConfusionMatrixDisplay
Who=Read
Data=Copy
Head() tail() shape _ _ info() duplicated().sum() .isnull().sum()
.describe().T
_____________________
histogram_boxplot
labeled_barplot
stacked_barplot
distribution_plot_wrt_target
df=data.copy()
Univariate Analysis
//// Observations on workclass
//// Observations on native_country
//// Observations on salary
Bivariate analysis
//// Correlation check
//// salary vs sex
//// salary vs education
//// salary vs occupation
//// salary vs workclass
//// salary vs age
//// salary vs working_hours_per_week
((untitled summary plot for salary against several parameters/factors
))
// Data Preprocessing
** Dropping capital_gain and capital_loss
/// Outlier Detection
/// Outlier Treatment
treat_outliers
treat_outliers_all
numerical_col=data.select_dtypes()
data=treat_outliers_all()
plt.figure(figsize=(20, 30))
for i, variable in enumerate(numerical_col):
	plt ((parameters))
plt.show()
/// Data Preparation for Modeling
** Encoding >50K as 0 and <=50K as 1 to find underprivileged
data["salary"]=data["salary"].apply(lambda x: 1 if x == " <=50K" else 0)
Creating training and test sets.
X = data.drop(["salary"], axis=1)
Y=data["salary"]
X=sm.add_constant(X)
Q/ What is sm? statsmodels.api
A/ # To build model for prediction
	import statsmodels.stats.api as sms
	from statsmodels.stats.outliers_influence import variance_inflation_factor
		import statsmodels.api as sm
	from statsmodels.tools.tools import add_constant
	from sklearn.linear_model import LogisticRegression
-----------
X=pd.get_dummies(X,drop_first=True)
X_train,X_tesst,y_train,y_tesst=train_test_split(
	X,Y,test_size=0.30)
prints -> Training set and test set both have 53 cols

Model Building - Logistic Regression
