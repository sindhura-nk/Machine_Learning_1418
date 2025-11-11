# To clean the data and preprocess it
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer

def clean_process_data(cat,con,X):
    # Creating numerical pipeline that handles missing data and also scales numerical data
    num_pipe = make_pipeline(SimpleImputer(strategy='mean'),StandardScaler())
    # Creating categorical pipeline that handles missing data and scales text data
    cat_pipe = make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder(handle_unknown='ignore',sparse_output=False))
    # combining both the pipelines
    pre = ColumnTransformer([
        ('cat',cat_pipe,cat),
        ('con',num_pipe,con)
    ]).set_output(transform='pandas')

    # use the pre to fit and transform X data
    X_pre = pre.fit_transform(X)

    # return the preprocessed X 
    return pre,X_pre

# WRITE A CODE THAT HANDLES PREPROCESSING AND CLEANING OF NUMERICAL DATA ALONE.
def clean_process_CON_data(X):
    # Creating numerical pipeline that handles missing data and also scales numerical data
    num_pipe = make_pipeline(SimpleImputer(strategy='mean'),StandardScaler()).set_output(transform='pandas')

    # use the pre to fit and transform X data
    X_pre = num_pipe.fit_transform(X)

    # return the preprocessed X 
    return num_pipe,X_pre

# WRITE A CODE THAT HANDLES PREPROCESSING AND CLEANING OF categorical DATA ALONE.
def clean_process_CAT_data(X):
    # Creating numerical pipeline that handles missing data and also scales numerical data
    cat_pipe = make_pipeline(SimpleImputer(strategy='most_frequent'),OneHotEncoder(handle_unknown='ignore',sparse_output=False)).set_output(transform='pandas')

    # use the pre to fit and transform X data
    X_pre = cat_pipe.fit_transform(X)

    # return the preprocessed X 
    return cat_pipe,X_pre