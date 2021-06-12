import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
pickle_in = open("pcalogistic.pkl","rb")
model=pickle.load(pickle_in)
dataset= pd.read_csv('Classification Dataset1.csv')

X = dataset.iloc[:,1:10].values
dataframe=pd.DataFrame(X,columns=['CreditScore','Geography','Gender','Age','Tenure','Balance','HasCrCard','IsActiveMember','EstimatedSalary'])


# Taking care of missing data
#handling missing data (Replacing missing data with the mean value)  
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.NaN, strategy= 'mean', fill_value=None, verbose=1, copy=True)
#Fitting imputer object to the independent variables x.   
imputer = imputer.fit(X[:,3:9]) 
#Replacing missing data with the calculated mean value  
X[:,3:9]= imputer.transform(X[:,3:9])  

# Taking care of missing data
#handling missing data (Replacing missing data with the mean value)  
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.NaN, strategy= 'constant', fill_value='female', verbose=1, copy=True)
#Fitting imputer object to the independent variables x.   
imputer = imputer.fit(X[:, 2:3]) 
#Replacing missing data with the constant value  
X[:, 2:3]= imputer.transform(X[:,2:3])  

# Encoding Categorical data:
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])

# Encoding Categorical data:
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit_transform(X)


def predict_note_authentication(CreditScore1,Geography1,Gender1,Age1,Tenure1,Balance1,HasCrCard1,IsActiveMember1,EstimatedSalary1,
                                         CreditScore2,Geography2,Gender2,Age2,Tenure2,Balance2,HasCrCard2,IsActiveMember2,EstimatedSalary2,
                                CreditScore3,Geography3,Gender3,Age3,Tenure3,Balance3,HasCrCard3,IsActiveMember3,EstimatedSalary3):
  X1=sc.fit_transform([[CreditScore1,Geography1,Gender1,Age1,Tenure1,Balance1,HasCrCard1,IsActiveMember1,EstimatedSalary1,]
                                        ,[CreditScore2,Geography2,Gender2,Age2,Tenure2,Balance2,HasCrCard2,IsActiveMember2,EstimatedSalary2],
                       [CreditScore3,Geography3,Gender3,Age3,Tenure3,Balance3,HasCrCard3,IsActiveMember3,EstimatedSalary3]])
  # Applying PCA
  from sklearn.decomposition import PCA
  pca = PCA(n_components = 3)
  X1 = pca.fit_transform(X1)

  output = model.predict(X1)
  res=[]
  for i in output:
    if i==[0]:
      res.append("Customer will not Leave")
    else:
      res.append("Customer will Leave")
  #print(prediction)
  return res
def main():
    
    html_temp = """
   <div class="" style="background-color:blue;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Department of Computer Engineering</p></center> 
   <center><p style="font-size:25px;color:white;margin-top:10px;">Machine Learning Experiment-11</p></center> 
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Customer Prediction")
    Age1 = st.number_input('Insert a Age',18,60)
    CreditScore1= st.number_input('Insert a CreditScore',400,1000)
    HasCrCard1 = st.number_input('Insert a HasCrCard 0 For No 1 For Yes',0,1)
    Tenure1 = st.number_input('Insert a Tenure',0,20)
    Balance1 = st.number_input('Insert a Balance',0)
    Gender1 = st.number_input('Insert 0 For Male 1 For Female ',0,1)
    Geography1= st.number_input('Insert Geography 0 For France 1 For Spain',0,1)
    IsActiveMember1= st.number_input('Insert a IsActiveMember 0 For No 1 For Yes',0,1)
    EstimatedSalary1= st.number_input('Insert a EstimatedSalary',0)

    Age2 = st.number_input('Insert a Age2',18,60)
    CreditScore2= st.number_input('Insert a CreditScore2',400,1000)
    HasCrCard2 = st.number_input('Insert a HasCrCard2 0 For No 1 For Yes',0,1)
    Tenure2 = st.number_input('Insert a Tenure2',0,20)
    Balance2 = st.number_input('Insert a Balance2',0)
    Gender2 = st.number_input('Insert 0 For Male2 1 For Female ',0,1)
    Geography2= st.number_input('Insert Geography2 0 For France 1 For Spain',0,1)
    IsActiveMember2= st.number_input('Insert a IsActiveMember2 0 For No 1 For Yes',0,1)
    EstimatedSalary2= st.number_input('Insert a EstimatedSalary2',0)

    Age3 = st.number_input('Insert a Age3',18,60)
    CreditScore3= st.number_input('Insert a CreditScore3',400,1000)
    HasCrCard3 = st.number_input('Insert a HasCrCard3 0 For No 1 For Yes',0,1)
    Tenure3 = st.number_input('Insert a Tenure3',0,20)
    Balance3 = st.number_input('Insert a Balance3',0)
    Gender3 = st.number_input('Insert 0 For Male3 1 For Female ',0,1)
    Geography3= st.number_input('Insert Geography3 0 For France 1 For Spain',0,1)
    IsActiveMember3= st.number_input('Insert a IsActiveMember3 0 For No 1 For Yes',0,1)
    EstimatedSalary3= st.number_input('Insert a EstimatedSalary3',0)
    
    
    resul=""
    if st.button("Predict"):
      result=predict_note_authentication(CreditScore1,Geography1,Gender1,Age1,Tenure1,Balance1,HasCrCard1,IsActiveMember1,EstimatedSalary1,
                                         CreditScore2,Geography2,Gender2,Age2,Tenure2,Balance2,HasCrCard2,IsActiveMember2,EstimatedSalary2,
                                         CreditScore3,Geography3,Gender3,Age3,Tenure3,Balance3,HasCrCard3,IsActiveMember3,EstimatedSalary3)
      st.success('Model has predicted {}'.format(result))
    if st.button("About"):
      st.subheader("Developed by Rajendra")
      st.subheader("C-Section,PIET")

if __name__=='__main__':
  main()
   
