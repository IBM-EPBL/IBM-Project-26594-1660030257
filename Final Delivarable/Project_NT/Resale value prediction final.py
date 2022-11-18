import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.preprocessing import LabelEncoder
import pickle


#Load the dataset
df = pd.read_csv(r"E:\car_resale\Data\autos.csv", header=0, sep=',', encoding='Latin1', )

#print all the different sellers
print(df.seller.value_counts()) 

#remove the seller type haveing only 3 car
df[df.seller != 'gewerblich'] 

#now all the sellers are same so we can get rid of this column
df=df.drop(columns=['seller'])  #1 refer the columns & 0 refer the index 

#print all different seller
print(df.offerType.value_counts())

#remove the offers type having only 12 listings
df[df.offerType != 'Gesuch']

#now all offer are sameso we can get rid this collumn
df=df.drop(columns=['offerType']) # 1 refer the columns & 0 refer the index

'''car having power les then 50ps and above 900ps seems a little suspicious, let's remove
them and see what we have got now'''
print(df.shape)
df = df[(df.powerPS > 50) & (df.powerPS < 900)]
print(df.shape)
#around 50000 cars ahave been removed which could have introunduced error toour data



#Simlarly, filtering our the cars having registeration years not in the mentioned range
#print(df.shape)
df = df[(df.yearOfRegistration >= 1950) & (df.yearOfRegistration < 2017)]
print(df.shape)
#not much of a difference but still, 10000 rows have been reduced. it's better to
#get rid of faulty data instead of keeping them just to increase the size

'''removing irrelevent columns which are either the same for all the cars in the dataset, or can
introduce bias, so removing them too..'''

df.drop(['name', 'abtest', 'dateCrawled', 'nrOfPictures', 'lastSeen', 
        'postalCode', 'dateCreated'], axis='columns', inplace=True)

'''dropping the duplicates from the dataframe and stroing it in a new
here all row having same value in all the mentioned columns will be deleted and by defult,
only first occurance of any such row is kept'''
new_df = df.copy()
new_df = new_df.drop_duplicates(['price', 'vehicleType', 'yearOfRegistration', 'gearbox', 'powerPS', 
                                 'model', 'kilometer', 'monthOfRegistration', 'fuelType', 'notRepairedDamage'])

#As the dataset contained same german words for many features, changing them to engilsh
new_df.gearbox.replace(('manuell','automatik'), ('manual','automatic'), inplace=True)
new_df.fuelType.replace(('benzin','andere','elektro'), ('petrol','others','electric'), inplace=True)
new_df.vehicleType.replace(('kleinwagen','cabrio','kombi','andere'),
                            ('small car','canvertible','combination','others'), inplace=True)
new_df.notRepairedDamage.replace(('ja','nein'), ('Yes','No'), inplace=True)                            

#### Removing the outliers
new_df = new_df[(new_df.price >= 100) & (new_df.price <= 150000)]

''' Filling NaN values for columns whose data might not be there with the information provider,
which might lead to some variance but our model but we will still be able to give some estimate to the user'''

new_df['notRepairedDamage'].fillna(value='not-declared',inplace=True)
new_df['fuelType'].fillna(value='not-declared',inplace=True)
new_df['gearbox'].fillna(value='not-declared',inplace=True)
new_df['vehicleType'].fillna(value='not-declared',inplace=True)
new_df['model'].fillna(value='not-declared',inplace=True)

#can save the csv for future purpose.
new_df.to_csv("autos_preprocessed.csv")


#Columns which contain categorical values, which we'll need to convert via label encoding
labels = ['gearbox', 'notRepairedDamage', 'model', 'brand', 'fuelType', 'vehicleType']

'''looping over the labels to the label encoding for all at once and 
saveing the LABEL ENCODING FILES'''
mapper = {}
for i in labels:
    mapper[i] = LabelEncoder()
    mapper[i].fit(new_df[i])
    tr = mapper[i].transform(new_df[i])
    np.save(str('classes'+i+'.npy'), mapper[i].classes_)
    print(i,":", mapper[i])
    new_df.loc[:, i+ '_labels'] = pd.Series(tr, index=new_df.index)
    
#Final data to be put in a new dataframe called "LABELED",
labeled = new_df[
    [
        'price',
        'yearOfRegistration',
        'powerPS',
        'kilometer',
        'monthOfRegistration'
    ] + [x+"_labels" for x in labels]
]

print(labeled.columns)    


#Storing price in Y and reset of the data in X
Y = labeled.iloc[:,0].values
X = labeled.iloc[:,1:].values

#need to reshape the Y values
Y = Y.reshape(-1,1)


#traing data and test data
from sklearn.model_selection import cross_val_score, train_test_split 
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.3, random_state=3)


#Model building and fitting
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
regressor = RandomForestRegressor(n_estimators = 1000, max_depth = 10, random_state = 34)

#fitting the model
regressor.fit(X_train , np.ravel(Y_train, order = 'C'))

#predicting the values of test
y_pred = regressor.predict(X_test)

#printing the Accuraccy for test set
print(r2_score(Y_test, y_pred))


#saving the model for future use.
filename = 'resale_model.pkl'
pickle.dump(regressor, open(filename, 'wb'))

