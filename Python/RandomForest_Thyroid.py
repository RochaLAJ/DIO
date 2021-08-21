import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


#Random Data
df = pd.DataFrame(np.random.randint(.01,15,size=(1000, 2)), columns=['TSH','FT4'])
df_2 = pd.DataFrame(np.random.randint(0.0,1.0+1,size=(1000, 4)), columns=['Sex','Depression','Addiction','Medication'])
merged = [df, df_2]
result = pd.concat(merged, axis=1)


#Split rule 70-30
d1_train, d1_test, d2_train, d2_test = train_test_split(result.drop(['Depression'], axis=1), 
                                                        result["Depression"], 
                                                        test_size=0.3, 
                                                        random_state=12345)
print([{'train': d1_train.shape},{'test': d1_test.shape}])

#Classification
forest = RandomForestClassifier(n_estimators=1000,
                                criterion='gini',
                                max_depth=5,
                                verbose=1)
forest.fit(d1_train, d2_train)

#Prediction
prob = forest.predict_proba(result.drop('Depression', axis=1))[:,1]
classification = forest.predict(result.drop('Depression',axis=1))

#Add new columns
result['prob'] = prob
result['class'] = classification

#Print proportion of class
print(result['class'].value_counts())

#Variation of prob
fig, ax = plt.subplots()
plt.boxplot(result['prob'])
plt.show()
