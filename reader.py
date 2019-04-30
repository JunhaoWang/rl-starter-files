import pickle
import pandas as pd

file = open('demonstratorSSrep_PPOwKL.pkl', 'rb')
demonstratorSSRep = pickle.load(file)
file = open('stateToIndex.pkl', 'rb')
stateToIndex = pickle.load(file)
file = open('indexToState.pkl', 'rb')
indexToState = pickle.load(file)

df = pd.DataFrame.from_dict(indexToState)

print(df)
df.to_csv("indexToState.csv",index=False)
print(demonstratorSSRep)

my_df = pd.DataFrame(demonstratorSSRep)

my_df.to_csv("demonstratorSSrep_PPOwKL.csv",index=False)

