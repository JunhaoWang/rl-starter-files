import pickle
import pandas as pd

file = open('demonstratorSSrep_drugadd.pkl', 'rb')
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

my_df.to_csv("drugSSRep.csv",index=False)

file = open('agentSSrep_PPOwKL1.pkl', 'rb')
demonstratorSSRep = pickle.load(file)

my_df = pd.DataFrame(demonstratorSSRep)

my_df.to_csv("agentSSrep_PPOwKL1.csv",index=False)

file = open('agentSSrep_PPOexpertOnlyNoKL.pkl', 'rb')
demonstratorSSRep = pickle.load(file)

my_df = pd.DataFrame(demonstratorSSRep)

my_df.to_csv("agentSSrep_PPOexpertOnlyNoKL.csv",index=False)

