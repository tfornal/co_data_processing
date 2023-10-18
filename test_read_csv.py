import pandas as pd
import numpy as np
df = pd.read_csv("test.csv", sep = "\t")

#time_in_ns_list = []
#for idx, times in enumerate(df.iterrows()):
#    ilosc_wierszy = 5
    



# Creating a dictionary
d = {
    'a':[2,4,6,8,10],
    'b':[1,3,5,7,9],
    'c':[2,4,6,8,10],
    'd':[1,3,5,7,9]
}

# Creating DataFrame
df = pd.DataFrame(d)

# Display original DataFrame
print("Original DataFrame:\n",df,"\n")

# Calculating mean
x = np.arange(len(df))
print(x)
res = df.groupby(np.arange(len(df))//5)
breakpoint()
res = df.groupby(np.arange(len(df))//5).mean()

# Display result
print("Result:\n",res)