#!/usr/bin/env python
# coding: utf-8

# ### Task 1.1

# In[36]:


import numpy as np

a = np.array([[2.3, 5.1, 4.7],
            [3.5, 6.7, 1.5],
            [8.4, 3.1, 9.2]]) 

b = np.array([[4.3, 8.1, 6.1],
            [3.7, 6.2, 1.5],
            [2.4, 5.7, 4.7]]) 

print (a.shape)
print(a + b)
print(b - a)
print(a*b)
print(a/b)

print(a**2)
print(b%2)

print (np.cos(a))
print (np.sin(b))
print(np.sin(b) + np.cos(a))


# ### Task 1.2

# In[37]:


import numpy as np

a = np.array([1.2, 2.4, 3.6, 3.5, 6.7, 1.5, 8.4, 3.1, 9.2]) 

rows = a.shape
print(rows)

num = a.reshape((9,1))
print(num)

num_1 = a.reshape((3,3))
print(num_1)

max_1 = num_1.max(axis = 0)
print(max_1)
max_2 = num_1.max(axis = 1)
print(max_2)

min_1 = num_1.min(axis = 0)
print(min_1)
min_2 = num_1.min(axis = 1)
print(min_2)

sum_1 = num_1.sum(axis = 0)
print(sum_1)
sum_2 = num_1.sum(axis = 1)
print(sum_2)


# ### Task 1.3

# In[38]:


import numpy as np

a = np.array([[4, 2], [9, 1]])
b = np.array([[5, 3], [2, 5]]) 

c = np.vstack((a, b))
print(c)

num = c[:-1,0]
print(num)

max = num.max()
print(max)
min = num.min()
print(min)
sum = num.sum()
print(sum)

d = np.hstack((a, b))
print(d)

num_1 = d[0,:-1]
print(num_1)

max_1 = num_1.max()
print(max_1)
min_1 = num_1.min()
print(min_1)
sum_1 = num_1.sum()
print(sum_1)


# ### Task 1.4

# In[39]:


import numpy as np

a = np.array([[5, 4], [2, -6]])
b = np.array([[14], [-2]]) 

c = np.linalg.solve(a, b)
print(c)


# ### Task 1.5

# In[40]:


import numpy as np

a = np.array([[2, 8], [1, -6]])
b = np.array([[3, 2 ,7], [4, 1, 8], [6, 3, 7]])
c = np.array([[4, 3, 2, 7], [6, 1, 1, -2], [7, 5, 8, 1], [9, 5, -3, -5]])


print(np.transpose(a))
print(np.transpose(b))
print(np.transpose(c))

print(np.linalg.inv(a))
print(np.linalg.inv(b))
print(np.linalg.inv(c))

print(np.linalg.det(a))
print(np.linalg.det(b))
print(np.linalg.det(c))

vector_1 = np.linalg.norm(b, 3, axis = 1) 
print(vector_1)

vector_2 = np.linalg.norm(c, 3, axis = 0) 
print(vector_2)


# ### Task 1.6

# In[41]:


import numpy as np
import random

a = np.array([random.randint(0, 100) for i in range(0, 100, 1)]) 

cnt = 0 
for a in a:
    if a > 50: 
        cnt += 1
print(cnt)

a = np.array([random.randint(0, 100) for i in range(0, 100, 1)]) 
print ((a > 50).sum())


# ### Task 2.1

# In[42]:


import pandas as pd 

heights = [188, 172, 187, 161, 183, 172, 185, 163, 173, 183, 174, 174, 175,
178, 183, 195, 178, 173, 174, 183, 174, 181, 162, 180, 170, 175, 182, 180, 183,
178, 182, 188, 175, 179, 184, 193, 182, 183, 175, 185, 182, 183, 156, 185, 199]

weights = [86, 71, 89, 64, 83, 71, 86, 68, 73, 84, 73, 72, 75, 78, 85, 93, 78,
70, 74, 80, 83, 82, 68, 80, 73, 78, 82, 81, 83, 79, 82, 86, 76, 77, 80, 103,
82, 83, 77, 89, 80, 85, 82, 85, 110]

ages = [58, 61, 54, 54, 58, 57, 63, 54, 37, 51, 34, 26, 50, 48, 45, 52, 56,
46, 54, 28, 52, 47, 55, 55, 54, 42, 51, 56, 55, 51, 54, 32, 21, 62, 43, 55,
56, 61, 53, 22, 64, 45, 54, 47, 38] 
 
df = pd.DataFrame ({"heights" : heights, "weights" : weights, "ages" : ages})

print(df)

print (df.loc[0:2])

print (df.loc[42:])


# ### Task 2.2

# In[45]:


import pandas as pd 

heights = [188, 172, 187, 161, 183, 172, 185, 163, 173, 183, 174, 174, 175,
178, 183, 195, 178, 173, 174, 183, 174, 181, 162, 180, 170, 175, 182, 180, 183,
178, 182, 188, 175, 179, 184, 193, 182, 183, 175, 185, 182, 183, 156, 185, 199]

weights = [86, 71, 89, 64, 83, 71, 86, 68, 73, 84, 73, 72, 75, 78, 85, 93, 78,
70, 74, 80, 83, 82, 68, 80, 73, 78, 82, 81, 83, 79, 82, 86, 76, 77, 80, 103,
82, 83, 77, 89, 80, 85, 82, 85, 110]

ages = [58, 61, 54, 54, 58, 57, 63, 54, 37, 51, 34, 26, 50, 48, 45, 52, 56,
46, 54, 28, 52, 47, 55, 55, 54, 42, 51, 56, 55, 51, 54, 32, 21, 62, 43, 55,
56, 61, 53, 22, 64, 45, 54, 47, 38] 
 
df = pd.DataFrame ({"heights" : heights, "weights" : weights, "ages" : ages})
print (df)

max = df.max()
print(max)

min = df.min()
print(min)

mid = df.mean()
print(mid)

sum_1 = df['heights'].sum()
mid_1 = sum_1/len(heights)
print ('heights', mid_1)

sum_2 = df['weights'].sum()
mid_2 = sum_2/len(weights)
print ('weights', mid_2)

sum_3 = df['ages'].sum()
mid_3 = sum_3/len(ages)
print ('ages', mid_3)


# ### Task 2.3

# In[ ]:


import pandas as pd 

heights = [188, 172, 187, 161, 183, 172, 185, 163, 173, 183, 174, 174, 175,
178, 183, 195, 178, 173, 174, 183, 174, 181, 162, 180, 170, 175, 182, 180, 183,
178, 182, 188, 175, 179, 184, 193, 182, 183, 175, 185, 182, 183, 156, 185, 199]

weights = [86, 71, 89, 64, 83, 71, 86, 68, 73, 84, 73, 72, 75, 78, 85, 93, 78,
70, 74, 80, 83, 82, 68, 80, 73, 78, 82, 81, 83, 79, 82, 86, 76, 77, 80, 103,
82, 83, 77, 89, 80, 85, 82, 85, 110]

ages = [58, 61, 54, 54, 58, 57, 63, 54, 37, 51, 34, 26, 50, 48, 45, 52, 56,
46, 54, 28, 52, 47, 55, 55, 54, 42, 51, 56, 55, 51, 54, 32, 21, 62, 43, 55,
56, 61, 53, 22, 64, 45, 54, 47, 38] 
 
df = pd.DataFrame ({"heights" : heights, "weights" : weights, "ages" : ages})
print (df)

print (df.loc[0:2])
print (df.loc[42:])

print(df['heights'].describe()) 
print(df['weights'].describe()) 
print(df['ages'].describe()) 


# ### Task 3.1

# In[ ]:


import pandas as pd 

heights = [188, 172, 187, 161, 183, 172, 185, 163, 173, 183, 174, 174, 175,
178, 183, 195, 178, 173, 174, 183, 174, 181, 162, 180, 170, 175, 182, 180, 183,
178, 182, 188, 175, 179, 184, 193, 182, 183, 175, 185, 182, 183, 156, 185, 199]

weights = [86, 71, 89, 64, 83, 71, 86, 68, 73, 84, 73, 72, 75, 78, 85, 93, 78,
70, 74, 80, 83, 82, 68, 80, 73, 78, 82, 81, 83, 79, 82, 86, 76, 77, 80, 103,
82, 83, 77, 89, 80, 85, 82, 85, 110]

ages = [58, 61, 54, 54, 58, 57, 63, 54, 37, 51, 34, 26, 50, 48, 45, 52, 56,
46, 54, 28, 52, 47, 55, 55, 54, 42, 51, 56, 55, 51, 54, 32, 21, 62, 43, 55,
56, 61, 53, 22, 64, 45, 54, 47, 38] 
 
df = pd.DataFrame ({"heights" : heights, "weights" : weights, "ages" : ages})


df = df.loc[0:40]

print (df)


# ### Task 3.2

# In[ ]:


import pandas as pd 

heights = [188, 172, 187, 161, 183, 172, 185, 163, 173, 183, 174, 174, 175,
178, 183, 195, 178, 173, 174, 183, 174, 181, 162, 180, 170, 175, 182, 180, 183,
178, 182, 188, 175, 179, 184, 193, 182, 183, 175, 185, 182, 183, 156, 185, 199]

weights = [86, 71, 89, 64, 83, 71, 86, 68, 73, 84, 73, 72, 75, 78, 85, 93, 78,
70, 74, 80, 83, 82, 68, 80, 73, 78, 82, 81, 83, 79, 82, 86, 76, 77, 80, 103,
82, 83, 77, 89, 80, 85, 82, 85, 110]

ages = [58, 61, 54, 54, 58, 57, 63, 54, 37, 51, 34, 26, 50, 48, 45, 52, 56,
46, 54, 28, 52, 47, 55, 55, 54, 42, 51, 56, 55, 51, 54, 32, 21, 62, 43, 55,
56, 61, 53, 22, 64, 45, 54, 47, 38] 
 
df = pd.DataFrame ({"heights" : heights, "weights" : weights, "ages" : ages})

df = df.loc[0:40]
print (df)

df_new = df[(df["heights"] > 180)]["heights"].count()
print (df_new)


# ### Task 3.3

# In[ ]:


import pandas as pd 
heights = [188, 172, 187, 161, 183, 172, 185, 163, 173, 183, 174, 174, 175,
178, 183, 195, 178, 173, 174, 183, 174, 181, 162, 180, 170, 175, 182, 180, 183,
178, 182, 188, 175, 179, 184, 193, 182, 183, 175, 185, 182, 183, 156, 185, 199]

weights = [86, 71, 89, 64, 83, 71, 86, 68, 73, 84, 73, 72, 75, 78, 85, 93, 78,
70, 74, 80, 83, 82, 68, 80, 73, 78, 82, 81, 83, 79, 82, 86, 76, 77, 80, 103,
82, 83, 77, 89, 80, 85, 82, 85, 110]

ages = [58, 61, 54, 54, 58, 57, 63, 54, 37, 51, 34, 26, 50, 48, 45, 52, 56,
46, 54, 28, 52, 47, 55, 55, 54, 42, 51, 56, 55, 51, 54, 32, 21, 62, 43, 55,
56, 61, 53, 22, 64, 45, 54, 47, 38] 
 
df = pd.DataFrame ({"heights" : heights, "weights" : weights, "ages" : ages})

df = df.loc[0:40]
print (df)

df_new = df[(df["weights"] < 80)]["weights"].count()
print (df_new)


# ### Task 3.4

# In[ ]:


import pandas as pd 

heights = [188, 172, 187, 161, 183, 172, 185, 163, 173, 183, 174, 174, 175,
178, 183, 195, 178, 173, 174, 183, 174, 181, 162, 180, 170, 175, 182, 180, 183,
178, 182, 188, 175, 179, 184, 193, 182, 183, 175, 185, 182, 183, 156, 185, 199]

weights = [86, 71, 89, 64, 83, 71, 86, 68, 73, 84, 73, 72, 75, 78, 85, 93, 78,
70, 74, 80, 83, 82, 68, 80, 73, 78, 82, 81, 83, 79, 82, 86, 76, 77, 80, 103,
82, 83, 77, 89, 80, 85, 82, 85, 110]

ages = [58, 61, 54, 54, 58, 57, 63, 54, 37, 51, 34, 26, 50, 48, 45, 52, 56,
46, 54, 28, 52, 47, 55, 55, 54, 42, 51, 56, 55, 51, 54, 32, 21, 62, 43, 55,
56, 61, 53, 22, 64, 45, 54, 47, 38] 
 
df = pd.DataFrame ({"heights" : heights, "weights" : weights, "ages" : ages})


df = df.loc[0:40]
print (df)

df_new = df[(df["ages"] < 30) & (df["ages"] < 50)]["ages"].count()
print (df_new)


# ### Task 3.5

# In[ ]:


import pandas as pd 

heights = [188, 172, 187, 161, 183, 172, 185, 163, 173, 183, 174, 174, 175,
178, 183, 195, 178, 173, 174, 183, 174, 181, 162, 180, 170, 175, 182, 180, 183,
178, 182, 188, 175, 179, 184, 193, 182, 183, 175, 185, 182, 183, 156, 185, 199]

weights = [86, 71, 89, 64, 83, 71, 86, 68, 73, 84, 73, 72, 75, 78, 85, 93, 78,
70, 74, 80, 83, 82, 68, 80, 73, 78, 82, 81, 83, 79, 82, 86, 76, 77, 80, 103,
82, 83, 77, 89, 80, 85, 82, 85, 110]

ages = [58, 61, 54, 54, 58, 57, 63, 54, 37, 51, 34, 26, 50, 48, 45, 52, 56,
46, 54, 28, 52, 47, 55, 55, 54, 42, 51, 56, 55, 51, 54, 32, 21, 62, 43, 55,
56, 61, 53, 22, 64, 45, 54, 47, 38] 
 
df = pd.DataFrame ({"heights" : heights, "weights" : weights, "ages" : ages})

df = df.loc[0:40]

df_new = df[(df["heights"] > 170) & (df["weights"] > 80)]
print (df_new)
print(df_new.describe()) 


# ### Task 4.1

# In[ ]:


import pandas as pd 

df = pd.read_csv('precious_metal.csv', sep=';') 
print(df.shape)

print(df.info())


# ### Task 4.2

# In[ ]:


import pandas as pd 

df = pd.read_csv('precious_metal.csv', sep=';') 
print(df.info())

df_new = df.set_axis(['gold', 'silver', 'platinum', 'palladium', 'date'], axis='columns')
print(df_new)


# ### Task 4.3

# In[47]:


import pandas as pd

df = pd.read_csv('precious_metal.csv', sep=';')

df_1 = df.rename(columns ={'GOLDA':'gold', 'Silver':'silver', 'platinu':'platinum', 'Palla':'palladium', 'Date':'date'})
print(df_1)

df_2 = df_1.isnull().sum()
print(df_2)

df_3 = df_1.fillna(0)
print(df_3)


# ### Task 4.4

# In[48]:


import pandas as pd

df = pd.read_csv('precious_metal.csv', sep=';')

df = df.rename(columns ={'GOLDA':'gold', 'Silver':'silver', 'platinu':'platinum', 'Palla':'palladium', 'Date':'date'})

def replace_symbol(x):
    x = str(x)
    x = float(x.replace(',','.'))
    return x 

df['gold'] = df['gold'].apply(replace_symbol)
df['silver'] = df['silver'].apply(replace_symbol)
df['platinum'] = df['platinum'].apply(replace_symbol)
df['palladium'] = df['palladium'].apply(replace_symbol)

print(df)

