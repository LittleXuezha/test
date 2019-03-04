import pandas as pd
df1 = pd.DataFrame({'employee': ['bob','jake', 'lisa', 'sue'],
                    'group':['accounting', 'engineering', 'engineering', 'HR']})
df2 = pd.DataFrame({'employee': ['lisa', 'bob', 'jake', 'sue'],
                    'hire_data': [2004, 2008, 2012, 2014]})
df4 = pd.DataFrame({'group':['accounting', 'engineering'],
                    'supervisor': ['carly', 'guido']})
df5 = pd.DataFrame({'group': ['accounting', 'accounting',
                              'engineering','engineering'
                              'HR', 'HR'],
                    'skills': ['math', 'spread', 'coding', 'linux',
                               'spr', 'org']})
df6 = pd.DataFrame({'name': ['jake', 'bob', 'lisa', 'sue'],
                    'salary': [7000,8000,9000,10000]})
df1a = df1.set_index('employee')
df2a = df2.set_index('employee')
print(pd.merge(df1a, df2, left_index=True, right_on='employee'))