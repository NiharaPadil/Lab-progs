#2 a
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

google = pd.read_csv('GOOGL_data.csv')
facebook = pd.read_csv('FB_data.csv')
apple = pd.read_csv('AAPL_data.csv')
amazon = pd.read_csv('AMZN_data.csv')
microsoft = pd.read_csv('MSFT_data.csv')

for df in [google, facebook, apple, amazon, microsoft]:
    df['date'] = pd.to_datetime(df['date'])

plt.figure(figsize=(16, 8), dpi=300)
plt.plot(google['date'], google['close'], label='Google')
plt.plot(facebook['date'], facebook['close'], label='Facebook')
plt.plot(apple['date'], apple['close'], label='Apple')
plt.plot(amazon['date'], amazon['close'], label='Amazon')
plt.plot(microsoft['date'], microsoft['close'], label='Microsoft')

plt.xticks(rotation=70)
plt.yticks(np.arange(0, 1450, 100))
plt.title('Stock Trend', fontsize=16)
plt.ylabel('Closing Price in $', fontsize=14)
plt.grid()
plt.legend()
plt.show()

#2a shorten code 
# import pandas as pd
# import matplotlib.pyplot as plt

# files = ['GOOGL_data.csv', 'FB_data.csv', 'AAPL_data.csv', 'AMZN_data.csv', 'MSFT_data.csv']
# labels = ['Google', 'Facebook', 'Apple', 'Amazon', 'Microsoft']
# data = [pd.read_csv(file).assign(date=lambda df: pd.to_datetime(df['date'])) for file in files]

# plt.figure(figsize=(16, 8), dpi=300)
# for df, label in zip(data, labels):
#     plt.plot(df['date'], df['close'], label=label)

# plt.xticks(rotation=70)
# plt.yticks(range(0, 1450, 100))
# plt.title('Stock Trend', fontsize=16)
# plt.ylabel('Closing Price in $', fontsize=14)
# plt.grid()
# plt.legend()
# plt.show()



#2b
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

movie_scores = pd.read_csv('movie_scores.csv')

plt.figure(figsize=(10, 5), dpi=300)
pos = np.arange(len(movie_scores))
width = 0.3

plt.bar(pos - width / 2, movie_scores['Tomatometer'], width, label='Tomatometer')
plt.bar(pos + width / 2, movie_scores['AudienceScore'], width, label='Audience Score')

plt.xticks(pos, movie_scores['MovieTitle'], rotation=10)
plt.yticks(np.arange(0, 101, 20), labels=['0%', '20%', '40%', '60%', '80%', '100%'])
plt.grid(axis='y', which='both', linestyle='--')

plt.title('Movie Comparison')
plt.legend()
plt.show()




#2c
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

bills = sns.load_dataset('tips')
days = ['Thur', 'Fri', 'Sat', 'Sun']
totals = np.array([
    [bills[(bills['day'] == day) & (bills['smoker'] == smoker)]['total_bill'].sum() for smoker in ['Yes', 'No']]
    for day in days
])

plt.figure(figsize=(10, 5), dpi=300)
plt.bar(range(len(days)), totals[:, 0], label='Smoker')
plt.bar(range(len(days)), totals[:, 1], bottom=totals[:, 0], label='Non-smoker')

plt.xticks(range(len(days)), days)
plt.ylabel('Daily Total Sales in $')
plt.title('Restaurant Performance')
plt.grid(axis='y')
plt.legend()
plt.show()



#2d

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

sales = pd.read_csv('smartphone_sales.csv')
plt.figure(figsize=(10, 6), dpi=300)
labels = sales.columns[2:]
plt.stackplot(sales['Quarter'], sales['Apple'], sales['Samsung'], sales['Huawei'], sales['Xiaomi'], sales['OPPO'], labels=labels, data=sales)
plt.legend()
plt.xlabel('Quarters')
plt.ylabel('Sales units in thousands')
plt.title('Smartphone sales units')
plt.show()

