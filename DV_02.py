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



#2b

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib inline

movie_scores = pd.read_csv('movie_scores.csv')

plt.figure(figsize=(10, 5), dpi=300)
pos = np.arange(len(movie_scores['MovieTitle']))
width = 0.3
plt.bar(pos - width / 2, movie_scores['Tomatometer'], width, label='Tomatometer')
plt.bar(pos + width / 2, movie_scores['AudienceScore'], width, label='Audience Score')
plt.xticks(pos, rotation=10)
plt.yticks(np.arange(0, 101, 20))

ax = plt.gca()
ax.set_xticklabels(movie_scores['MovieTitle'])
ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
ax.set_yticks(np.arange(0, 100, 5), minor=True)
ax.yaxis.grid(which='major')
ax.yaxis.grid(which='minor', linestyle='--')

plt.title('Movie comparison')
plt.legend()
plt.show()



#2c
import pandas as sb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

bills = sns.load_dataset('tips')
days = ['Thur', 'Fri', 'Sat', 'Sun']
days_range = np.arange(len(days))
smoker = ['Yes', 'No']
bills_by_days = [bills[bills['day'] == day] for day in days]
bills_by_days_smoker = [[bills_by_days[day][bills_by_days[day]['smoker'] == s] for s in smoker] for day in days_range]
total_by_days_smoker = [[bills_by_days_smoker[day][s]['total_bill'].sum() for s in range(len(smoker))] for day in days_range]
totals = np.asarray(total_by_days_smoker)

plt.figure(figsize=(10, 5), dpi=300)
plt.bar(days_range, totals[:, 0], label='Smoker')
plt.bar(days_range, totals[:, 1], bottom=totals[:, 0], label='Non-smoker')
plt.legend()
plt.xticks(days_range)
ax = plt.gca()
ax.set_xticklabels(days)
ax.yaxis.grid()
plt.ylabel('Daily total sales in $')
plt.title('Restaurant performance')
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

