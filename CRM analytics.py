#######################################
# Customer Segmentation with RFM
#######################################

############################
# Data prep.
############################
# Libraries to be used throughout the project
import pandas as pd
import numpy as np
import datetime as dt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.offline import iplot
import plotly.express as px
% matplotlib inline

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


#Let's load the data between 2010-2011
retail1=pd.read_csv("online_retail_10_11.csv",dtype = {'CustomerID': str})
retail=retail1.copy()
retail.head()


# create a function called check_df to examine the data in general.
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe().T)

check_df(retail)

# if there are missing values in our data,we will check
retail.isnull().sum()

#We see that there are missing values (NA).
retail.dropna(inplace=True)
retail.isnull().sum()

# Let's examine the unique product review in the dataset
retail["Description"].nunique()

# How many of each product are there?
retail["Description"].value_counts().head()

# Let's find the most ordered product with the help of agg funciton
# If you don't head, there will be no visualization process.
d=retail.groupby("Description").agg({"Quantity": "sum"}).sort_values("Quantity", ascending=False).head()
print(d)

fig = px.bar(d, title="Most ordered products", text_auto=True)
fig.update_traces(marker_color='red', marker_line_width=1.5, opacity=0.6)
iplot(fig)

# Now let's create the total price variable because we only have q and p in our dataset.
retail["TotalPrice"] = retail["Quantity"]*retail["UnitPrice"]

a=retail.groupby("CustomerID").agg({"TotalPrice": "sum"}).sort_values("TotalPrice", ascending=False).head(20)
print(a)

fig2 = px.bar(a, title="Top 20 customers", text_auto=True)
fig2.update_traces(marker_color='red', marker_line_width=1.5, opacity=0.6)
iplot(fig2)

# Let's go back to our data set and remove the canceled products.
retail =retail[~retail["InvoiceNo"].str.contains("C",na=False)]

check_df(retail)

#Calculate RFM metrics
retail.info()

retail['InvoiceDate'] = pd.to_datetime(retail['InvoiceDate'])
retail['InvoiceDate'].max()
#Timestamp('2011-12-09 12:50:00')

today_date=dt.datetime(2011,12,11)


rfm =retail.groupby("CustomerID").agg({"InvoiceDate": lambda InvoiceDate :(today_date-InvoiceDate.max()).days,
                                  "InvoiceNo": lambda InvoiceNo: InvoiceNo.nunique(),
                                  "TotalPrice": lambda TotalPrice: TotalPrice.sum()})

rfm.columns=["recency","frequency","monetary"]
rfm.describe().T

rfm = rfm[rfm["monetary"] > 0]
rfm=  rfm[rfm["frequency"]>0]
rfm.shape

# Create RFM score and segmentation
#Recency score
rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
#Monetary score
rfm["monetary_score"]= pd.qcut(rfm['monetary'],5, labels=[1, 2, 3, 4, 5])
#FREQUENCY SCORE
rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])


rfm["RF_SCORE"]=rfm["frequency_score"].astype(str)+rfm["recency_score"].astype(str)


# RFM naming
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)
rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])

#Customer Segmentation Map
pip install squarify
import squarify
segments = rfm['segment'].value_counts().sort_values(ascending = False)
fig = plt.gcf()
ax = fig.add_subplot()
fig.set_size_inches(16, 10)
squarify.plot(sizes=segments,
              label=[label for label in seg_map.values()],
              color=['#AFB6B5', '#F0819A', '#926717', '#F0F081', '#81D5F0',
                     '#C78BE5', '#748E80', '#FAAF3A', '#7B8FE4', '#86E8C0'],
              pad = False,
              bar_kwargs = {'alpha': 1},
              text_kwargs = {'fontsize':15})
plt.title("Customer Segmentation Map", fontsize = 20)
plt.xlabel('Frequency', fontsize = 18)
plt.ylabel('Recency', fontsize = 18)
plt.show()

#Visualation
c = rfm["segment"].value_counts(sort=False)
colors = (
"grey", "seagreen", "cornflowerblue", "plum", "mediumorchid", "peru", "lightskyblue", "darkslategrey", "yellow",
"orange")
explodes = [0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20, 0.20]

c.plot.pie(colors=colors, textprops={'fontsize': 12}, autopct='%8.1f', startangle=180, radius=3, rotatelabels=True,
           shadow=True, explode=explodes)

#Customer Strategy
#Loyal Customers
#Loyal customer accounts for 17.3% of the total number of customers. The number of Loyal customers is 752 people. They last received service an average of 16 days ago. They bought services from a company with a mean of £1,457. Our company for these customers should increase the pound value they spend. Specific campaigns can be organised. For instance, discount coupons may be given or discounts may be made on products that are forgotten in the basket.

#At_risk
# At-risk customers accounts for 13% of the total number of customers. There are 565 people at risk. They last received services about 42 days ago on average. They were served an average of 450£. Our company can send an email showing that these customers have made a small discount on the products they searched for, or they can prepare and send cards on special days to show that they still care about these customers.
