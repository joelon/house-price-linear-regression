#Tools needed :
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


#Adding the dataset :
data = {

    'Size': [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050],
    'Price': [200000, 250000, 300000, 350000, 400000, 450000, 500000, 550000, 600000, 650000, 700000, 750000, 800000, 850000, 900000, 950000, 1000000, 1050000, 1100000, 1150000]
}


df = pd.DataFrame(data)
print("Our dataset : ")
print(df)

x = df[['Size']]
y = df['Price']

#Splitting the dataset into training and testing sets :
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

#Training the model :
model = LinearRegression()
model.fit(x_train, y_train)
print("Model trained successfully.")

#Making predictions :
new_house = pd.DataFrame({'Size': [1200]})
price = model.predict(new_house)
print(f"The predicted price for a house of size 1200 is: ${price[0]:,.2f}")

#Drawing the graph :
plt.scatter(df['Size'], df['Price'], label='Real prices')
plt.plot(df['Size'],model.predict(x), color='red', label='Predicted prices')
plt.xlabel('Size of the house (in sq ft)')
plt.ylabel('Price of the house (in $)')
plt.title('House Price Prediction')
plt.legend()
plt.show()