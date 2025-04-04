This project uses a Linear Regression model to predict house prices based on factors like location, income, and other housing details.

pip install pandas numpy matplotlib scikit-learn
About the Dataset
The dataset, housing.csv, has columns like:

longitude, latitude: Coordinates for the location.

housing_median_age: How old the houses are, on average.

total_rooms, total_bedrooms: How many rooms and bedrooms are in the houses.

population, households: Data about the people living in the area.

median_income: The income level of people in the area.

median_house_value: The house price (this is what we're predicting).

ocean_proximity: Whether the house is near the ocean or not.

Place housing.csv in the same folder as the script.

Load the data and fix any missing values.

Train a Linear Regression model to predict house prices.

Show a plot that compares the actual vs predicted prices.

What You’ll See
After running the script, you’ll get:

RMSE (Root Mean Squared Error): Tells you how much the predictions are off on average.

R²: Shows how well the model explains the data.
