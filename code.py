#1

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv("C:\\Users\\aswin\\Desktop\\GDP per capita (current US$).csv")

# Select the columns for clustering (e.g., GDP per capita)
columns_of_interest = ['Country Name', 'Country Code', '1960', '1970', '1980', '1990', '2000', '2010', '2020']

# Filter the data for the selected columns
filtered_data = data[columns_of_interest]

# Drop rows with missing values
filtered_data.dropna(inplace=True)

# Normalize the data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(filtered_data.iloc[:, 2:])

# Perform clustering using K-means
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(normalized_data)

# Add cluster labels to the data
filtered_data['Cluster'] = clusters

# Plot the clusters
plt.figure(figsize=(10, 6))
plt.scatter(filtered_data['2020'], filtered_data['1970'], c=filtered_data['Cluster'], cmap='viridis')
plt.xlabel('GDP per capita in 2020')
plt.ylabel('GDP per capita in 1970')
plt.title('Clustering of Countries based on GDP per capita')
plt.colorbar(label='Cluster')
plt.show()

# Compute cluster centers
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)

# Display cluster centers
print('Cluster Centers:')
for i, center in enumerate(cluster_centers):
    print(f'Cluster {i}: {center}')



#2

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

# Define the function model
def exponential_growth(x, a, b):
    return a * np.exp(b * x)

# Get the data from GDPdata DataFrame
df = pd.read_csv("C:\\Users\\aswin\\Desktop\\GDP per capita (current US$).csv")  # Assuming the data is in a CSV file
x = df.columns[4:].astype(int)  # Assuming the columns are in the format '1960', '1961', ...
y = df.loc[df['Country Name'] == 'United States'].values[0, 4:].astype(float)  # Replace 'Your Country Name'

# Fit the data to the model
params, _ = curve_fit(exponential_growth, x, y)

# Generate predictions for the next 10 years
future_years = np.arange(2022, 2032)
predictions = exponential_growth(future_years, *params)

# Calculate confidence ranges using err_ranges function
def err_ranges(x, y, params, function):
    popt = params
    perr = np.sqrt(np.diag(np.abs(np.linalg.inv(hessian))))
    nstd = 1.96  # 95% confidence interval
    predicted_data = function(x, *popt)
    interval = perr * nstd
    lower = predicted_data - interval
    upper = predicted_data + interval
    return lower, upper

hessian = np.outer(np.gradient(predictions, future_years), np.gradient(predictions, future_years))
lower, upper = err_ranges(future_years, predictions, params, exponential_growth)

# Plot the data, best fitting function, and confidence range
plt.plot(x, y, 'bo', label='Data')
plt.plot(future_years, predictions, 'r-', label='Best Fit')
plt.fill_between(future_years, lower, upper, color='gray', alpha=0.3, label='Confidence Range')
plt.xlabel('Years')
plt.ylabel('GDP per capita (current US$)')
plt.legend()
plt.title('Exponential Growth Model Fit')
plt.show()


#3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the data
GDPdata = pd.read_csv("C:\\Users\\aswin\\Desktop\\GDP per capita (current US$).csv")

# Extract the relevant columns for clustering
GDPdata_cluster = GDPdata[['Country Name'] + [str(year) for year in range(1960, 2022)]]

# Remove rows with missing values
GDPdata_cluster.dropna(inplace=True)

# Normalize the data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(GDPdata_cluster.iloc[:, 1:])  # Exclude the 'Country Name' column

# Perform clustering using K-means
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(normalized_data)

# Add cluster labels to the data
GDPdata_cluster['Cluster'] = clusters

# Select two countries from the same cluster for comparison
cluster_id = 0  # Replace with the desired cluster ID
countries_to_compare = GDPdata_cluster[GDPdata_cluster['Cluster'] == cluster_id].sample(n=2, random_state=42)

# Plot the scatter plots for the selected countries
plt.figure(figsize=(30, 6))
for _, country_row in countries_to_compare.iterrows():
    country_name = country_row['Country Name']
    country_data = country_row.values[1:-1]  # Exclude the 'Country Name' and 'Cluster' columns
    years = GDPdata_cluster.columns[1:-1]  # Exclude the 'Country Name' and 'Cluster' columns
    plt.plot(years, country_data, label=country_name)

plt.xlabel('Year')
plt.ylabel('GDP per capita (current US$)')
plt.title('Comparison of GDP per capita for Countries in Cluster {}'.format(cluster_id))
plt.legend()

# Rotate x-axis labels
plt.xticks(rotation='vertical')

plt.show()

