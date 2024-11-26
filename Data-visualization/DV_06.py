#a
import geoplotlib

from geoplotlib.utils import DataAccessObject

import pandas as pd

 

# Load the dataset

try:

    # Read the original dataset

    data = pd.read_csv('bus.csv')

   

    # Check for required columns

    if 'lat' not in data.columns or 'lon' not in data.columns:

        print("Adding dummy geographical data ('lat' and 'lon') for testing.")

       

        # Add dummy latitude and longitude columns

        # You can replace these values with real coordinates if needed

        data['lat'] = [12.9716, 13.0827, 28.7041, 19.0760, 22.5726, 0, 0]

        data['lon'] = [77.5946, 80.2707, 77.1025, 72.8777, 88.3639, 0, 0]

 

    # Save the modified dataset (optional)

    data.to_csv('bus_with_coords.csv', index=False)

 

    # Create a DataAccessObject for geoplotlib

    geodata = DataAccessObject.from_dataframe(data)

 

    # Plot the points on the map

    geoplotlib.dot(geodata)

    geoplotlib.show()

 

except FileNotFoundError:

    print("Error: The file 'bus.csv' was not found. Please check the file path.")

except Exception as e:

    print(f"An error occurred: {e}")





#b

import geoplotlib

from geoplotlib.utils import DataAccessObject

import pandas as pd

 

# Load the dataset

try:

    # Read the original dataset

    data = pd.read_csv('bus.csv')

   

    # Check for required columns

    if 'lat' not in data.columns or 'lon' not in data.columns:

        print("Adding dummy geographical data ('lat' and 'lon') for testing.")

       

        # Add dummy latitude and longitude columns

        # You can replace these values with real coordinates if needed

        data['lat'] = [12.9716, 13.0827, 28.7041, 19.0760, 22.5726, 0, 0]

        data['lon'] = [77.5946, 80.2707, 77.1025, 72.8777, 88.3639, 0, 0]

 

    # Save the modified dataset (optional)

    data.to_csv('bus_with_coords.csv', index=False)

 

    # Create a DataAccessObject for geoplotlib

    geodata = DataAccessObject.from_dataframe(data)

 

    # Plot the points on the map

    geoplotlib.dot(geodata)

    geoplotlib.show()

 

except FileNotFoundError:

    print("Error: The file 'bus.csv' was not found. Please check the file path.")

except Exception as e:

    print(f"An error occurred: {e}")
