#a
import geoplotlib
from geoplotlib.utils import DataAccessObject
import pandas as pd

try:
    data = pd.read_csv('bus.csv')

    if 'lat' not in data.columns or 'lon' not in data.columns:
        data['lat'] = [12.9716, 13.0827, 28.7041, 19.0760, 22.5726, 0, 0]
        data['lon'] = [77.5946, 80.2707, 77.1025, 72.8777, 88.3639, 0, 0]

    data.to_csv('bus_with_coords.csv', index=False)

    geodata = DataAccessObject.from_dataframe(data)
    geoplotlib.dot(geodata)
    geoplotlib.show()

except FileNotFoundError:
    print("Error: The file 'bus.csv' was not found.")

except Exception as e:
    print(f"An error occurred: {e}")


#b
import geoplotlib
from geoplotlib.utils import DataAccessObject
import pandas as pd

try:
    data = pd.read_csv('bus.csv')

    if 'lat' not in data.columns or 'lon' not in data.columns:
        data['lat'] = [12.9716, 13.0827, 28.7041, 19.0760, 22.5726, 0, 0]
        data['lon'] = [77.5946, 80.2707, 77.1025, 72.8777, 88.3639, 0, 0]

    data.to_csv('bus_with_coords.csv', index=False)

    geodata = DataAccessObject.from_dataframe(data)
    geoplotlib.heatmap(geodata, radius=15, blur=15)
    geoplotlib.show()

except FileNotFoundError:
    print("Error: The file 'bus.csv' was not found.")

except Exception as e:
    print(f"An error occurred: {e}")

