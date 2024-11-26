# Import folium package
import folium

# ------------------------------------------------------------
# (a) Basic Map
my_map1 = folium.Map(location=[28.5011226, 77.4099794], zoom_start=12)
# Save the map as an HTML file
my_map1.save("my_map1.html")
print("Map 1 (Basic Map) saved as 'my_map1.html'")

# ------------------------------------------------------------
# (b) Map with a Circle Marker
my_map2 = folium.Map(location=[28.5011226, 77.4099794], zoom_start=12)
# Add a CircleMarker to the map
folium.CircleMarker(
    location=[28.5011226, 77.4099794],
    radius=50,  # Radius of the circle
    popup='FRI'  # Popup text
).add_to(my_map2)
# Save the map as an HTML file
my_map2.save("my_map2.html")
print("Map 2 (Circle Marker) saved as 'my_map2.html'")

# ------------------------------------------------------------
# (c) Map with a Single Marker
my_map3 = folium.Map(location=[28.5011226, 77.4099794], zoom_start=15)
# Add a Marker with a popup
folium.Marker(
    [28.5011226, 77.4099794],
    popup='Geeksforgeeks.org'
).add_to(my_map3)
# Save the map as an HTML file
my_map3.save("my_map3.html")
print("Map 3 (Single Marker) saved as 'my_map3.html'")

# ------------------------------------------------------------
# (d) Map with Multiple Markers and a Polyline
my_map4 = folium.Map(location=[28.5011226, 77.4099794], zoom_start=12)
# Add markers
folium.Marker(
    [28.704059, 77.102490], popup='Delhi'
).add_to(my_map4)
folium.Marker(
    [28.5011226, 77.4099794], popup='GeeksforGeeks'
).add_to(my_map4)
# Add a polyline connecting the markers
folium.PolyLine(
    locations=[
        (28.704059, 77.102490),  # Delhi coordinates
        (28.5011226, 77.4099794)  # GeeksforGeeks coordinates
    ],
    line_opacity=0.5
).add_to(my_map4)
# Save the map as an HTML file
my_map4.save("my_map4.html")
print("Map 4 (Multiple Markers and Polyline) saved as 'my_map4.html'")

# ------------------------------------------------------------
# Check the current working directory to locate saved HTML files
import os
print("Current working directory:", os.getcwd())
