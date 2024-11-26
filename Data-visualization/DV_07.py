# a)
import folium
my_map1 = folium.Map(location = [28.5011226, 77.4099794],zoom_start = 12 )
my_map1.save(" my_map1.html ")




# b)
import folium
my_map2 = folium.Map(location = [28.5011226, 77.4099794],zoom_start = 12)
folium.CircleMarker(location = [28.5011226, 77.4099794],radius = 50, popup = ' FRI ').add_to(my_map2)
my_map2.save(" my_map2.html ")



# c)
import folium
my_map3 = folium.Map(location = [28.5011226, 77.4099794],zoom_start = 15)
folium.Marker([28.5011226, 77.4099794],popup = ' Geeksforgeeks.org ').add_to(my_map3)
my_map3.save(" my_map3.html ")




import folium
my_map4 = folium.Map(location=[28.5011226, 77.4099794], zoom_start=12)
folium.Marker([28.704059, 77.102490], popup='Delhi').add_to(my_map4)
folium.Marker([28.5011226, 77.4099794], popup='GeeksforGeeks').add_to(my_map4)
folium.PolyLine(locations=[(28.704059, 77.102490), (28.5011226, 77.4099794)], line_opacity=0.5).add_to(my_map4)
my_map4.save("my_map4.html")







import os
print(os.getcwd())  
