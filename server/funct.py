import numpy as np
from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd

df1 = pd.read_csv('C:/yolov5-master/yolov5-master/Trafficlight.csv')
df2 = pd.read_csv('C:/yolov5-master/yolov5-master/Stop_Line.csv')


def haversine(lat1, lon1, lat2, lon2):
    radius = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = radius * c
    return distance * 1000

def gps(web):
    response = urlopen(web)
    soup = BeautifulSoup(response, "html.parser")
    value = soup.find("body")
    value2 = value.text.strip()
    value3 = value2.split(',')
    latitude = float(value3[0])
    longitude = float(value3[1])
    return latitude, longitude

def nearest_TLpoint(current_lat, current_long):
    df1['distance'] = df1.apply(lambda row: haversine(current_lat, current_long,
                                                      row['latitude'], row['longitude']), axis=1)
    closest_point = df1.loc[df1['distance'].idxmin()]
    TL_Distance = closest_point['distance']

    return TL_Distance

def nearest_SLpoint(current_lat, current_long):
    df2['distance'] = df2.apply(lambda row: haversine(current_lat, current_long,
                                                      row['latitude'], row['longitude']), axis=1)
    closest_point = df2.loc[df1['distance'].idxmin()]
    SL_Distance = closest_point['distance']

    return closest_point, SL_Distance


def split_line(line):
    try:
        cur = str(line[0])
        cur = cur[7:]
        cur = cur.split('.')
        cur_class = int(cur[0])
        x = line[1]
        y = line[2]
        w = line[3]
        h = line[4]

        return cur_class, x, y, w, h  # class와 size 반환
    except Exception as error:
        return print('error')