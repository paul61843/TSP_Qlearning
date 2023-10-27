import csv
import numpy as np
import copy


def write(path='./result/train_table.csv', content = []):
    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(
            csvfile,
            quotechar='|', 
            quoting=csv.QUOTE_MINIMAL
        )

        for item in content:
            writer.writerow(item)


def read(path='./result/train_table.csv'):
    with open(path, newline='') as f:
        reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONE)

        csv_content = [row for row in reader]

    return csv_content

def writeDataToCSV(path, data):
    new_row = np.array([[
        'run_time'.ljust(15), 
        'total_data'.ljust(15),  
        'mutihop'.ljust(15), 
        'sensor_data_origin'.ljust(15),   
        'sensor_data_calc'.ljust(15),   
        'sensor_data'.ljust(15),   
        'uav_data'.ljust(15),
        'lost_data'.ljust(15),
    ]])
    data = np.array(copy.deepcopy(data))
    
    for (x,y), value in np.ndenumerate(data):
        data[x][y] = str(data[x][y]).ljust(15)
    
    concatArr = np.concatenate((new_row, data), axis=0)
    
    
    write(path, concatArr)