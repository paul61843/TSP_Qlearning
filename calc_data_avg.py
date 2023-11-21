import csv_utils

csv_num = 5

methods_name = ['greedy_and_mutihop', 'subTree_and_mutihop', 'NJNP_and_mutihop', 'q_learning']
time_interval = 1000


rows = csv_utils.read('./result/csv/csv0/q_learning.csv')
rowLength = len(csv_utils.read('./result/csv/csv0/q_learning.csv'))
columnLength = len(rows[0])


for method in methods_name:
    result = []
    for row_idx in range(1, rowLength):
        if (row_idx * 100) % time_interval == 0:
            
            values = [0] * columnLength
            
            for index in range(0, csv_num):
                rows = csv_utils.read(f'./result/csv/csv{index}/{method}.csv')
                columns = rows[row_idx]
                
                values = [a + int(b) for a, b in zip(values, columns)]
                
            values = [a / csv_num for a in values]
            result.append(values)
    csv_utils.writeDataToCSV(f'./result/csv/avg/{method}.csv', result)