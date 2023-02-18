import csv


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