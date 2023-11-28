import re
import pathlib
from utils import get_directory

def extract_data(file_path = 'database/2023-10-20.csv', overwrite = False):
    # Check if the data was already processed
    if not overwrite and pathlib.Path(file_path).is_file():
        return
    
    # Otherwise extract the data of interest
    f = open(file_path, 'r')
    lines = f.readlines()[1:]
    abbrevations = []
    for line in lines:
        columns = line.split(',')
        # We add the company name followed by its abbrevation
        columns[1] = re.sub("'", "", columns[1])
        abbrevations.append("'" + columns[1] + "' : '" + columns[0] + "'")
    f.close()
    
    # Create a python file with a dictionary containing the abbrevations
    data_path = get_directory(file_path) + "data.py"
    f = open(data_path, 'w')
    f.write("abbrevations = {\n")
    f.write(',\n'.join(abbrevations))
    f.write('\n}')
    f.close()    

if __name__ == "__main__":
    extract_data(overwrite = True)
        