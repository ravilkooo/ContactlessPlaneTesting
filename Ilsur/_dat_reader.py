import numpy as np
import re

# Function to read the table of numbers
def read_connectivity_table(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        start_index = None
        end_index = None

        # Find the indices of "connectivity" and "coordinates"
        for i, line in enumerate(lines):
            if "connectivity" in line:
                start_index = i + 1
            elif "coordinates" in line:
                end_index = i - 1
                break

        # Read the table of numbers between "connectivity" and "coordinates"
        if start_index is not None and end_index is not None:
            # table = [list(map(int, line.split())) for line in lines[start_index+1:end_index + 1]]
            def int_mns_1(x):
                return int(x) - 1
            

            def line_record(line):
                return np.array(list(map(int_mns_1, line.split())))[np.arange(len(line.split())) != 1].tolist()
            
            
            num = len(lines[start_index+1:end_index + 1])
            table = np.zeros((num, 3))
            for line in lines[start_index+1:end_index + 1]:
                rec_line = line_record(line)
                table[rec_line[0]] = rec_line[1:]



            return table
        else:
            print("Table not found in the file.")
            return []

# Function to read the table of numbers
def read_coordinates_table(filename):
    with open(filename, 'r') as file:
        data = file.read()

        # Extract the text between "coordinates" and "attach node"
        pattern = r"coordinates(.*?)attach node"
        table_text = re.search(pattern, data, re.DOTALL).group(1)

        # Split the table text into lines and extract numbers from each line
        lines = table_text.strip().split('\n')

        def split_row(line):
            res = line.split()
            if len(res) == 4:
                return res
            minus_spl = res[0].split('-')
            idx = minus_spl[0]
            first = '-' + '-'.join(minus_spl[1:])
            return [idx, first, *res[1:]]

        def convert_to_scientific_notation(s):
            # Use a regular expression to find the pattern "Mp" where M is any number and p is optional
            pattern = r'(\d+\.\d*)[eE]?([\+\-]?\d*)'
            result = re.sub(pattern, r'\1e\2', s)
            return result
        
        def row_to_scirow(l):
            return list(map(convert_to_scientific_notation, split_row(l)))

        def float_mm2px(x):
            return float(x) * 10.6

        def scirow_to_vals(l):
            _l = row_to_scirow(l)
            return list([int(_l[0])-1, *list(map(float_mm2px, _l[1:]))])
        
        _, num, _, _ = map(int, lines[0].split())
        
        table = np.zeros((num, 3))
        for line in lines[1:]:
            all_vals = scirow_to_vals(line)
            table[all_vals[0]] = all_vals[1:]

        return num, table
