import re
import glob

# Construct output file name
output_file = f'averaged_file.test_case'

skip_lines: dict = {2, 14, 26, 27, 32, 37, 42}

# Construct a simple local function to load in the contents of each file in the directory
def read_file(filename):
    with open(filename, 'r') as file:
        return file.readlines()

# Read all files in the directory
file_pattern = f'forces.*'
files = glob.glob(file_pattern)
content = [read_file(file) for file in files]
        
# Transpose content to group corresponding lines together
transposed_content = list(map(list, zip(*content)))
average_content = []

# Regular expression to match variable = value pairs with varying spaces
var_value_pattern = re.compile(r'([\w\s]+)\s*=\s*([-+]?\d*\.?\d+([eE][-+]?\d+)?)')

# Regular expression to match scientific notation and numeric values
value_pattern = re.compile(r'[-+]?\d*\.?\d+([eE][-+]?\d+)?')

# Process each group of corresponding lines
for idx, lines in enumerate(transposed_content):
    line_num = idx + 1

    # Check if the current line should be skipped
    if line_num in skip_lines:
        average_content.append(lines[0])
        continue

    line_text = lines[0]

    # Handling single values after "="
    if all('=' in line and len(line.split('=')[1].split()) == 1 for line in lines):
        values = [float(value_pattern.search(line.split('=')[1]).group()) for line in lines]
        average_value = sum(values) / len(values)
        line_text = f'{line_text.split("=")[0].strip()} = {average_value:.5E}\n'
            
    # Handling multiple values in "variable=data1 variable=data2" structure
    elif all('=' in line for line in lines) and any(len(line.split('=')) > 2 for line in lines):
        var_values_dict = {}
        for line in lines:
            var_values = var_value_pattern.findall(line)
            for var, value, _ in var_values:
                if var not in var_values_dict:
                    var_values_dict[var] = []
                var_values_dict[var].append(float(value))
        avg_values = [f'{var} = {sum(values) / len(values):.5E}' for var, values in var_values_dict.items()]
        line_text = ' '.join(avg_values) + '\n'
            
    # Handling multiple values separated by spaces in "variable: data1 data2 data3 data4" structure
    elif all(':' in line for line in lines):
        text_part = lines[0].split(':')[0].strip() + ': '
        all_values = [list(map(float, line.split(':')[1].split())) for line in lines]
        avg_values = [sum(col) / len(col) for col in list(zip(*all_values))]
        line_text = text_part + '    '.join(f'{val:.5E}' for val in avg_values) + '\n'

    average_content.append(line_text)

    # Write the averaged content to a new file
    with open(output_file, 'w') as file:
        file.writelines(average_content)