import glob

# Path to the directory containing your files
output_file = 'averaged_file.test_case'

# Function to read the contents of each file
def read_file(filename):
    with open(filename, 'r') as file:
        return file.readlines()

# Read all files in the directory
file_pattern = f'forces.*'
files = glob.glob(file_pattern)
content = [read_file(file) for file in files]

# Transpose the content to group corresponding lines together and exclude headers if necessary
transposed_content = map(list, zip(*content))
average_content = []

# Calculate averages for each line
for lines in transposed_content:
    if all(line.strip() == lines[0].strip() for line in lines):  # Check if the line is text
        average_content.append(lines[0])  # Keep the text as is
    else:
        # Calculate the average of numeric values in lines
        values = [float(line.strip()) for line in lines]
        average_value = sum(values) / len(values)
        average_content.append(f'{average_value}\n')  # Add the averaged value

# Write the averaged content to a new file
with open(output_file, 'w') as file:
    file.writelines(average_content)
