"""
This code is provided as a courtesy from Bram Meijerink. 
It is fairly simple to understand/use, and left untouched. 

"""

import subprocess
import threading
import numpy as np
import re
import csv
import matplotlib.pyplot as plt

def read_output(process):
    i = 1
    j = 1
    for line in process.stdout:
        J_match = re.search(r"J\s*:\s*([\d\.E+-]+)", line)
        CT_match = re.search(r"Ct\s*:\s*([\d\.E+-]+)", line)
        CP_match = re.search(r"Cp\s*:\s*([\d\.E+-]+)", line)
        TC_match = re.search(r"Tc\s*:\s*([\d\.E+-]+)", line)
        if J_match:
            i = i +1
            if i % 2 == 0:
                J = float(J_match.group(1))
                CT = float(CT_match.group(1))
                CP = float(CP_match.group(1))
                J_list.append(J)
                CT_list.append(CT)
                CP_list.append(CP)
        if TC_match:
            j = j +1
            if j % 2 == 0:
                TC = float(TC_match.group(1))
                TC_list.append(TC)

# Start the process
process = subprocess.Popen(
    ["dfdc.exe"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1
)

# Start a thread to read output
thread = threading.Thread(target=read_output, args=(process,), daemon=True)
thread.start()

v_list = np.linspace(0, 100, 50)
rpm_list = np.linspace(500, 4000, 50)
J_list = []
CT_list = []
CP_list = []
TC_list = []


# Send input continuously
try:
    while process.poll() is None:
        process.stdin.write("load beta_29.case\n")
        process.stdin.write("ppar\n")
        process.stdin.write("\n")
        process.stdin.write("\n")
        process.stdin.write("pane\n")
        process.stdin.write("oper\n")
        # for v in v_list:
        process.stdin.write(f"vinf 35\n")
        for rpm in rpm_list:
            process.stdin.write(f"RPM {rpm}\n")
            process.stdin.write("e\n")
        next = input("Next?")
        if next == "":
            process.terminate()
        else:
            process.stdin.write(f"{next}\n")
except KeyboardInterrupt:
    print("Exiting interaction...")

process.terminate()
data_out=np.transpose([J_list, CT_list, CP_list, TC_list])
data_out=np.vstack((np.array(["J", "CT", "CP", "TC"]), data_out))
with open('dfdc_J_range.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data_out)

plt.plot(J_list, CT_list, label="CT")
plt.plot(J_list, CP_list, label="CP")
plt.legend()
plt.show()

plt.plot(J_list, TC_list)
plt.show()
