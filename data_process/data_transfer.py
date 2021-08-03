# Zips pulse.csvs, sends to marconi
# Deletes remaining

import os, csv
import tarfile
import re

"""
from paramiko import SSHClient
from scp import SCPClient
"""

# directory of csvs
input_dir = "../JET_EFIT_magnetic/"

# place to put gzip output
# output_dir = "./output_dir"

# writes tar file from working directory (not necessarily the script location)
# ideally place in same dir as csvs?

dir_path = os.path.dirname(os.path.realpath(input_dir))

file_list = []
regexp = re.compile(r"[0-9]+[.]csv")

for folder, subfolder, files in os.walk(dir_path):
    for f in files:
        if regexp.search(f):
            # ignore self
            print(f)
            full_path = os.path.join(folder, f)
            file_list.append(full_path)

print(file_list)

# open for gzip compressed writing
with tarfile.open("pulses.tar.gzip", "w:gz") as tar:
    for name in file_list:
        tar.add(name)

print("TAR FILE CREATED")


# TODO potentially automate scp transfer here
"""
ssh = SSHClient()
ssh.load_system_host_keys()
ssh.connect('MARCONI_SERVER')

scp = SCPClient(ssh.get_transport())
scp.put('file_path_on_local_machine', 'file_path_on_remote_machine')
scp.close()
# see https://github.com/jbardin/scp.py 
"""


while True:
    try:
        delete_file = input("Delete remaining csvs? (y/n)")
        if (delete_file == "y") | (delete_file == "n"):
            break
        else:
            raise ValueError
    except ValueError:
        print("Error: must enter y or n")

if delete_file == "y":
    for file_to_delete in file_list:
        if os.path.isfile(file_to_delete):
            os.remove(file_to_delete)
        else:  ## Show an error ##
            print("Error: %s file not found" % file_to_delete)

# note: be careful where you run script from as will delete all csvs in that dir

