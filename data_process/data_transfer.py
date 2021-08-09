# Zips pulse.csvs, sends to marconi
# Deletes remaining

import os, csv
import tarfile
import re

from paramiko import SSHClient
from scp import SCPClient
import sys

# Define progress callback that prints the current percentage completed for the file
def progress(filename, size, sent):
    sys.stdout.write(
        "%s's progress: %.2f%%   \r" % (filename, float(sent) / float(size) * 100)
    )


# writes tar file to directory it was called from (not necessarily the script location)
# ideally place in same dir as csvs, bc gathers csvs from directory it is placed in

dir_path = os.path.dirname(os.path.realpath(__file__))

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

# TODO make gzip file naming more specific

# open for gzip compressed writing
with tarfile.open("pulses.tar.gzip", "w:gz") as tar:
    for name in file_list:
        tar.add(name)

print("TAR FILE CREATED")


# TODO potentially automate scp transfer here

ssh = SSHClient()
ssh.load_system_host_keys()
ssh.connect("login.marconi.cineca.it")

scp = SCPClient(ssh.get_transport(), progress=progress)
scp.put(
    "/common/scratch/elewis/pulses.tar.gzip",
    "/marconi_work/FUA35_WPJET1/elewis/pulses.tar.gzip",
)
scp.close()


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

