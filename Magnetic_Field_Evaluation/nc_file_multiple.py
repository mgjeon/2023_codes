import os 

with open("noaa.txt", "r") as f:
    lines = f.readlines()

for line in lines:
    noaanum = line.strip()
    print(noaanum) 
    os.system(f'bash nc_file.sh {noaanum}')