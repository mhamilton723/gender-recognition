import requests
import zipfile
import os
import subprocess

url = 'https://github.com/mhamilton723/gender-recognition/archive/master.zip'
r = requests.get(url)

print("Downloading Repo")
repo_zip = "repo.zip"
with open(repo_zip, 'wb') as f:
    f.write(r.content)

zip_ref = zipfile.ZipFile(repo_zip, 'r')
zip_ref.extractall(".")
zip_ref.close()

os.remove("repo.zip")
os.rename("gender-recognition-master", "gender-recognition")


print("Creating conda environment")
os.system("cd gender-recognition")
os.system("conda env create -f conda-env.yml")
os.system("activate deep-learning-env")
os.system("cd src")
os.system("python get_data.py")








