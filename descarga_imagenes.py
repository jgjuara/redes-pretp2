#%%
import urllib.request
import zipfile

# Download the zip file
url = "https://www.dropbox.com/scl/fi/mxv5o39ekf6bgshz0g5jf/kaggle_flower_images.zip?rlkey=90gi41kzonieowglmbw75yop3&st=ynk3dv7h&dl=1"
urllib.request.urlretrieve(url, "kaggle_flower_images.zip")

# Extract the zip file
with zipfile.ZipFile("kaggle_flower_images.zip", 'r') as zip_ref:
    zip_ref.extractall()

image_path = "kaggle_flower_images"
path = sorted([os.path.join(image_path, file)
for file in os.listdir(image_path )
if file.endswith('.png')])

print(len(path))
print(path[0])

path

"""Recupero archivo labels"""

image_path = "kaggle_flower_images"
pathLabel = sorted([os.path.join(image_path, file)
for file in os.listdir(image_path )
if file.endswith('.csv')])

print(len(pathLabel))
print(pathLabel[0])
