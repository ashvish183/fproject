import os

folder = 'D:\\project\\kaggle\\colored_images\\Severe\\'
count = 1
for file_name in os.listdir(folder):
    # Construct old file name
    source = folder + file_name
    # print(file_name)
    # Adding the count to the new file name and extension
    destination = str(count) + ".png"
    print(destination)
    # os.rename(source, destination)
    count += 1
print('All Files Renamed')

print('New Names are')
# verify the result
res = os.listdir(folder)
# print(res)