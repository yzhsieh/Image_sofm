import os
import json
all_name = dict()


if __name__ == '__main__':
    all_dir = os.listdir('CorelDB2')
    for item in all_dir:
        print("Processing : ",item)
        file_list = os.listdir('./CorelDB2/' + item)
        if "Thumbs.db" in file_list:
            print("Find Thumbs.db, deleted it")
            file_list.remove("Thumbs.db")
        all_name[item] = file_list
    # print(all_name)
    file = open('filename_list.txt','w')
    json.dump(all_name, file)