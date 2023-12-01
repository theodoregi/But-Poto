#### copy the files from the data/mask/log1 folder to the ML/mask/log1 folder and register the name of the files
import os
import shutil
import csv
import sys
import numpy as np

def copy_mask_files(repo_name):
    ### mkdir the repository
    if not os.path.exists('./ML/mask/'+repo_name):
        os.makedirs('./ML/mask/'+repo_name)

    # copy the files from the data/mask/log1 folder to the ML/mask/log1 folder
    for filename in os.listdir('./data/mask/'+repo_name):
        shutil.copy('./data/mask/'+repo_name+'/'+filename, './ML/mask/'+repo_name+'/'+filename)

    # register the name of the files
    with open('./ML/mask/'+repo_name+'/'+repo_name+'.csv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for filename in os.listdir('./ML/mask/'+repo_name):
            spamwriter.writerow([filename])
    return

### for each file in csv files in ML/mask/log*, copy this files from ./data/log*/ to ./ML/image/log*/

def copy_image_files(repo_name):
    ### mkdir the repository
    if not os.path.exists('./ML/image/'+repo_name):
        os.makedirs('./ML/image/'+repo_name)

### check if the file is in the csv file
    with open('./ML/mask/'+repo_name+'/'+repo_name+'.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            filename = row[0]
            if filename.endswith('.png'):
                shutil.copy('./data/'+repo_name+'/'+filename, './ML/image/'+repo_name+'/'+filename)
    return

if __name__ == '__main__':
    copy_mask_files('log1')
    copy_mask_files('log2')
    copy_mask_files('log3')
    copy_mask_files('log4')

    copy_image_files('log1')
    copy_image_files('log2')
    copy_image_files('log3')
    copy_image_files('log4')
    sys.exit(0)

