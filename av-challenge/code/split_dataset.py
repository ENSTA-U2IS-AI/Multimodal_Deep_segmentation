import os, random, shutil

def movefile(org_path, org_label_path, target_path, target_label_path, n):
    file_name = os.listdir(org_path)
    file_number = len(file_name)
    target_name = random.sample(file_name, n)
    print(target_name)
    for target in target_name:
        test_Label_name = target.split('leftImg8bit')[0] + 'leftLabel.png'
        shutil.move(org_path + '/' + target, target_path + '/')
        shutil.move(org_label_path + '/' + test_Label_name, target_label_path + '/')

if __name__ == "__main__":
    org_path = '../train/leftImg8bit'
    org_label_path = '../train/leftLabel'
    target_path = '../val/leftImg8bit'
    target_label_path = '../val/leftLabel'
    movefile(org_path, org_label_path, target_path, target_label_path, n=3)
