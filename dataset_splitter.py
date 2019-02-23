from dataset import Tobacco
import os
from shutil import copy


CLASSES = ("ADVE", "Email", "Form", "Letter", "Memo", "News", "Note", "Report", "Resume", "Scientific")

ROOT = "splits/"

def check_and_make_dir(set_name, classes):
    dirs = list(map(lambda x: ROOT + set_name + "/" + x, classes))
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)

t = Tobacco("datasets/Tobacco/", num_splits=1)


phases = ['train', 'val', 'test']
for phase in phases:
    check_and_make_dir(phase, CLASSES)
    pass

for phase in phases:
    dir_path = ROOT + phase + "/"
    for i in t.splits[0][phase]:
        dest = dir_path + CLASSES[i[1]] + "/"
        copy(i[0], dest)
        print("copied %s to %s" % (i[0], dest))
        