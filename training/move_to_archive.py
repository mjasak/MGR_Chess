import os
import shutil

run_number = 12
run_dir = r'C://Users//jasak//PycharmProjects/pythonProject//run//archive//chess'
run_dir = run_dir + '//run' + str(run_number).zfill(4)

if not os.path.exists(run_dir):
    os.mkdir(run_dir)
    os.mkdir(run_dir + "//models")
    os.mkdir(run_dir + "//logs")

src_dir = r'C://Users//jasak//PycharmProjects/pythonProject//run'

for item in os.listdir(src_dir + "//logs"):
    shutil.move(src_dir + "//logs//" + item, run_dir + "//logs//" + item)

for item in os.listdir(src_dir + "//models"):
    shutil.move(src_dir + "//models//" + item, run_dir + "//models//" + item)

shutil.move(src_dir + "//config.py", run_dir + "//config.py")

# shutil.move(src_dir + "//plot//plot.jpg", run_dir + "//config.py")