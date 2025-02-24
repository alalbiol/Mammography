import subprocess
import os

libreta="train_mamo"

if os.path.exists(libreta + ".py"):
    os.remove(libreta + ".py")

subprocess.run(["jupyter","nbconvert",libreta+".ipynb", "--to","script"])
subprocess.run(["python",libreta+".py"])