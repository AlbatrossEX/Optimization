import os
folder = "Logs"
for filename in os.listdir(folder):
    filepath = os.path.join(folder, filename)
    if os.path.isfile(filepath):
        os.remove(filepath)
