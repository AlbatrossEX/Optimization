import os
folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Logs")
for filename in os.listdir(folder):
    if not filename.endswith(".txt"):
        continue
    filepath = os.path.join(folder, filename)
    if os.path.isfile(filepath):
        os.remove(filepath)
