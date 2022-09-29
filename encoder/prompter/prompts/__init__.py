import os

prompts_directory = os.path.dirname(os.path.abspath(__file__))
prompts = {}
for file_name in os.listdir(prompts_directory):
    file_path = os.path.join(prompts_directory, file_name)
    if os.path.isfile(file_path) and file_name.endswith(".txt"):
        with open(file_path, "r") as file:
            prompts[file_name.split(".txt")[0]] = file.read()
