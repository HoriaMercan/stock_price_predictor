
def get_directory(filepath):
    # Gets the directory in which the indicated file is found
    index = filepath.find('/')
    if index == -1:
        return ""
    return filepath[0:index+1]