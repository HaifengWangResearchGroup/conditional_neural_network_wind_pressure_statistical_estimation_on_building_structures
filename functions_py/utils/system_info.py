import os


def function_chdir_to_project_root():
    if "PROJECT_ROOT" in os.environ:
        os.chdir(os.environ["PROJECT_ROOT"])
    else:
        if "functions_py" not in os.listdir('.'):
            path_abs = os.path.abspath('.')
            list_search = ["functions_py", "data"]
            list_match = [path_abs.find(name_dir) for name_dir in list_search]
            temp_index = max(list_match)
            os.chdir(path_abs[:temp_index])

        str_error = "cannot find correct path to project root"
    assert "functions_py" in os.listdir('.'), str_error
    return None
