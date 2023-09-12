import subprocess

def run_shell(cmd_str, no_output=False):
    """Function to run shell/bash command in a safe manner
    """
    cmd_lst = cmd_str.split(' ')

    if no_output:
        subprocess.run(cmd_lst, check=True, text=True, stdout=subprocess.DEVNULL)
    else:
        subprocess.run(cmd_lst, check=True, text=True)