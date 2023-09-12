import subprocess

def run_shell(cmd_str, output_to_list=False, delimiter='\n', wait=True):
    """
    Function to run shell/bash command in a safe manner.
    
    Parameters
    ----------
    cmd_str: (str) 
        The shell/bash command to execute
    output_to_list: (bool) 
        Whether to split the output by delimiter and return as a list
    delimiter: (str)
        Delimiter to split the output by (default is newline)
    
    Return
    ------
    The output of the shell/bash command as a string or list, depending on the value of output_to_list
    """    
    try: 
        if wait:
            # Run the command using subprocess.run(), capturing the output
            result = subprocess.run(cmd_str, check=True, text=True, shell=True, capture_output=True).stdout.strip()
        else:
            subprocess.Popen(cmd_str, shell=True, stdin=None, stdout=None, stderr=None, close_fds=True)
            result = None
    except subprocess.CalledProcessError as error:
        if '#000: /Users/vfonov/src/build/minc-toolkit-v2/HDF5/src/H5F.c line 675 in H5Fclose(): closing file ID failed' in error.stderr:
            result = ''
            # This is an error produced by the minc toolkit. However, the output files are fine. Hence, the error is ignored. 
        else:
            print(f'The command "{cmd_str}" failed:')
            print(error.stderr)
            result = ''
            # sys.exit() 
    
    # Split the output by delimiter and return as a list (if output_to_list is True)
    if output_to_list:
        result = result.split(delimiter)
    
    # Return the output as a string or list, depending on the value of output_to_list
    return result