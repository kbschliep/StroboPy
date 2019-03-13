## \strobopy\strobopy
'''Getting data directory so data can easily be imported '''

import sys, os # python modules for interacting with computer
from tkinter import filedialog # interface for opening a file explorer
from tkinter import * # * imports all files in tkinter
import glob

def get_dm3(new=True):
    '''Opens a window to select directory if new==False gets current directory
    Returns the directory as a string
    '''
    if new==True:
        root = Tk() # Tk() is a function in tkinter that opens a window
        root.directory = filedialog.askdirectory() # opens explorer window so you can find the folder of choice
        root.withdraw() # closes the tkinter window since it's unnecessary
        folder_path=root.directory
        # oldcwd = os.getcwd() # saves old called working directory (place where data is drawn from) as oldcwd use os.chdir(oldcwd) to go back
        # os.chdir(root.directory) # sets new directory
        # newcwd = os.getcwd() # saves new directory name as newcwd
        # return root.directory
    if new==False:
        folder_path=os.getcwd()
        # return cd
    path = folder_path + '/'+'*.dm3' # Change '' as needed
    file_list=sorted(glob.glob(path))
    return file_list
