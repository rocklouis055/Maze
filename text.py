from tkinter import *
from tkinter import filedialog  #provides classes and functions for creating file
def new_file():
    text.delete(0.0,END)    #to delete all items in the list
def open_file():
    file1=filedialog.askopenfile(mode='r')   #opens a file in reading only mode
    data=file1.read()
    text.delete(0.0,END)
    text.insert(0.0,data)

def save_file():
    filename="Untitled.txt"
    data=text.get(0.0,END)
    file=open(filename,"w")
    file1.write(data)
def save_as():
    file1=filedialog.asksaveasfile(mode='w')    #opens the file in write only mode
    data=text.get(0.0,END)
    file1.write(data)


gui=Tk()
gui.title("NOTEPAD")
gui.geometry("800x1000")

text=Text(gui)
text.pack()
mymenu=Menu(gui)

list1=Menu()
list1.add_command(label='Create File',command=new_file)

list1.add_command(label='Save File',command=save_file)
list1.add_command(label='Save File As',command=save_as)
list1.add_command(label='Open File',command=open_file)
list1.add_command(label='Exit',command=gui.quit)
mymenu.add_cascade(label='Options',menu=list1)


gui.config(menu=mymenu)
gui.mainloop()