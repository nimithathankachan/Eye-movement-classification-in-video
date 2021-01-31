import tkinter
from tkinter import * 
import sys
import os
import tkinter.messagebox
window = tkinter.Tk()
window.configure(background='black')
# to rename the title of the window
window.title("GUI")
window.configure(width=500,height=700)
# pack is used to show the object in the window
label = tkinter.Label(window, text = "EYE MOVEMENT CLASSIFICATION",fg="black",bg="white",font="Times 28 bold").pack()
def ExitApplication():
    MsgBox = tkinter.messagebox.askquestion ("CLASSIFY MOVEMENTS","CAPTURE")
    if MsgBox == 'yes':
       os.system('python3 3fin3.py')
       
    else:
         window.destroy()
        
B = tkinter.Button(window, text ="NEXT", command = ExitApplication)
#B.pack()
B.place(x = 900,y = 100)
window.mainloop()

