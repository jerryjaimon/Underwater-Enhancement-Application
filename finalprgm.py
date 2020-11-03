from tkinter import *

from AUV import AUV150
from image_enhance import ColorCorrect, enhanceOnly, enhanceColorCorrect
from image_enhancement_color_model import underwater_integrated_color_model
from clahe import clahe_fun
from tkinter import filedialog
from detection import detection_fun
from functools import partial

filename = []
location = "Output/"
var = 0

def browseFiles():
    file = filedialog.askopenfilename(initialdir="/",
                                      title="Select a File",
                                      filetypes=(("jpg files",
                                                  "*.jpg*"),
                                                 ("all files",
                                                  "*.*")))
    filename.append(file)
    label_file_explorer.configure(text="File Opened: " + file)


def colorcorrect(path, location):
    index = path[-1].rindex("/")
    filename = path[-1][index + 1:]
    location = location + "colorcorrected_" + filename
    ColorCorrect(path[-1], location)
    label_output.configure(text="Color Corrected and saved to " + location)


def enhance(path, location):
    print(path[-1])
    index = path[-1].rindex("/")
    filename = path[-1][index + 1:]
    location = location + "enhanced_" + filename
    enhanceOnly(path[-1], location)
    label_output.configure(text="Enhanced and saved to " + location)


def enhanceandcc(path, location):
    index = path[-1].rindex("/")
    filename = path[-1][index + 1:]
    location = location + "enhanced_cc_" + filename
    enhanceColorCorrect(path[-1], location)
    label_output.configure(text="Enhanced and Color Corrected and saved to " + location)


def paper1(path, location):
    index = path[-1].rindex("/")
    filename = path[-1][index + 1:]
    location = location + "paper1_" + filename
    AUV150(path[-1], location)
    label_output.configure(text="Color Corrected[Paper-1] and saved to " + location)


def paper2(path, location):
    index = path[-1].rindex("/")
    filename = path[-1][index + 1:]
    location = location + "paper2_" + filename
    underwater_integrated_color_model(path[-1], location)
    label_output.configure(text="Color Corrected[Paper-2] and saved to " + location)


def clahe(path, location):
    index = path[-1].rindex("/")
    filename = path[-1][index + 1:]
    location = location + "clahe_" + filename
    clahe_fun(path[-1], location)
    label_output.configure(text="CLAHE applied and saved to " + location)

def detection(path, location,enhancement = 0):
    index = path[-1].rindex("/")
    filename = path[-1][index + 1:]
    if enhancement == 1:
        location_in = "Output/enhancedfordetection_" + filename
        enhanceOnly(path[-1],location_in)
        location = location + "detection_enhance" + filename
        detection_fun(location_in, location)
    else:
        location = location + "detection_" + filename
        detection_fun(path[-1], location)
    label_output.configure(text="Detection performed and saved to " + location)

window = Tk()

window.title('Underwater Trash Detection and Enhancement')

window.geometry("500x700")

window.config(background="black")
label_file_explorer = Label(window,
                            text="Underwater Image Detection",
                            width=70, height=4,
                            fg="white", background="black")
label_color = Label(window,
                    text="Color Correction",
                    width=70, height=4,
                    fg="white", background="black")

label_enhancement = Label(window,
                          text="Image Enhancement",
                          width=70, height=4,
                          fg="white", background="black")
label_detection = Label(window,
                        text="Detection",
                        width=70, height=4,
                        fg="white", background="black")
label_output = Label(window,
                     width=70, height=4,
                     fg="white", background="black")

button_selectfile = Button(window,
                           text="Select File",
                           command=browseFiles)

button_detection = Button(window,
                          text="Detection",
                          command=partial(detection, filename, location))

button_detection_enhancement = Button(window,
                                      text="Enhancement and Detection",
                                      command=partial(detection, filename, location,1))
button_colorcorrect = Button(window,
                             text="Color Correction",
                             command=partial(colorcorrect, filename, location))

button_clahe = Button(window,
                      text="CLAHE",
                      command=partial(clahe, filename, location))
button_paperbased1 = Button(window,
                            text="Paper Based-1",
                            command=partial(paper1, filename, location))
button_paperbased2 = Button(window,
                            text="Paper Based-2",
                            command=partial(paper2, filename, location))
button_enhanceonly = Button(window,
                            text="Enhance Only",
                            command=partial(enhance, filename, location))
button_enhance_color_correct = Button(window,
                                      text="Enhance and Color Correct",
                                      command=partial(enhanceandcc, filename, location))

button_exit = Button(window, text="Exit", command=exit)



label_file_explorer.grid(column=1, row=1, sticky=W)
#R1 = Radiobutton(window, text="Image", variable=var, value=1)
#R1.grid(column=1, row=2,pady=1)
#R2 = Radiobutton(window, text="Video", variable=var, value=2)
#R2.grid(column=1, row=3,pady=1)
button_selectfile.grid(column=1, row=4, pady=5)
label_color.grid(column=1, row=5)
button_colorcorrect.grid(column=1, row=6, pady=1)
button_paperbased1.grid(column=1, row=7, pady=1)
button_paperbased2.grid(column=1, row=8, pady=1)
label_enhancement.grid(column=1, row=9)
button_enhanceonly.grid(column=1, row=10, pady=1)
button_enhance_color_correct.grid(column=1, row=11, pady=1)
label_detection.grid(column=1, row=12, pady=1)
button_detection.grid(column=1, row=13, pady=1)
button_detection_enhancement.grid(column=1, row=14, pady=1)
label_output.grid(column=1, row=15)
button_exit.grid(column=1, row=16, pady=3)
window.mainloop()
