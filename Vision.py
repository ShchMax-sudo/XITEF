import tkinter as tk
from tkinter import filedialog
from tkinter.messagebox import showinfo
import traceback
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

class Vision:
    events = None
    zeroTime = 0
    duration = 0
    bincount = 0
    picPsf = 0
    localToPic = None
    tkroot = None
    imgcanvas = None
    fig = None
    timedEvents = None
    timeEntry1 = None
    timeEntry2 = None
    mode = "Image"
    padX = 10
    padY = 10
    timeChanged = False
    brightnessEntry = None
    xEntry = None
    yEntry = None
    binNumEntry = None
    transients = None
    timedTransients = None
    transientBegin = None
    transientEnd = None
    transientsVerdict = None
    activeTransients = None
    leftLabel = None

    def __init__(self, events, bincount, picPsf, localToPic):
        self.events = events
        self.timedEvents = events
        self.zeroTime = np.min(events[:, 2])
        self.duration = np.max(events[:, 2]) - self.zeroTime
        self.bincount = bincount
        self.picPsf = picPsf
        self.localToPic = localToPic
        self.activeTransients = []

    def addAuxiliary(self, transients):
        self.transients = transients

    def addTransients(self, transients):
        self.transients = transients
        self.transientsVerdict = []
        self.updateTransient()

    def showEvents(self, photons=None, gamma=0.7, xys=None, redSquareSize=1, plot=plt, imageOnly=False):
        if photons is None:
            photons = self.events
        if xys is None:
            xys = []
        pixels = np.zeros((self.bincount, self.bincount), dtype=float)
        for event in photons:
            x, y = self.localToPic(event)
            pixels[x, y] += 1
        xmin, xmax = self.bincount, 0
        ymin, ymax = self.bincount, 0
        for i in range(self.bincount):
            for j in range(self.bincount):
                if pixels[i, j] != 0:
                    xmin = min(xmin, i)
                    xmax = max(xmax, i)
                    ymin = min(ymin, j)
                    ymax = max(ymax, j)
                pixels[i, j] = (pixels[i, j] ** (1 - gamma))
        if imageOnly:
            plot.set_axis_off()
            pixels = pixels[xmin:xmax + 1, ymin:ymax + 1]
        plot.imshow(pixels, cmap="Greys")
        pixels = np.zeros((self.bincount, self.bincount), dtype=float)
        Psf = round(self.picPsf * redSquareSize)
        for xy in xys:
            x, y = self.localToPic(xy)
            for k in range(-Psf, Psf + 1):
                pixels[x - Psf, y + k] = 1
                pixels[x + Psf, y + k] = 1
                pixels[x + k, y - Psf] = 1
                pixels[x + k, y + Psf] = 1
        if imageOnly:
            pixels = pixels[xmin:xmax + 1, ymin:ymax + 1]
        if xys:
            plot.imshow(pixels, alpha=pixels, cmap="Reds")
        if plot == plt:
            plot.show()

    def timeTrim(self, begin, end):
        if not self.timeChanged:
            return
        if begin > end:
            begin, end = end, begin
        num = 0
        for event in self.events:
            if begin <= event[2] - self.zeroTime <= end:
                num += 1
        shape = (num, self.events.shape[1])
        self.timedEvents = np.zeros(shape)
        num = 0
        for event in self.events:
            if begin <= event[2] - self.zeroTime <= end:
                self.timedEvents[num] = event
                self.timedEvents[num, 2] -= self.zeroTime
                num += 1
        self.timedTransients = []
        if self.timeEntry2 is not None:
            for event in self.transients:
                if begin <= event[2] - self.zeroTime <= end:
                    self.timedTransients.append(event[:2] + tuple([event[2] - self.zeroTime]) + event[3:])
        self.timeChanged = False

    def firstTimeTrim(self):
        begin = self.checkRange(self.timeEntry1, 0, self.duration)
        end = self.checkRange(self.timeEntry2, 0, self.duration)
        self.timeTrim(begin, end)

    def secondTimeTrim(self):
        padding = self.checkRange(self.timeEntry1, 0, self.duration)
        begin = max(0.0, self.transientBegin - padding)
        end = min(self.duration, self.transientEnd + padding)
        self.timeTrim(begin, end)

    def updateImage(self):
        self.mode = "Image"

    def updateHist(self):
        self.mode = "Hist"

    def updateScatter(self):
        self.mode = "Scatter"

    def updatePlot(self):
        self.mode = "Plot"

    def getFormula(self, form):
        if form.capitalize() == "Time":
            return self.timedEvents[:, 2]
        replaces = {
            "events": "self.timedEvents",
            "transients": "self.timedTransients",
        }
        for repl in replaces:
            form = form.replace(repl, replaces[repl])
        return eval(form)

    @staticmethod
    def checkRange(entry, minv, maxv):
        val = float(entry.get())
        if val < minv:
            val = minv
            entry.delete(0, tk.END)
            entry.insert(0, str(minv))
        elif val > maxv:
            val = maxv
            entry.delete(0, tk.END)
            entry.insert(0, str(maxv))
        return val

    def update(self):
        try:
            self.updateBody()
        except Exception:
            showinfo(title="Error", message=traceback.format_exc())

    def updateBody(self):
        if self.timeEntry2 is None:
            self.secondTimeTrim()
        else:
            self.firstTimeTrim()
        self.fig.clear()
        plot = self.fig.add_subplot()
        self.fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        if self.mode == "Image":
            self.showEvents(photons=self.timedEvents,
                            gamma=self.checkRange(self.brightnessEntry, 0, 1),
                            plot=plot, xys=self.activeTransients, redSquareSize=3,
                            imageOnly=(self.timeEntry2 is None))
        elif self.mode == "Hist":
            binNum = self.checkRange(self.binNumEntry, 0, self.duration)
            args = {}
            if binNum != 0:
                args["bins"] = int(binNum)
            if self.yEntry.get().capitalize() == "Entire":
                tmin = self.checkRange(self.timeEntry1, 0, self.duration)
                tmax = self.checkRange(self.timeEntry2, 0, self.duration)
                args["range"] = (min(tmin, tmax), max(tmin, tmax))
            plot.hist(self.getFormula(self.xEntry.get()), **args)
        elif self.mode == "Scatter":
            plot.scatter(self.getFormula(self.xEntry.get()), self.getFormula(self.yEntry.get()), marker=".")
        elif self.mode == "Plot":
            plot.plot(self.getFormula(self.xEntry.get()), self.getFormula(self.yEntry.get()), marker=".")
        self.imgcanvas.draw()
        if self.leftLabel is not None:
            self.leftLabel["text"] = f"{len(self.transientsVerdict)} / {len(self.transients)}"

    def save(self):
        filepath = filedialog.asksaveasfile(defaultextension="png", filetypes=[("PNG", "*.png")], initialdir=".")
        self.fig.savefig(filepath.name)

    def acceptTransient(self):
        self.transientsVerdict.append("Accept")
        if self.updateTransient():
            return
        self.update()

    def rejectTransient(self):
        self.transientsVerdict.append("Reject")
        if self.updateTransient():
            return
        self.update()

    def unsureTransient(self):
        self.transientsVerdict.append("Unsure")
        if self.updateTransient():
            return
        self.update()

    def updateTransient(self):
        i = len(self.transientsVerdict)
        if i == len(self.transients):
            self.tkroot.destroy()
            return True
        tmean = self.transients[i]["tmean"] - self.zeroTime
        tdev = self.transients[i]["tdev"]
        x = self.transients[i]["x"]
        y = self.transients[i]["y"]
        self.transientBegin = tmean - tdev
        self.transientEnd = tmean + tdev
        self.activeTransients = [(x, y)]
        self.timeChanged = True
        return False

    def initGUI(self, transientsMode=False):
        self.tkroot = tk.Tk()
        self.tkroot.title("Vision")
        self.tkroot.geometry("700x800")
        self.tkroot.resizable(False, False)

        self.fig = Figure(figsize=(5, 5), dpi=100)
        self.imgcanvas = FigureCanvasTkAgg(self.fig, master=self.tkroot)
        toolbar = NavigationToolbar2Tk(self.imgcanvas, self.tkroot, pack_toolbar=False)
        toolbar.update()
        self.imgcanvas.get_tk_widget().pack(padx=self.padX, pady=self.padY)

        if not transientsMode:
            modesFrame = tk.Frame(self.tkroot)

            imageButton = tk.Button(modesFrame, text="Image", command=self.updateImage)
            imageButton.pack(side=tk.LEFT)
            histButton = tk.Button(modesFrame, text="Histogram", command=self.updateHist)
            histButton.pack(side=tk.LEFT)
            scatterButton = tk.Button(modesFrame, text="Scatter", command=self.updateScatter)
            scatterButton.pack(side=tk.LEFT)
            plotButton = tk.Button(modesFrame, text="Plot", command=self.updatePlot)
            plotButton.pack(side=tk.LEFT)

            modesFrame.pack()

        def isnum(s):
            if s.replace(".", "1", 1).isnumeric():
                self.timeChanged = True
                return True
            return False
        check = (self.tkroot.register(isnum), "%P")

        if not transientsMode:
            timesFrame = tk.Frame(self.tkroot)

            timeLabel1 = tk.Label(timesFrame, text="Min time:")
            timeLabel1.pack(side=tk.LEFT, pady=self.padY)
            self.timeEntry1 = tk.Entry(timesFrame, validate="key", validatecommand=check)
            self.timeEntry1.insert(0, "0")
            self.timeEntry1.pack(side=tk.LEFT, padx=self.padX, pady=self.padY)
            timeLabel2 = tk.Label(timesFrame, text="Max time:")
            timeLabel2.pack(side=tk.LEFT, pady=self.padY)
            self.timeEntry2 = tk.Entry(timesFrame, validate="key", validatecommand=check)
            self.timeEntry2.insert(0, str(self.duration))
            self.timeEntry2.pack(side=tk.LEFT, padx=self.padX, pady=self.padY)

            timesFrame.pack()

        if transientsMode:
            timesFrame = tk.Frame(self.tkroot)

            timeLabel1 = tk.Label(timesFrame, text="Padding time:")
            timeLabel1.pack(side=tk.LEFT, pady=self.padY)
            self.timeEntry1 = tk.Entry(timesFrame, validate="key", validatecommand=check)
            self.timeEntry1.insert(0, "0")
            self.timeEntry1.pack(side=tk.LEFT, padx=self.padX, pady=self.padY)

            updateButton = tk.Button(timesFrame, text="Update", command=self.update)
            updateButton.pack(side=tk.LEFT, padx=self.padX)

            timesFrame.pack()

        parametersFrame = tk.Frame(self.tkroot)

        brightnessLabel = tk.Label(parametersFrame, text="Brightness:")
        brightnessLabel.pack(side=tk.LEFT)
        self.brightnessEntry = tk.Entry(parametersFrame, validate="key", validatecommand=check)
        self.brightnessEntry.insert(0, "0.7")
        self.brightnessEntry.pack(side=tk.LEFT, padx=self.padX)
        if not transientsMode:
            binNumLabel = tk.Label(parametersFrame, text="Bin number:")
            binNumLabel.pack(side=tk.LEFT)
            self.binNumEntry = tk.Entry(parametersFrame, validate="key", validatecommand=check)
            self.binNumEntry.insert(0, "0")
            self.binNumEntry.pack(side=tk.LEFT, padx=self.padX)

        parametersFrame.pack()

        if not transientsMode:
            valuesFrame = tk.Frame(self.tkroot)

            xLabel = tk.Label(valuesFrame, text="X axis:")
            xLabel.pack(side=tk.LEFT, pady=self.padY)
            self.xEntry = tk.Entry(valuesFrame)
            self.xEntry.pack(side=tk.LEFT, padx=self.padX, pady=self.padY)
            yLabel = tk.Label(valuesFrame, text="Y axis:")
            yLabel.pack(side=tk.LEFT, pady=self.padY)
            self.yEntry = tk.Entry(valuesFrame)
            self.yEntry.pack(side=tk.LEFT, padx=self.padX, pady=self.padY)

            valuesFrame.pack()

            finalFrame = tk.Frame(self.tkroot)

            renderButton = tk.Button(finalFrame, text="Render", command=self.update)
            renderButton.pack(side=tk.LEFT, padx=self.padX)
            saveButton = tk.Button(finalFrame, text="Save", command=self.save)
            saveButton.pack(side=tk.LEFT, padx=self.padX)

            finalFrame.pack()

        if transientsMode:
            fateFrame = tk.Frame(self.tkroot)
            acceptButton = tk.Button(fateFrame, text="Accept", command=self.acceptTransient)
            acceptButton.pack(side=tk.LEFT, padx=self.padX)
            rejectButton = tk.Button(fateFrame, text="Reject", command=self.rejectTransient)
            rejectButton.pack(side=tk.LEFT, padx=self.padX)
            unsureButton = tk.Button(fateFrame, text="Unsure", command=self.unsureTransient)
            unsureButton.pack(side=tk.LEFT, padx=self.padX)
            fateFrame.pack()

            leftFrame = tk.Frame(self.tkroot)
            self.leftLabel = tk.Label(leftFrame, text="")
            self.leftLabel.pack(side=tk.LEFT)
            leftFrame.pack()
            self.update()

        self.tkroot.mainloop()
