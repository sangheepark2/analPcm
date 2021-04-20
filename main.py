# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# from multiprocessing.queues import Queue
import queue
import multiprocessing as mp

import numpy as np
import pylab as pl
import soundfile as sf
import tkinter as tk
from tkinter.ttk import *
import getpass
import os
import threading as th
import time
import matplotlib
from matplotlib.offsetbox import AnchoredText

top = tk.Tk()
duration = 0
def moving_average(a, n=3, samplerate=44100):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]

    # moving average
    #MIN_VAL = 0.2
    MIN_VAL = 0.1
    ret = ret[n - 1:] / n
    ret[ret < MIN_VAL] = 0

    # set to zeros after first non-zero value
    sec = 0.1  # second
    samples = int(0.1 * samplerate)
    count = 0
    pos = np.arange(60 * 60 * 2)  # 60 second * 60 min * 2 hours
    for i in range(len(ret)):
        if ret[i] >= MIN_VAL:
            pos[count] = i

            ret[i + 1:i + samples] = 0
            i += samples
            count += 1

    return count, pos


def getDriftProc(fileName):
    print(fileName)
    READ_UNIT = 5

    sample_rate = int(entry_samplerate.get())
    os.path.getsize(fileName)
    statinfo = os.stat(fileName)
    filesize = statinfo.st_size
    duration = filesize / (2 * 2 *sample_rate) # bit depth (2) * channels (2) * samplerate

    readunit = sample_rate * READ_UNIT * 60
    readstart = 0
    result = None
    print("readUnit: %d" % readunit)
    break_flag = True
    while break_flag:
        print(".")
        data, sample_rate = sf.read(fileName, start=readstart, stop=readstart + readunit, channels=2,
                                    samplerate=sample_rate, format='RAW', subtype='PCM_16', endian='LITTLE')
        readstart += readunit

        if len(data) != readunit:
            break_flag = False

        lcount, lPos = moving_average(data[:, 0])
        rcount, rPos = moving_average(data[:, 1])
        lPos = lPos[0:lcount]
        rPos = rPos[0:rcount]

        maxOffset = 0.2  # max offset: 300 ms
        diff = np.zeros(len(data[:, ]))

        lIdx = 0
        rIdx = 0
        print("lcound: %d rcount: %d" % (lcount, rcount))
        while lIdx < lcount and rIdx < rcount:
            if abs(lPos[lIdx] - rPos[rIdx]) < maxOffset * sample_rate:
                #print( "hahhahaha - 1" )
                diff[lPos[lIdx]] = (lPos[lIdx] - rPos[rIdx]) / sample_rate
                result = np.append(result, diff[lPos[lIdx]])
                if diff[lPos[lIdx]] == 0:
                    print("diff == 0 lDix: %d rIdx: %d" % (lIdx, rIdx))
                lIdx += 1
                rIdx += 1
            elif lPos[lIdx] - rPos[rIdx] > maxOffset * sample_rate:
                #print("hahhahaha - 2")
                rIdx += 1
            elif lPos[lIdx] - rPos[rIdx] < -maxOffset * sample_rate:
                #print("hahhahaha - 3")
                lIdx += 1
            else:
                #print("hahhahaha - 4")
                diff[lPos[lIdx]] = (lPos[lIdx] - rPos[rIdx]) / sample_rate
                result = np.append(result, diff[lPos[lIdx]])
                lIdx += 1
                rIdx += 1

    #print("end of getDriftProc result type: %s\n" % type(result));
    #print("end of getDriftProc result.size(): %d\n" % result.size());
    return result


def cbExit():
    top.destroy()
    exit


def browseFiles():
    user = getpass.getuser()
    fileName = tk.filedialog.askopenfilename(initialdir=os.getcwd(),
                                             title="Select a File",
                                             filetypes=(("all files",
                                                         "*.*"),
                                                        ("Text files",
                                                         "*.txt*")))

    # Change label contents
    label_filename.configure(text="File Opened: " + fileName)

    samplerate = entry_samplerate.get()

    print("browseFiles: %s" % fileName)
    return fileName, samplerate


def showProgress(q):
    val = 0
    while True:
        try:
            item = q.get(True, 0)
            if item == "quit":
                break
        except:
            pass

        val += 10
        val = val % 101
        progress['value'] = val
        top.update_idletasks()
        time.sleep(0.5)


def resetProgress():
    progress['value'] = 0
    top.update_idletasks()


class ThGetDrift(th.Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        th.Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self, *args):
        th.Thread.join(self, *args)
        return self._return


def cbGetDrift():
    filename, samplerate = browseFiles()
    samplerate = int(samplerate)

    filename_queue.put(filename)


def update_figure(window):
    window.after(10, update_figure, window)

class ThDrift(th.Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        th.Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self, *args):
        th.Thread.join(self, *args)
        return self._return


def getDriftDaemon(return_q):
    while True:
        try:
            filename = filename_queue.get(True, 0)
            print("filename in getDriftDaemon: %s" % filename)
            if filename == "quit":
                break
            else:
                t2 = th.Thread(target=showProgress, args=(prog_q,))
                t2.start()

                t1 = ThGetDrift(target=getDriftProc, args=(filename,))
                t1.start()
                result = t1.join()
                return_q.put(result)
                prog_q.put("quit")
                resetProgress()

        except: # mp.Queue.Empty:
            pass


def update_plot(return_q):
    try:
        data = return_q.get(False)

        if data is not None:
            pl.title("Drift over Time")
            #pl.title("Sync")
            pl.xlabel("Time (Minutes)")
            #pl.xlabel("test samples")
            #pl.ylabel("Sync (Second)")

            data2 = data[data != np.array(None)]
            var = np.var(data2)
            var = "var: %.6f" % var
            mean = np.mean(data2)
            mean = "mean: %.4f" % mean
            range = np.max(data2) - np.min(data2)
            range = "range: %.4f" % range
            min = "min: %.3f" % np.min(data2)
            max = "max: %.3f" % np.max(data2)
            median = "median: %.3f" % np.median(data2)
            text_info = mean + " " + range + " " + var + " " + max + " " + min + " " + median
            #print(text_info)

            #click = int(entry_click.get())
            click = float(entry_click.get())
            xlen = data2.shape[0]
            x = np.arange(0, xlen)
            x = x * click / 60
            #pl.plot(x, data2.tolist(), label="Drift over Time")
            pl.plot(x, data2.tolist(), marker='o', markersize=3, label="Drift over Time")
            pl.text(0.1, 0.5, text_info, bbox=dict(boxstyle="square", facecolor="white"))
            text2 = AnchoredText(text_info, loc=2)
            ax = pl.gca()
            ax.add_artist(text2)

            label_var_out.configure(text=var)
            label_mean_out.configure(text=mean)

            pl.show()
        else:
            print("\nupdate_plot-------------10\n");
            None
    except queue.Empty:
        #print("\nupdate_plot-------------100\n");
        pass
    finally:
        top.after(500, update_plot, return_q)


if __name__ == '__main__':
    global send_queue, return_queue, prog_q
    return_queue = mp.Queue()
    filename_queue = mp.Queue()
    prog_q = mp.Queue()
    getDriftTh = th.Thread(target=getDriftDaemon, args=(return_queue,))
    getDriftTh.setDaemon(True)
    getDriftTh.start()

    update_plot(return_queue)

    top.geometry("250x180")

    rows = 0
    frame_1 = tk.Frame(top, width=100)
    frame_1.grid(stick='w', column=0, row=0)
    label_filename = tk.Label(frame_1, text="File Name...", height=1, fg="blue")
    # Grid method is chosen for placing
    # the widgets at respective positions
    # in a table like structure by
    # specifying rows and columns
    label_filename.pack(side=tk.LEFT)
    rows += 1

    frame_2 = tk.Frame(top, width=100)
    frame_2.grid(stick='w', column=0, row=rows)
    label_samplerate = tk.Label(frame_2, text="Sample Rate: ", height=1, fg="black")
    label_samplerate.pack(side=tk.LEFT)
    text_entry = tk.StringVar()
    text_entry.set("44100")
    entry_samplerate = tk.Entry(frame_2, width=10, textvariable=text_entry)
    entry_samplerate.pack(side=tk.LEFT)
    label_samplerate_unit = tk.Label(frame_2, text=" Hz", height=1, fg="black")
    label_samplerate_unit.pack(side=tk.LEFT)
    rows += 1

    frame_3 = tk.Frame(top, width=100)
    frame_3.grid(stick='w', column=0, row=rows)
    label_click = tk.Label(frame_3, text="Click Term: ", height=1, fg="black")
    label_click.pack(side=tk.LEFT)
    frame_click = Frame(frame_3, width=50, height=1)
    frame_click.pack(side=tk.LEFT)
    text_click = tk.StringVar()
    text_click.set("0.5")
    entry_click = tk.Entry(frame_click, width=10, textvariable=text_click)
    entry_click.pack(side=tk.LEFT)
    rows += 1

    label_click_unit = tk.Label(frame_3, text=" Second", height=1, fg="black")
    label_click_unit.pack(side=tk.LEFT)
    frame_4 = tk.Frame(top, width=100)
    frame_4.grid(stick='w', column=0, row=rows)
    btnGetDrift = tk.Button(frame_4, text="getDrift", command=cbGetDrift)
    btnGetDrift.pack(side=tk.LEFT)
    btnExit = tk.Button(frame_4, text="Exit", command=exit)
    btnExit.pack(side=tk.LEFT)
    rows += 1

    frame_5 = tk.Frame(top, width=100)
    frame_5.grid(stick='w', column=0, row=rows)
    progress = Progressbar(frame_5, orient='horizontal',
                           length=200, mode='determinate')
    progress.pack(side=tk.LEFT)
    rows += 1

    frame_output0 = tk.Frame(top)
    frame_output0.grid(stick='w', column=0, row=rows)
    label_output0 = tk.Label(frame_output0, text="Outputs", height=1, fg="black")
    label_output0.pack(side=tk.LEFT)
    rows += 1

    frame_output1 = tk.Frame(top)
    frame_output1.grid(stick='w', column=0, row=rows)
    label_var = tk.Label(frame_output1, text="Variance: ", height=1, fg="black")
    label_var.pack(side=tk.LEFT)
    label_var_out = tk.Label(frame_output1, text="to be calculated ", height=1, fg="black")
    label_var_out.pack(side=tk.LEFT)
    rows += 1

    frame_output2 = tk.Frame(top, width=50)
    frame_output2.grid(stick='w', column=0, row=rows)
    label_mean = tk.Label(frame_output2, text="Mean     : ", height=1, fg="black")
    label_mean.pack(side=tk.LEFT)
    label_mean_out = tk.Label(frame_output2, text="to be calculated ", height=1, fg="black")
    label_mean_out.pack(side=tk.LEFT)

    top.mainloop()
