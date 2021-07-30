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

filenameStat = ""

top = tk.Tk()
duration = 0
def moving_average(inputPcm, samples=3, samplerate=44100):
    ms = 3  # 2 ms
    absInput = np.abs(inputPcm)
    samples = int(ms * samplerate / 1000)
    #tmpAverage = np.cumsum(absInput, dtype=float)
    #tmpAverage[samples:] = tmpAverage[samples:] - tmpAverage[:-samples]

    # moving average: average over next specific time frame (ex. 5 ms)
    tmp = np.flipud(absInput).cumsum()
    tmp[samples:] = tmp[samples:] - tmp[:-samples]
    ret = np.flipud(tmp)

    #ret = np.abs(inputPcm)
    #ret = np.abs(tmpAverage)
    MIN_VAL = float(entry_minval.get())
    MIN_VAL = MIN_VAL * samples
    #print("===========================================")
    #print("type of ret: %s" % type(ret))
    #print("size of ret: %s" % ret.size)
    ret[ret < MIN_VAL] = 0

    # set to zeros after first non-zero value
    #reset_duration = 0.1  # second
    reset_duration = float(entry_resetval.get())  # second
    samples = int(reset_duration * samplerate)
    count = 0
    pos = np.arange(60 * 60 * 2)  # 60 second * 60 min * 2 hours

    ms = 0.5
    non_zero_samples_len = int(ms * samplerate / 1000)
    for i in range(len(ret)):
        #print(ret[i:i+non_zero_samples_len])
        tmp = ret[i] >= MIN_VAL
        #if np.count_nonzero(ret[i:non_zero_samples_len + i]) > non_zero_samples_len * 0.8:
            #print("first cond: %d , non zero count: %d" % (tmp, np.count_nonzero(ret[i:non_zero_samples_len + i] ) )
            #print("first cond: %d non zero count: %d" % ( tmp, np.count_nonzero(ret[i:non_zero_samples_len + i]) ) )

        #if tmp == 1:
        #if tmp == 1 and np.count_nonzero(ret[i:non_zero_samples_len + i]) >= non_zero_samples_len * 0.5:
        #    print("tmp = %d,  %d,   %d -> %d" % (tmp, np.count_nonzero(ret[i:non_zero_samples_len + i]), non_zero_samples_len * 0.5, np.count_nonzero(ret[i:non_zero_samples_len + i]) > non_zero_samples_len * 0.5 ))

        if tmp and (np.count_nonzero(ret[i:non_zero_samples_len + i]) > int(non_zero_samples_len * 0.5) ):
        #if ret[i] >= MIN_VAL:
        #    print("non zero count: %d" % np.count_nonzero(ret[i:non_zero_samples_len + i]))
            pos[count] = i

            ret[i + 1:i + samples] = 0
            i += samples
            count += 1

    return count, pos

def moving_average_test(inputPcm, samples=3, samplerate=44100):
    absInput = np.abs(inputPcm)
    tmpAverage = np.cumsum(absInput, dtype=float)
    tmpAverage[samples:] = tmpAverage[samples:] - tmpAverage[:-samples]

    # moving average
    #MIN_VAL = 0.2
    #MIN_VAL = 0.1
    ret = tmpAverage[samples - 1:] / samples

    ret = np.abs(inputPcm)
    MIN_VAL = float(entry_minval.get())
    #print("===========================================")
    #print("type of ret: %s" % type(ret))
    #print("size of ret: %s" % ret.size)
    ret[ret < MIN_VAL] = 0

    # set to zeros after first non-zero value
    #reset_duration = 0.1  # second
    reset_duration = float(entry_resetval.get())  # second
    samples = int(reset_duration * samplerate)
    count = 0
    pos = np.arange(60 * 60 * 2)  # 60 second * 60 min * 2 hours

    ms = 0.5
    non_zero_samples_len = int(ms * samplerate / 1000)
    print("non_zero_samples_len = %d" % non_zero_samples_len)
    for i in range(len(ret)):
        #print(ret[i:i+non_zero_samples_len])
        #if np.count_nonzero(ret[i:non_zero_samples_len + i]) > non_zero_samples_len * 0.8:
        #    print("non zero count: %d" % np.count_nonzero(ret[i:non_zero_samples_len + i]))

        #if ret[i] >= MIN_VAL and np.count_nonzero(ret[i:non_zero_samples_len + 1]) > non_zero_samples_len * 0.5:
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

    filenameStat = fileName + ".txt"
    print( "filenameStat: %s" % filenameStat )
    fileStat = open(filenameStat, "wt")

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
        print("lcount: %d rcount: %d" % (lcount, rcount))
        while lIdx < lcount and rIdx < rcount:
            if abs(lPos[lIdx] - rPos[rIdx]) < maxOffset * sample_rate:
                #print( "hahhahaha - 1" )
                diff[lPos[lIdx]] = (lPos[lIdx] - rPos[rIdx]) / sample_rate
                result = np.append(result, diff[lPos[lIdx]])
                if diff[lPos[lIdx]] == 0:
                    print("diff == 0 lDix: %d rIdx: %d" % (lIdx, rIdx))
                else:
                    strtmp = "%.3f" % diff[lPos[lIdx]]
                    fileStat.write(strtmp + "\n")
                lIdx += 1
                rIdx += 1
            elif lPos[lIdx] - rPos[rIdx] > maxOffset * sample_rate:
                rIdx += 1
            elif lPos[lIdx] - rPos[rIdx] < -maxOffset * sample_rate:
                lIdx += 1
            else:
                diff[lPos[lIdx]] = (lPos[lIdx] - rPos[rIdx]) / sample_rate
                result = np.append(result, diff[lPos[lIdx]])
                strtmp = "%.3f" % diff[lPos[lIdx]]
                fileStat.write(strtmp + "\n")
                lIdx += 1
                rIdx += 1

    #print("end of getDriftProc result type: %s\n" % type(result));
    #print("end of getDriftProc result.size(): %d\n" % result.size());

    fileStat.close()

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
            None
    except queue.Empty:
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

    top.geometry("250x240")

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
    label_click_unit = tk.Label(frame_3, text=" Second", height=1, fg="black")
    label_click_unit.pack(side=tk.LEFT)
    rows += 1

    frame_min = tk.Frame(top, width=100)
    frame_min.grid(stick='w', column=0, row=rows)
    label_minval = tk.Label(frame_min, text="Min Val: ", height=1, fg="black")
    label_minval.pack(side=tk.LEFT)
    frame_minval = Frame(frame_min, width=50, height=1)
    frame_minval.pack(side=tk.LEFT)
    text_minval = tk.StringVar()
    text_minval.set("0.07")
    entry_minval = tk.Entry(frame_minval, width=5, textvariable=text_minval)
    entry_minval.pack(side=tk.LEFT)
    label_minval_range = tk.Label(frame_minval, text="(0 ~ 1)", height=1, fg="black")
    label_minval_range.pack(side=tk.LEFT)
    rows += 1

    frame_reset = tk.Frame(top, width=100)
    frame_reset.grid(stick='w', column=0, row=rows)
    label_reset = tk.Label(frame_reset, text="Reset Duration: ", height=1, fg="black")
    label_reset.pack(side=tk.LEFT)
    frame_resetval = Frame(frame_reset, width=20, height=1)
    frame_resetval.pack(side=tk.LEFT)
    text_resetval = tk.StringVar()
    text_resetval.set("0.4")
    entry_resetval = tk.Entry(frame_resetval, width=5, textvariable=text_resetval)
    entry_resetval.pack(side=tk.LEFT)
    label_reset_unit = tk.Label(frame_reset, text=" Second", height=1, fg="black")
    label_reset_unit.pack(side=tk.LEFT)
    rows += 1

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
