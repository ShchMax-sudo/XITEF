from multiprocessing import current_process, Pool, Manager
from time import time
import sys
import traceback
from copy import deepcopy
import os
from math import log10
from urllib.request import urlretrieve
from tqdm import tqdm
import pandas as pd
import Calc
from Vision import Vision

panic = None
args = {
    "XMMMaster": r"https://nxsa.esac.esa.int/ftp_public/heasarc_obslog/xsaobslog.txt",
    "allowedModes": "PN-FF, PN-EFF, PN-LW, PN-SW, PN-FMK",
    "debug": "n",
    "reviewModes": "Unreviewed, Unsure",
}
form = (lambda x: (f"{x:.6f}") if (type(x) is float) else x)


class Task:
    ID: str
    conf: Calc.Config

    def __init__(self, ID: str, conf: Calc.Config):
        self.ID = ID
        self.conf = conf

    def run(self, filelock, filepath, panic, onSuccess, success, onFail, onException, repeat):
        try:
            cntSuccess = 0
            cntFail = 0
            for _ in range(max(repeat, 1)):
                tf = Calc.TransientFinder(self.ID, self.conf)
                tf.process()
                filelock.acquire()
                if success(tf.result):
                    if repeat == 0:
                        onSuccess(tf.result, filepath, self.ID)
                    else:
                        cntSuccess += 1
                else:
                    onFail(tf.result, filepath, self.ID)
                    if repeat == 0:
                        panic.value = 1
                    else:
                        cntFail += 1
                del tf
                filelock.release()
        except Exception:
            filelock.acquire()
            onException(filepath, self.ID)
            panic.value = 1
            filelock.release()
        if repeat != 0:
            onSuccess(filepath, self.ID, cntSuccess, cntFail,
                      self.conf.ftransientProb, self.conf.ftransientTime)

    @staticmethod
    def processingSuccess(result):
        return result["type"] == "Success"

    @staticmethod
    def processingOnSuccess(result, filepath, ID):
        global form
        with open(filepath, "a") as file:
            print(f"#{ID} {round(result['CodeTime'])}", file=file)
            for transient in result["transients"]:
                transient["prob"] = log10(transient["prob"])
                print(*[form(transient[val]) for val in ["x", "y", "tmean", "tdev",
                                                    "ra", "dec", "timebegin",
                                                    "cntGTI", "backgroundGTI",
                                                    "cntBTI", "backgroundBTI",
                                                    "prob"]], sep="\t", file=file)

    @staticmethod
    def processingOnFail(result, filepath, ID):
        with open(filepath, "a") as file:
            print(f"~{ID} {result['type']}", file=file)

    @staticmethod
    def processingOnException(filepath, ID):
        with open(filepath, "a") as file:
            print(f"!{ID}", file=file)
            print(traceback.format_exc(), file=file)

    @staticmethod
    def fakeSuccess(result):
        return (
            result["type"] == "Success" and result["FakeTransientStatus"] == "Probable"
        )

    @staticmethod
    def fakeOnSuccess(filepath, ID, cntSuccess, cntFail, prob, dur):
        with open(filepath, "a") as file:
            print(
                f"#{ID} probability {prob} duration {dur} sucess rate:" +
                f"{round((cntSuccess / (cntSuccess + cntFail)) * 100)}%",
                file=file,
            )

    @staticmethod
    def fakeOnFail(result=None, filepath=None, ID=None):
        pass

    @staticmethod
    def fakeOnException(filepath, ID):
        with open(filepath, "a") as file:
            print(f"!{ID}", file=file)
            print(traceback.format_exc(), file=file)


def processor(task, filelock, filepath, panic, onSuccess, success, onFail,
              onException, repeat=0):
    global args

    ind = current_process().name
    if not panic.value:
        if args["debug"] == "y":
            print(f"Process {ind} started {task.ID}")
        task.run(filelock, filepath, panic, onSuccess, success, onFail, onException, repeat)
        if args["debug"] == "y":
            print(f"Process {ind} finished {task.ID}")


def processorWrapper(args):
    processor(*args)


def processImages(obsIDs, coreNum, conf, filelock, filepath, panic):
    tasks = []

    for ID in obsIDs:
        tasks.append(Task(ID, conf))

    with Pool(coreNum) as pool:
        list(
            tqdm(
                pool.imap_unordered(
                    processorWrapper,
                    [(task, filelock, filepath, panic, Task.processingOnSuccess,
                      Task.processingSuccess, Task.processingOnFail,
                      Task.processingOnException) for task in tasks],
                ),
                total=len(tasks),
            )
        )


def fakeImages(obsIDs, coreNum, conf, filelock, filepath, panic, lens, pglobs, repeats):
    tasks = []

    for ID in obsIDs:
        for length in lens.split(","):
            for pglob in pglobs.split(","):
                confcurr = deepcopy(conf)
                confcurr.addFakeTransient = True
                confcurr.ftransientTime = float(length)
                confcurr.ftransientProb = float(pglob)
                tasks.append(Task(ID, confcurr))

    with Pool(coreNum) as pool:
        list(
            tqdm(
                pool.imap_unordered(
                    processorWrapper,
                    [(task, filelock, filepath, panic, Task.fakeOnSuccess,
                      Task.fakeSuccess, Task.fakeOnFail, Task.fakeOnException,
                      int(repeats)) for task in tasks],
                ),
                total=len(tasks),
            )
        )


def getObservations():
    global args
    data = []
    observations = []
    XMMMaster = args["XMMMaster"]
    allowedModes = list(map(lambda x: x.strip(), args["allowedModes"].split(",")))

    if "observations" not in args:
        raise FileNotFoundError('No "observations" file is provided.')

    with open(urlretrieve(XMMMaster)[0], "r") as file:
        lines = file.readlines()
        for row in lines[3:]:
            if row.strip() == "":
                continue
            data.append(list(map(lambda x: x.strip(), row.split("|"))))
        table = pd.DataFrame(
            data, columns=list(map(lambda x: x.strip(), lines[1].split("|")))
        )
        N = len(table)
        print(f"Number of entries: {N}.")
        for i in range(N):
            mode = None
            for mask in allowedModes:
                if (table["mode_filter"][i] is not None) and (
                    table["mode_filter"][i].find(mask) != -1
                ):
                    mode = mask
            if mode is not None:
                observations.append(table["obsno"][i] + " " + mode)
    observations.sort()
    with open(args["observations"], "w") as file:
        print(*observations, sep="\n", file=file)
    print("Observation parsing is finished.")


def getObservationIDs():
    global args
    if "observations" not in args:
        raise FileNotFoundError('No "observations" file is provided.')

    obsIDs = None
    with open(args["observations"], "r") as file:
        obsIDs = list(map(lambda x: x.split()[0].strip(), file.readlines()))

    if "processed" in args:
        result = []
        exclude = set()
        lines = []
        if os.path.isfile(args["processed"]):
            with open(args["processed"], "r") as file:
                for line in file.readlines():
                    if line[0] == "#":
                        exclude.add(line.split()[0].strip()[1:])
                        lines.append(line.strip())
                    elif line[0] == "~" or line[0] == "!":
                        result.append(line.split()[0].strip()[1:])
                        exclude.add(result[-1])
                    else:
                        lines.append(line.strip())
        for ID in obsIDs:
            if ID not in exclude:
                result.append(ID)
        with open(args["processed"], "w") as file:
            for line in lines:
                print(line, file=file)
        obsIDs = result

    return obsIDs


def extractTransients(processedFile, transientsFile):
    transients = []
    observationID = None
    with open(processedFile, "r") as file:
        for row in file.readlines():
            if row[0] == '#':
                observationID = row.split()[0][1:]
            elif row[0] == '!' or row[0] == '~':
                continue
            else:
                transients.append(tuple([observationID] + row.split() + ["Unreviewed"]))

    if os.path.isfile(transientsFile):
        with open(transientsFile, "r") as file:
            for row in file.readlines():
                transients.append(tuple(row.split()))

    with open(transientsFile, "w") as file:
        for transient in transients:
            print(*transient, sep="\t", file=file)


def reviewTransients(transientsFile, reviewModes):
    transients = {}
    reviewModes = list(map(lambda x: x.strip(), reviewModes.split(",")))
    with open(transientsFile, "r") as file:
        for row in file:
            tr = row.split()
            if tr[-1] not in reviewModes:
                continue
            if tr[0] not in transients:
                transients[tr[0]] = []
            transients[tr[0]].append(tr[1:])

    config = Calc.Config()
    for obsID, currtransients in transients.items():
        tf = Calc.TransientFinder(obsID, config)
        try:
            tf.getDetections(obsID)
        except Exception:
            print("Network error is occured during {obsID} review.")
        events = None
        try:
            events = tf.getEvents(obsID)
        except Exception:
            print("PIEVLI error is occured during {obsID} review.")

        visualiserTransients = []
        for transient in currtransients:
            visualiserTransients.append({
                "x": float(transient[0]),
                "y":float(transient[1]),
                "tmean": float(transient[2]),
                "tdev": float(transient[3]),
            })
        visualiser = Vision(events, tf.bincount, tf.picPsf, tf.localToPic)
        visualiser.addTransients(visualiserTransients)
        visualiser.initGUI(True)
        verdicts = visualiser.transientsVerdict
        for i, verdict in enumerate(verdicts):
            currtransients[i][-1] = verdict

        with open(transientsFile, "w") as file:
            for observation, trs in transients.items():
                for transient in trs:
                    print(observation, *transient, sep="\t", file=file)

def downloadObservation(args):
    ID, config = args
    tf = Calc.TransientFinder(ID, config)
    try:
        tf.getDetections(ID)
    except Exception:
        pass
    del tf

def downloadFiles(obsIDs):
    config = Calc.Config
    config.removeStars = True
    observations = [(ID, config) for ID in obsIDs]

    with Pool(1) as pool:
        list(tqdm(pool.imap_unordered(downloadObservation, observations), total=len(obsIDs)))

def catalogueFiles(cataloguefile, coreNum, lock):
    config = Calc.Config
    config.removeStars = True

    observations = next(os.walk(os.getcwd() + "/data/"), (None, [], None))[1]

    open(cataloguefile, "w").close()

    def checkObservation(args):
        ID, filelock, config, cataloguefile = args
        tf = Calc.TransientFinder(ID, config)
        verdict = ""
        try:
            events = tf.getEvents(ID)
            try:
                tf.removeStars(events, ID, show=False)
                verdict = "PO"
            except Exception:
                verdict = "Po"
        except Exception:
            verdict = "p"
        del tf
        filelock.acquire()
        with open(cataloguefile, "a") as file:
            print(ID, verdict, sep="\t", file=file)
        filelock.release()

    with Pool(coreNum) as pool:
        list(
            tqdm(
                pool.imap_unordered(
                    checkObservation,
                    [(ID, lock, config, cataloguefile) for ID in observations],
                ),
                total=len(observations),
            )
        )


if __name__ == "__main__":
    obsIDs = [
        "0303110101",
        "0004210201",
        "0004610401",
        "0002740501",
        "0001730201",
        "0000110101",
        "0001730601",
        "0001930301",
    ]

    timeStamp = time()
    conf = Calc.Config()
    coreNumber = 1

    for arg in sys.argv[2:]:
        name, val = arg.split("=", 1)
        if name == "config":
            with open(val, "r") as r:
                for line in r.readlines():
                    exec("conf." + line.strip())
        elif name == "coreNumber":
            coreNumber = int(val)
        else:
            args[name] = val

    if len(sys.argv) < 2:
        raise ValueError("No mode is provided.")
    mode = sys.argv[1]

    manager = Manager()
    lock = manager.Lock()
    panic = manager.Value("i", False)

    if mode == "getObservations":
        getObservations()
    elif mode == "process":
        obsIDs = getObservationIDs()
        if "processed" not in args:
            raise FileNotFoundError('No "processed" file is provided.')
        processImages(obsIDs, coreNumber, conf, lock, args["processed"], panic)
    elif mode == "image":
        if "ID" not in args:
            raise ValueError('No observation "ID" is provided.')
        tf = Calc.TransientFinder(args["ID"], conf)
        tf.process()
        print(tf.result)
        del tf
    elif mode == "test":
        obsIDs = getObservationIDs()
        if "result" not in args:
            raise FileNotFoundError('No "result" file is provided.')
        if "durations" not in args:
            raise FileNotFoundError('No "durations" are provided.')
        if "probs" not in args:
            raise FileNotFoundError('No "probs" are provided.')
        if "repeat" not in args:
            raise FileNotFoundError('No "repeat" number is provided')
        fakeImages(obsIDs, coreNumber, conf, lock, args["result"], panic,
                   args["durations"], args["probs"], args["repeat"])
    elif mode == "extract":
        if "processed" not in args:
            raise FileNotFoundError('No "processed" file is provided.')
        if "transients" not in args:
            raise FileNotFoundError('No "transients" file is provided.')
        extractTransients(args["processed"], args["transients"])
    elif mode == "review":
        if "transients" not in args:
            raise FileNotFoundError('No "transients" file is provided')
        reviewTransients(args["transients"], args["reviewModes"])
    elif mode == "download":
        obsIDs = getObservationIDs()
        downloadFiles(obsIDs)
    elif mode == "catalogue":
        if "catalogue" not in args:
            raise FileNotFoundError('No "catalogue" file is provided')
        catalogueFiles(args["catalogue"], coreNumber, lock)
    else:
        raise ValueError(f"Unknown mode: {mode}.")

    print(f"Time: {round(time() - timeStamp)} s")
    sys.exit(0)
