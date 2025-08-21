import Calc
import multiprocessing
from time import time, sleep

def run(ind, dur, prob, obsID, sumtime, success, found):
    conf = Calc.Config()
    conf.ftransientProb = prob
    conf.ftransientTime = dur
    conf.addFakeTransient = True
    tf = Calc.TransientFinder(obsID, conf)
    tf.process()
    sumtime.value += tf.result["CodeTime"]
    success.value += tf.result["FakeTransientStatus"] == "Probable"
    found.value += tf.result["FakeTransientStatus"] != "NotFound"
    print(ind, end=" ")
    del tf

def emptyFunc():
    pass

if __name__ == "__main__":
    """ List of usable IDs
    0000110101
    0001730201
    0001730601
    0001930301
    0002740301
    0002740501
    0004610401
    0004210201
    """

    obsIDMain = "0303110101"  # Remember where you've started
    obsIDs = [
        "0303110101",
        "0000110101",
        "0004210201",
        "0004610401",
    ]
    num = 100
    obsNum = 0
    coreNum = 4

    timeStamp = time()

    for dur in [100]:
        for prob in [0.01]:
            processes = [multiprocessing.Process(target=emptyFunc)] * coreNum
            processes[0].start() # Since all the array elements are references to the same null process.
            processes[0].join()
            sumtime = multiprocessing.Value('d', 0)
            success = multiprocessing.Value('i', 0)
            found = multiprocessing.Value('i', 0)
            for i in range(num):
                # Waiting for any process to finish
                pnum = 0
                while True:
                    if processes[pnum].exitcode is not None:
                        processes[pnum] = multiprocessing.Process(target=run, args=(i, dur, prob, obsIDs[obsNum], sumtime, success, found))
                        processes[pnum].start()
                        break
                    pnum += 1
                    pnum %= coreNum
                    sleep(1)
            for proc in processes:
                proc.join()
            print("")
            print(f"Average time: {round(sumtime.value / num, 2)}s\n"
                  + f"Success rate: {round(success.value / num * 100)}%\n"
                  + f"Find rate: {round(found.value / num * 100)}%\n"
                  + f"Observation parameters: {obsIDs[obsNum]}, {dur}, {prob}\n")

    print(f"Machine time: {round(sumtime.value, 2)}s\n" +
          f"Real time: {round(time() - timeStamp, 2)}s\n")