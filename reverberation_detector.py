# script to detect all point of interests in an audiosample
import numpy as np

def interval_len(interval: tuple[int, int]) -> int:
    return interval[1] - interval[0]

def generate_intervals(total_lenght: int, n_intervals: int) -> list[tuple[int, int]]:
    
    intervals = []
    len_interval = int(total_lenght / n_intervals)

    for i in range(n_intervals):
        start = i * len_interval
        intervals.append((start, start + len_interval))

    return intervals


def detect_intervals(data: np.array, l: int = 9) -> list[tuple[int, int]]:
    
    steps = generate_intervals(len(data), int(len(data) / 500))
    print(interval_len(steps[0]))

    intervals = []
    duration_samples = 430000.0
    #curr_interval = (0, 0)
    curr_interval = [0, 0]

    # convert wave into db
    dbs = 10 * np.log10(abs(data) / (3.3656 * 10 ** -11))
    db_thresh = 72.0

    for step in steps:

        #avg_db_step = 10 * np.log10(np.average(abs(dbs[step[0] : step[1]])) / (3.3656 * 10 ** -11)) 
        avg_db_step = np.average(dbs[step[0] : step[1]])
        #avg_db_step = max(dbs[step[0] : step[1]])
        # print(avg_db_step)
        if avg_db_step > db_thresh:
            
            # set current interval end to end of current step
            curr_interval[1] = step[1]

        elif interval_len(curr_interval) > duration_samples:
            intervals.append(curr_interval)
            curr_interval = [step[1], step[1]]
        else:
            curr_interval = [step[1], step[1]]


    return intervals



if __name__ == "__main__":
    import audiofile
    import matplotlib.pyplot as plt
    data, _ = audiofile.read("data/wav1.wav")

    intervals = detect_intervals(data)
    print(intervals)

    plt.plot(data)

    starts = [ s[0] for s in intervals ]
    ends = [ s[1] for s in intervals ]

    plt.vlines(starts, 0, 1, colors=["green"])
    plt.vlines(ends, 0, 1, colors=["red"])

    [ print(interval_len(i)) for i in intervals ]


    plt.show()
    

