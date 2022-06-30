# script to detect all point of interests in an audiosample

from email.mime import audio
import numpy as np


def detect_intervals(data: np.array, l: int = 9) -> list[tuple[int, int]]:
    
    intervals = []
    duration_samples = 460000.0

    # convert wave into db
    dbs = 10 * np.log10(abs(data) / (3.3656 * 10 ** -11))
    db_thresh = 90.0
    i = 0

    while i < len(dbs):

        db = dbs[i]

        # db over threshold
        if db > db_thresh:
            # count section length
            j = i
            length = 0
            while dbs[j] > db_thresh:
                print(j)
                length += 1
                j += 1
            # section long enough
            if length > 470000:
                intervals.append((i, i + length))
    
        i += 1

    return intervals

if __name__ == "__main__":
    import audiofile
    data, _ = audiofile.read("data/wav1.wav")

    print(detect_intervals(data))
