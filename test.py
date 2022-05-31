import numpy as np
import matplotlib.pyplot as plt
from fft_util import calc_fft as fft

SAMPLERATE = 192000

# amount of frames we split the data
N_FRAMES = 40

def generate_test_sine():

    # gedämpfte 
    nums = np.arange(SAMPLERATE)
    sine1 = np.zeros(SAMPLERATE)
    sine2 = np.zeros(SAMPLERATE)
    sine3 = np.zeros(SAMPLERATE)
    sine4 = np.zeros(SAMPLERATE)
    sine6 = np.zeros(SAMPLERATE)
    sine7 = np.zeros(SAMPLERATE)

    sine1 = np.power(10, -0.0000025 * nums) *  np.sin(0.01 * 2 * np.pi * nums)
    sine2 = np.power(10, -0.000005 * nums) *  np.sin(0.005 * 2 * np.pi * nums)
    sine3 = np.power(10, -0.000005 * nums) *  np.sin(0.0025 * 2 * np.pi * nums)
    sine4 = np.power(10, -0.0000025 * nums) *  np.sin(0.02 * 2 * np.pi * nums)

    sine5 = np.power(10, -0.000006 * nums) *  np.sin(0.025 * 2 * np.pi * nums)

    sine6 = np.power(10, -0.000004 * nums) *  np.sin(0.0225 * 2 * np.pi * nums)

    sine7 = np.power(10, -0.0000025 * nums) *  np.sin(0.003 * 2 * np.pi * nums)

    final_sine = sine1 + sine2 + sine3 + sine4 + sine5 + sine6 + sine7

    plt.plot(sine1)
    plt.plot(sine2)
    plt.plot(sine3)
    plt.plot(sine4)
    plt.plot(sine5)
    plt.plot(sine6)
    plt.plot(sine7)
    plt.title("einzelne Frequenzen")
    plt.show()

    plt.plot(final_sine)
    plt.title("kombinierte Frequenzen")
    plt.show()

    return final_sine

data = generate_test_sine()

# show spektrum for whole data
fs, amps = fft(data, SAMPLERATE)
plt.scatter(fs, amps)
plt.show()

frame_size = int(SAMPLERATE / N_FRAMES)
# split data into frames
frames = [ data[start : start + frame_size] for start in range(0, SAMPLERATE, frame_size) ]

# calc fft for each frame
spectrum_each_frame = [ fft(frame, SAMPLERATE) for frame in frames ]

# the max map for each freq
max_amp = dict()

# the wanted difference in amplitude; TODO this should relate to 60dB for RT60
# now just take 40% of the max amp of the first frame
wanted_diff = max(spectrum_each_frame[0][1]) * 0.4

# RT in samples for each freq
rt_for_freq = dict()

for t, (f, a) in enumerate(spectrum_each_frame):

    # cut away negative frequencies and above 5kHz
    tmp_zip = np.array(list(filter(lambda x: 5000.0 > x[0] >= 0, zip(f, a))))

    for freq, amp in tmp_zip:

        # RT has already been calculated for this frequency
        if max_amp.get(freq) == -1:
            continue

        # max amp has not been set for this freq yet
        if not freq in max_amp:

            # first frame, set amp, assuming its the max
            max_amp[freq] = amp
            continue

        rt_for_freq[freq] = 0

        # the wanted diff (60dB) has been reached
        if max_amp[freq] - amp >= wanted_diff:

            # mark freq as done
            # TODO do this smoother
            max_amp[freq] = -1

            # RT in samples
            samples = t * frame_size

            #rt_for_freq[freq] = samples / SAMPLERATE
            rt_for_freq[freq] = samples

# display rt
xaxis = np.array(list(rt_for_freq.keys()))
yaxis = np.array(list(rt_for_freq.values()))

plt.bar(xaxis, yaxis, width=20)
plt.ylabel("RT60 (samples)")
plt.xlabel("Frequenz (Hz)")
plt.title("Nachhallzeit für 192k samples")
plt.show()

print(xaxis)
print(yaxis)
