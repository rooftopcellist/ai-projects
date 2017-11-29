'''
Implementation of Cooley and Tukey FFT Algorithm
Author: Christian M. Adams

Current Architecture:
FFT --> Markov Model (Predict)

Future Architecture:
FFT --> Neural Network (Classify) ------> Neural Network (Predict)
									|
									 ---> Markov Model (Predict)

How to use:
 1. Place prepared chord audio files in order (alphabetically) in the "songs" directory
 2.

'''

import os
import cmath
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile # get the api
import heapq
from pitch_id import pitch

SAMPLERATE = 44100
NUMFRAMES = 8192 * 2
#Lower Note Threshold values result in ignoring smaller peaks more
NOTE_THRESHOLD = 10
#Number of peaks used from the FFT, set = to len(psd_peaks) to use all
NUM_PEAKS = 6
LP = 70
#Higher HP equates to lower high pass frequency
HP = 1200

print "Audio Files will be Low-passed at", float(NUMFRAMES/LP), "Hz\nHigh-passed at ",SAMPLERATE/2    #see ln 138

iArray = (26160.0,19011.0,18757.0,18405.0,17888.0,14720.0,14285.0,17018.0,18014.0,17119.0,16400.0,17497.0,17846.0,15700.0,17636.0,17181.0)

def FFT(d,iArray):

	N = len(iArray)
	r = N//2
	z=[]
	for i in iArray:
			z.append(i + 0j)
	# print("Input: \n",z,"\n\n")
	theta = -2 * cmath.pi * d / N
	if N > 1:
			i=1
			while i <= N-1:
					k=0
					m=0
					# print("i = ", i)
					w = complex(cmath.cos(i*theta), cmath.sin(i*theta))
					while k <= N-1:
							# print("k = ", k)
							u = 1
							for m in range(r):
									#This prints all of the indexes
									# print("i= ", i,"k = ", k,"m = ",m, "r= ", r)
									t = z[k + m] - z[k + m + r]
									z[k + m] = z[k + m] + z[k + m +r]
									z[k+m+r] = t*u
									u = w * u
							k = k + 2*r
					r = r//2
					i = 2*i
			i=0
			for i in range(N):
					r = i
					k = 0
					m = 1
					while m <= N-1:
							k = 2*k + (r % 2)
							r = r//2
							m = 2*m
					if k > i:
							t = z[i]
							z[i] = z[k]
							z[k] = t

			if d < 0:
					i = 0
					while i <= N-1:
							z[i] = z[i]//N

	return z

#######################################################################
# Process FFT to get PSD
#######################################################################

def readSong(path):
	#Reads in wav audio file
	fs, data = wavfile.read(path) # load the data
	#splits to just use LEFT
	# if data.T[0]:
	# 	data = data.T[0]

	return data

def processData(data):
	b = [] 									# this is 8-bit track, b is now normalized on [-1,1)
	i=0
	while len(b) < NUMFRAMES:				#32768 is the highest power of 2 less than the sample rate (44100)
		b.append((data[i]/2**8.)*2-1)
		i+=1

	# print b

	#Computes the FFT of the input data and stores in fftData
	fftData = FFT(1,b)

	#Finds the complex conjugate and stores it in ccData
	ccData = []
	for item in fftData:
		temp = np.conjugate(item)
		ccData.append(temp)


	#Multiplies the FFT Data and their Complex Conjugates and stores in fftccData
	fftccData = []
	i=0
	for i in range(len(fftData)):
		temp = fftData[i]*ccData[i]
		fftccData.append(temp)

	#Converts from Complex to Real to get Power Spectrum Density
	psdData = []
	i=0
	for i in range(len(fftData)):
		temp = fftccData[i].real
		psdData.append(temp)


	#x axis scaling factor
	xAxis = []
	for i in range(len(psdData)):
		xAxis.append(i*SAMPLERATE/NUMFRAMES)

	d = int((HP/float(SAMPLERATE)) * NUMFRAMES)
	# d = NUMFRAMES/2  						# you only need half of the fft list (real signal symmetry)
	# d = NUMFRAMES/HP						#shrinks the affective window of the frequency domain
	#This is manually resizes the data to exclude everything above the Nyquist Frequency

	printDataY = []

	#Low Passes's psd at frequency = SAMPLERATE//LP
	low_pass_f = int((LP/float(SAMPLERATE)) * NUMFRAMES)
	# low_pass_f = len(psdData)//LP
	count = 0
	for item in psdData:
		count +=1
		if count > (low_pass_f):
			printDataY.append(item)
			if count >=d:
				break
		elif (count <= (low_pass_f)):
			printDataY.append(0)
			if count >=d:
				break

	printDataX = []
	low_pass_f = len(xAxis)//LP
	count = 0
	for item in xAxis:
		count +=1
		printDataX.append(item)
		if count >= d:
			break
	#Graphs PSD
	graph_raw(b)
	graph_psd(printDataX, printDataY)
	return printDataX,printDataY



#######################################################################
# Analytics
#######################################################################

def make_normal(printDataY):

	# norm_notes = Frequency peaks, note_names = Pitch names
	notes = []
	psd_peaks = heapq.nlargest(NUM_PEAKS, printDataY)

	count = 0
	for peak in psd_peaks:
		notes.append(printDataY.index(peak))


	norm_notes = []
	for note in notes:
		if note != 0:
			norm_notes.append((note * SAMPLERATE)/NUMFRAMES)

	# if norm_notes == []:
	# 	return None
	loudest_note = max(norm_notes)

	norm_notes = sorted(norm_notes, key=int)
	for i in range(len(norm_notes) - 1):
		# if i == 0:
		# 	continue
		if psd_peaks[i+1] <= ((psd_peaks[0]) / NOTE_THRESHOLD):
			norm_notes[i+1] = norm_notes[0]


	note_names = []
	for item in norm_notes:
		note_names.append(pitch(item))
	return note_names, norm_notes


def graph_raw(data):
#Plots raw input data (in this case a waveform - voltage readings)
	plt.plot(data)
	plt.show()

def graph_psd(printDataX, printDataY):
#Plots FFT Peaks
	plt.plot(printDataX,printDataY)
	plt.show()


#######################################################################
# Calls FFT & processing to get PSD and add chords to csv
#######################################################################

def process_song(songpath):
	data = readSong(songpath)
	printDataX,printDataY = processData(data)
	return make_normal(printDataY)


def write_csv(chords, name):
	# print chords
	with open('song_csv/'+name+'_report.csv', 'wb') as file:
		for i in range(len(chords)):
			temp = ''
			for item in chords[i]:
				temp += str(item)+','
			file.write(str(temp))
			file.write("\n")
	file.close()


def prepare_data_split():
	output = []
	chord_frequencies = []
	all_psd = []
	count = 0
	csv_output = []
	print "----------------------- Reading & Splitting .wav file every .5 seconds ------------------------"



	print "----------------------- Reading in all .wav files in the songs directory ------------------------"


	for filename in os.listdir('markov_songs'):
		if filename.endswith(".wav"):
			print "Reading: ",filename
			data = readSong("markov_songs/"+filename)
			n = 44100
			#splits every half second
			data = [data[i:i+n] for i in range(0, len(data), n)]
			for chord_sample in data:
				printDataX,printDataY = processData(chord_sample)
				chord_notes, chord_freq = make_normal(printDataY)
				# if chord_freq == None:
				# 	continue
				chord_frequencies.append(chord_notes)
				output.append(chord_notes)

				csv_output.append(chord_notes)

			# print "CHORD NOTES:\N",output, "CHORD PEAKS:\N", chord_frequencies

			write_csv(csv_output, filename)
			print chord_frequencies, output
	return chord_frequencies, output

# def prepare_data():
# 	output = []
# 	print "----------------------- Reading in all .wav files in the songs directory ------------------------"
# 	for filename in os.listdir('markov_songs'):
# 		if filename.endswith(".wav"):
# 			print filename
# 			output.append(process_song("markov_songs/"+filename)[0])
# 	write_csv(output, filename)

def main():

	print("Welcome to Audio FFT, by: Christian M. Adams.  This version displays only the audio data below the Nyquist Frequency. \n")

	print process_song('1ksine.wav')
	# prepare_data_split()

main()
