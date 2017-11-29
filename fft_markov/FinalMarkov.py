import random
import os
import midi
from make_mid import *

#Author: Devan Corcoran

#This markov chain looks as many chords back as you wish


def main():

    songgroups = []

    history = int(input("How far back? "))
    amountofsongs = 0
    for filename in os.listdir('song_csv'):
        print("File read in: ", filename)
        if filename.endswith(".csv"):

            #read file for each song
            chordlist1 = readfile("song_csv/"+filename)

            #add endindicators for each song
            chords = addEndIndicators(chordlist1, history)

            #add each song chord list to songgroups
            songgroups.append(chords)

            #be sure to determine the number of songs given
            amountofsongs += 1

        print("The Markov Model has been trained on", amountofsongs, "songs. ")

    finalchords = []

    for each in songgroups:

        for chord in each:

            finalchords.append(chord)

    tranchords, trancount = maptransitions(finalchords, history, amountofsongs)
    # print("\n\n", tranchords,"\n\n")
    # print(trancount)

    song = generatemusic(tranchords, trancount, history)

    finalsong = prepare_song(song)

    make_midi(finalsong)

def readfile(filename):

    f = open(filename, 'r')

    listofchords = f.read().splitlines()

    f.close()

    return listofchords


def addEndIndicators(listofchords, his):

    newlist = []
    for i in range(his):

        newlist.append('$')

    for each in listofchords:

        newlist.append(each)

    newlist.append('#')

    return newlist


def maptransitions(chords, his, numofsongs):

    transitionchords = []
    transitioncount = []
    endcount = 0
    bigcount = 0
    done = False

    while not done:

        smallcount = 0
        tran = []

        for i in range(his+1):

            tran.append(chords[smallcount+bigcount])

            if chords[smallcount+bigcount] == '#':
                endcount+=1

                if numofsongs < 2:
                    done = True
                elif endcount == (numofsongs+his):
                    done = True

            smallcount += 1

        bigcount += 1

        if tran in transitionchords:

            location = transitionchords.index(tran)
            transitioncount[location] += 1

        elif len(tran) > (his):

            transitionchords.append(tran)
            transitioncount.append(int(1))

    return transitionchords, transitioncount


def generatemusic(tranchords, trancounts, his):

    song = []

    startchords = []

    done = False

    for i in range(his):

        startchords.append('$')

    while not done:

        nextchord = determinenextchord(startchords, tranchords, trancounts, his)

        if nextchord == '#':
            done = True
        else:

            song.append(nextchord)

        del startchords[0]
        startchords.append(nextchord)

    return song


def determinenextchord(startchords, tranchords, trancounts, his):

    possiblechords = []

    for each in tranchords:

        count = 0
        equalscount = 0

        for k in range(len(startchords)):

            if startchords[k] == each[count]:

                equalscount += 1

            count += 1

        if equalscount == his:

            for p in range(trancounts[tranchords.index(each)]):

                possiblechords.append(each[-1])

    nextchord = random.choice(possiblechords)

    return nextchord

def prepare_song(chords):
	song = []
	for item in chords:
		temp = item.split(",")
		temp.remove('')
		song.append(temp)

	return song


main()
