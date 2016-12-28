#include <stdio.h>
#include <vector>
#include <iostream>
#include <string.h>

#include "TracksIO.h"

using namespace std;

// writeTracks
void CTracksIO::writeTracks(char *filename, std::vector<CSimpleTrack>& mTracks, int& mSequenceLength)
{
	// Create a filename and open a file
// 	char buffer[50];
// 	sprintf(buffer,"Tracks%d.dat",mSequenceLength);
// 	std::ofstream aFile(buffer);
	std::ofstream aFile(filename);
	// write number of frames considered
	aFile<< mSequenceLength << std::endl;
	// write number of tracks
	int aCount = 0;
	for(int i = 0; i < mTracks.size(); i++)
		if (mTracks[i].mLabel >= 0) aCount++;
	aFile<<aCount<<std::endl;
	// write each track
	for(int i = 0; i < mTracks.size(); i++)
	{
		// Ignore tracks marked as outliers
		if (mTracks[i].mLabel < 0) continue;
		// write label and length of track
		int aSize = mTracks[i].mPoints.size();
		aFile<<mTracks[i].mLabel<<" "<<aSize<<std::endl;
		// write x,y coordinates and frame number of the tracked point
		for (int j = 0; j < aSize; j ++)
			aFile<< mTracks[i].mPoints[j].x<<" "<<mTracks[i].mPoints[j].y<<" "<<mTracks[i].mPoints[j].frame<<std::endl;
	}
}

// readTracks
void CTracksIO::readTracks(char *filename, std::vector<CSimpleTrack>& mTracks, int& mSequenceLength)
{
	//Create a filename and open a file
// 	char buffer[50];
// 	sprintf(buffer,"Tracks%d.dat",mSequenceLength);
// 	std::ofstream aFile(buffer);

	std::ofstream aFile(filename);

	// Read number of frames considered
	aFile>>mSequenceLength;
	// Read number of tracks
	int aCount;
	aFile>>aCount;
	// Read each track
	for (int i = 0; i < aCount; i++)
	{
		// Read label and length of track
		int aSize;
		mTracks.push_back(CSimpleTrack());
		CSimpleTrack& aTrack = mTracks.back();
		aFile >> aTrack.mLabel;
		aFile >> aSize;
		aTrack.mPoints.setSize(aSize);
		// Read x,y coordinates and frame number of the tracked point
		for (int j = 0; j < aSize; j++)
		{
			aFile>>aTrack.mPoints[j].x;
			aFile>>aTrack.mPoints[j].y;
			aFile>>aTrack.mPoints[j].frame;
		}

	}
}
