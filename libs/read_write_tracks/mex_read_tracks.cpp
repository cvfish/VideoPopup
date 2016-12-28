#include "mex.h"
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

class CPoint
{
public:
	CPoint() {}
	float x,y,frame;
};

class CSimpleTrack
{
public:
	CSimpleTrack() {};
	int mLabel;
	vector<CPoint> mPoints;
};

void readTracks(char *filename, std::vector<CSimpleTrack>& mTracks, int& mSequenceLength)
{
	//Create a filename and open a file
	// 	char buffer[50];
	// 	sprintf(buffer,"Tracks%d.dat",mSequenceLength);
	// 	std::ofstream aFile(buffer);

	ifstream aFile(filename);
	if(aFile.fail())
	  {
	    printf("%s doesn't exist\n",filename);
	    return;
	  }

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

		CSimpleTrack  aTrack;
		aFile >> aTrack.mLabel;
		aFile >> aSize;
		// Read x,y coordinates and frame number of the tracked point
		for (int j = 0; j < aSize; j++)
		{
			CPoint point;
			aFile>>point.x;
			aFile>>point.y;
			aFile>>point.frame;
			aTrack.mPoints.push_back(point);
		}
		mTracks.push_back(aTrack);

	}

}

void mexFunction(int nlhs, mxArray *out[], int nrhs, const mxArray *in[])
{
	// input parameters: name of the tracks file
	// output parameters: measurement matrix, label vector

	char*  filename;
	mwSize buflen;
	int status;

	/* Check for proper input type */
	if(! mxIsChar(in[0]) || mxGetM(in[0]) != 1)
	{
		mexErrMsgTxt("Input argument must be a string");
	}

	buflen = mxGetN(in[0])*sizeof(mxChar)+1;
	filename = (char *)mxMalloc(buflen);

	/* Copy the string data into buf. */
	status = mxGetString(in[0], filename, buflen);

	//mexPrintf("The input string is:  %s\n", filename);

	vector<CSimpleTrack> mTracks;
	int mSequenceLength;

// 	if(access(filename,F_OK) == -1)
// 	  {
// 	    // printf("%s doesn't exist\n",filename);
// 	    // return;
// 	    mexPrintf("The input filename is: %s\n", filename);
// 	    mexErrMsgTxt("Filename doesn't exist\n");
// 	  }

	readTracks(filename,mTracks,mSequenceLength);

	int framenum = mSequenceLength;
	int tracksnum = mTracks.size();

	int * label = new int[tracksnum];       // label
	double * mm = new double[2*framenum*tracksnum];   // measurement matrix

	memset(label,0,tracksnum*sizeof(int));
	memset(mm,0,2*framenum*tracksnum*sizeof(double));

	for(int i = 0; i < tracksnum; i++)
	{
			CSimpleTrack& pTrack = mTracks[i];
			vector<CPoint>::iterator it;
			for(it=pTrack.mPoints.begin();it!=pTrack.mPoints.end();it++)
			{
				int frame = it->frame;
				mm[2*frame*tracksnum+i] = it->x;
				mm[(2*frame+1)*tracksnum+i] = it->y;
			}
			label[i] = pTrack.mLabel;
	}

	out[0]= mxCreateDoubleMatrix(2*framenum,tracksnum,mxREAL);
	out[1]= mxCreateDoubleMatrix(1,tracksnum,mxREAL);

	for(int i = 0; i < framenum; ++i)
	{
		for(int j = 0; j < tracksnum; ++j)
		{
			mxGetPr(out[0])[2*j*framenum+2*i]= mm[2*i*tracksnum+j];
			mxGetPr(out[0])[2*j*framenum+2*i + 1]= mm[(2*i+1)*tracksnum+j];
		}
	}

	for(int i = 0; i < tracksnum; ++i)
		mxGetPr(out[1])[i] = label[i];

	delete []label;
	delete []mm;

	mxFree(filename);
}
