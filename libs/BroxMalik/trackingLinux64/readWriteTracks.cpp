class CPoint {
public:
  CPoint() {}
  float x,y,frame;
};

class CSimpleTrack {
public:
  CSimpleTrack() {}
  int mLabel;
  CVector<CPoint> mPoints;
};

std::vector<CSimpleTrack> mTracks;
int mSequenceLength;

// writeTracks
void writeTracks() {
  // Create a filename and open a file
  char buffer[50];
  sprintf(buffer,"Tracks%d.dat",mSequenceLength);
  std::ofstream aFile(buffer);
  // Write number of frames considered
  aFile << mSequenceLength << std::endl;
  // Write number of tracks
  int aCount = mTracks.size();
  aFile << aCount << std::endl;
  // Write each track
  for (int i = 0; i < mTracks.size(); i++) {
    // Write label and length of track
    int aSize = mTracks[i].mPoints.size();
    aFile << mTracks[i].mLabel << " " << aSize << std::endl;
    // Write x,y coordinates and frame number of the tracked point 
    for (int j = 0; j < aSize; j++)
      aFile << mTracks[i].mPoints[j].x << " " << mTracks[i].mPoints[j].y << " " << mTracks[i].mPoints[j].frame << std::endl;
  }
}

// readTracks
void readTracks() {
  // Create a filename and open a file
  char buffer[50];
  sprintf(buffer,"Tracks%d.dat",mSequenceLength);
  std::ofstream aFile(buffer);
  // Read number of frames considered
  aFile >> mSequenceLength;
  // Read number of tracks
  int aCount;
  aFile >> aCount;
  // Read each track
  for (int i = 0; i < aCount; i++) {
    // Read label and length of track
    int aSize;
	mTracks.push_back(CSimpleTrack());
	CSimpleTrack& aTrack = mTracks.back();
	aFile >> aTrack.mLabel;
	aFile >> aSize;
	aTrack.mPoints.setSize(aSize);
	// Read x,y coordinates and frame number of the tracked point 
	for (int j = 0; j < aSize; j++) {
	  aFile >> aTrack.mPoints[j].x;
	  aFile >> aTrack.mPoints[j].y;
	  aFile >> aTrack.mPoints[j].frame;
	}
  }
}