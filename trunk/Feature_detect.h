// Header File Feature_detect.h
int initFaceDet(const char * faceCascadePath,
				const char * noseCascadePath,
				const char * eyesCascadePath,
				const char * mouthCascadePath);

void     closeFaceDet();

IplImage* detect_features( IplImage* img );