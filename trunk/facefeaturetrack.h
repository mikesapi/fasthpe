// Header File Template_match.h


void closeTemplateMatch();

int initTracker(IplImage* frame, CvPoint L, CvPoint R, CvPoint N, CvPoint M );
//int dynamicTracker(IplImage* frame, CvPoint L, CvPoint R, CvPoint N, CvPoint M );
IplImage* trackObject( IplImage* frame );
