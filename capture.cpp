#include "capture.h"

#include <stdio.h>

#include <opencv2/highgui.hpp>


int frames;

//camera image resolution
int W = 320; //width
int H = 240; //height

CvCapture *capture = 0; // Structure for getting video from camera or avi
int frame_number = 1;

//CvVideoWriter *writer;

//Initialize capture from avi
int initVideoCapture(char* video_path)
{


	capture = cvCreateFileCapture(video_path);

	frames = (int) cvGetCaptureProperty(
		capture,
		CV_CAP_PROP_FRAME_COUNT
		);
  
// 	writer = cvCreateVideoWriter(
// 		"my_video.avi",
// 		CV_FOURCC('P','I','M','1'),
// 		30,
// 		size,
// 		isColor
// 		);

	printf("no of frames = %d", frames);

	if( !frames )
	{
		fprintf(stderr, "failed to initialize camera capture\n");
		return 0;
	}

	return 1;
}

// Initialize capture from camera
int initCapture()
{
	//capture = cvCaptureFromCAM( CV_CAP_ANY );
	//capture = cvCreateCameraCapture(0);
	//capture = cvCaptureFromCAM( 1 );
	capture = cvCaptureFromCAM(0);
	cvSetCaptureProperty( capture, CV_CAP_PROP_FRAME_WIDTH, W );
	cvSetCaptureProperty( capture, CV_CAP_PROP_FRAME_HEIGHT, H );
	//cvSetCaptureProperty(capture, CV_CAP_PROP_FPS, 30.0);

	if( !capture )
	{
		fprintf(stderr, "failed to initialize camera capture\n");
		return 0;
	}

	return 1;
}



// closeCapture()
void closeCapture()
{
	// Terminate video capture and free capture resources
	cvReleaseCapture( &capture );
	//cvReleaseVideoWriter( &writer );

	return;
}


// Get the next frame from the camera
IplImage * nextFrame()
{
	IplImage * frame = cvQueryFrame( capture );
	
	if( !frame ){
		fprintf(stderr, "failed to get a video frame\n");
		frame_number = frame_number -1;

		return NULL;
	}

	return frame;
}

/*void writeVideo( IplImage* frame)
{
	cvWriteFrame( writer, frame);
	
	
}
*/
