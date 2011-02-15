#include "opencv/highgui.h"
#include "opencv/cv.h"
#include <stdio.h>

#include "capture.h"
#include "Feature_detect.h"
#include "Template_match.h"
#include "draw_and_calculate.h"

//camera resolution
extern int W;
extern int H;

extern CvPoint Face_center;
extern CvPoint LeftEye_center;
extern CvPoint RightEye_center;
extern CvPoint Nose_center;
extern CvPoint Mouth_center;

extern int is_tracking;

extern int frame_number;

extern float pitch[900];
extern float yaw[900];
extern float roll[900];

const char * DISPLAY_WINDOW = "DisplayWindow";

IplImage  * FrameCopy = 0;
IplImage  * DisplayFrame = 0;



extern CvRect* r;


int initAll();
void exitProgram(int code);
void captureVideoFrame();
void equalize(IplImage* frame);

int key;

// Main Program
//void main( int argc, char** argv )
int main(int argc, char** argv)
{
	int i;
	
	if( !initAll() ) exitProgram(-1);
	while(1){

	while(1)
	{
		for(i=0;i<2;i++){		//repeat 2 times to give person enough time to position their face in a frontal pose approx (250ms*10)

			captureVideoFrame();//get frame from camera

			double t = (double)cvGetTickCount();//start timer
			
			DisplayFrame = detect_features(DisplayFrame);//detect face and facial features

			t = (double)cvGetTickCount() - t;//end timer
			//printf( "detection time = %gms\n", t/((double)cvGetTickFrequency()*1000.) );//display timem in ms

			//printf( "Left Eye x=%d y=%d\n", LeftEye_center.x, LeftEye_center.y) ;
			//printf( "Right Eye x=%d y=%d\n", RightEye_center.x, RightEye_center.y) ;
			//printf( "Nose x=%d y=%d\n", Nose_center.x, Nose_center.y) ;
			//printf( "Mouth x=%d y=%d\n\n\n", Mouth_center.x, Mouth_center.y) ;

			cvShowImage(DISPLAY_WINDOW, DisplayFrame );//show result

			key = cvWaitKey( 10 );
			if(key == 1048689 || key == 1048603 || key == 'q' )  exitProgram(0);//if user presses Esc or q , exit program
		}
		// exit loop when a face is detected
		if(LeftEye_center.x > 0 && RightEye_center.x > 0 && Nose_center.x > 0 && Mouth_center.x > 0
			&& (RightEye_center.x - LeftEye_center.x) > (r->width)/4
			&& ((RightEye_center.x + LeftEye_center.x + Mouth_center.x)/3) > Nose_center.x -5
			&& ((RightEye_center.y + LeftEye_center.y + Mouth_center.y)/3) > Nose_center.y -20
			&& ((RightEye_center.x + LeftEye_center.x + Mouth_center.x)/3) < Nose_center.x +5
			&& ((RightEye_center.y + LeftEye_center.y + Mouth_center.y)/3) < Nose_center.y +10
			) 
			break;
	}

	
	//captureVideoFrame();//get frame from camera

	// initialize tracking (NOTE:try make it work with grayscale instead of colour)
	is_tracking	= initTracker(FrameCopy, LeftEye_center, RightEye_center, Nose_center, Mouth_center);
	//is_tracking	= dynamicTracker(FrameCopy, LeftEye_center, RightEye_center, Nose_center, Mouth_center);
	i=0;
	init_geometric_model();
	init_kalman_filter();

	while(1)
	{
		captureVideoFrame();//get frame from camera

		if(is_tracking)
		{
			//dymanic updating suffers from drift over time in its current state
			/*i++;
		
			if (i == 8){
				//is_tracking	= dynamicTracker(FrameCopy, LeftEye_center, RightEye_center, Nose_center, Mouth_center);
				is_tracking	= initTracker(FrameCopy, LeftEye_center, RightEye_center, Nose_center, Mouth_center);
				i=0;
			}*/

		double t = (double)cvGetTickCount();//start timer

		FrameCopy = trackObject(FrameCopy);

		FrameCopy = draw_and_calculate(FrameCopy);

		//printf( "Left Eye	x=%d y=%d\n",		LeftEye_center.x, LeftEye_center.y) ;
		//printf( "Right Eye	x=%d y=%d\n",		RightEye_center.x, RightEye_center.y) ;
		//printf( "Nose		x=%d y=%d\n",		Nose_center.x, Nose_center.y) ;
		//printf( "Mouth		x=%d y=%d\n\n",		Mouth_center.x, Mouth_center.y) ;

		t = (double)cvGetTickCount() - t;//end timer
		printf( "detection time = %gms\n\n", t/((double)cvGetTickFrequency()*1000.) );//display timem in ms
		}

		cvShowImage( DISPLAY_WINDOW, FrameCopy );//show result

	
		//writeVideo( FrameCopy );
		
		cvReleaseImage( &FrameCopy );
		cvReleaseImage( &DisplayFrame);
		key = cvWaitKey( 10 );
		
		if(key == 1048689 || key == 1048603 || key == 'q')  exitProgram(0);//if user presses Esc or q , exit program
		if(key == 1048690 || key == 'r') is_tracking = 0;//manual reinitialization press 'r'
		if(is_tracking == 0)
			break;
	}

	}
	
exitProgram(0);
//return 0;
}


int initAll()
{
	if( !initCapture() ) return 0;
	
	//if( !initVideoCapture() ) return 0;

	if( !initFaceDet(	
						
						"haarcascades/haarcascade_frontalface_alt2.xml",
						"haarcascades/Nariz_nuevo_20stages.xml",
						//"haarcascades/haarcascade_eye_tree_eyeglasses.xml",
						"haarcascades/haarcascade_eye.xml",
						"haarcascades/mouth.xml"))
	

		return 0;

	// Startup message tells user how to begin and how to exit
	printf( "\n*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*\n"
			"To begin, look at the camera and press 'Enter'\n\n"
			"To manually reinitialise the system press 'r'\n\n"
	        "To exit, click inside the video display,\n"
	        "then press the ESC key\n\n\n"
			"To play the game, hit the blue circles that\n" 
			"appear on the screen with the green circle\n"
			"that you can control with your head pose\n\n\n"
			"***WARNING***\n"
			"--Prolonged use of this product may cause neck injuries--"
			"\n*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*\n");
	fgetc(stdin);

	// Create the display window
	cvNamedWindow( DISPLAY_WINDOW, 1 );
	//cvMoveWindow ( DISPLAY_WINDOW, cvRound(1920/2)-cvRound(W/2),	20);
	cvMoveWindow ( DISPLAY_WINDOW, cvRound(1280/2)-cvRound(W/2),	20);

	return 1;
}


// exit Program
void exitProgram(int code)
{
	//printResults(roll, yaw, pitch);

	// Release resources allocated in this file
	cvDestroyWindow( DISPLAY_WINDOW );
	cvReleaseImage( &FrameCopy );
	    

	// Release resources allocated in other project files
	closeCapture();
	closeFaceDet();
	closeTemplateMatch();
	closeDraw();

	exit(code);
}

//get frame from camera

void captureVideoFrame()
{
	// Capture the next frame
	IplImage  * frame = nextFrame();
	if( !frame ){
		exitProgram(-1);
	}
	// Copy it to the display image, inverting it if needed
	if( !FrameCopy )
	//FrameCopy = cvCreateImage( cvGetSize(frame), 8, 3 );
	//DisplayFrame = cvCreateImage( cvGetSize(frame), 8, 3 );
	FrameCopy = cvCreateImage( cvSize(320,240), 8, 3 );
	DisplayFrame = cvCreateImage( cvSize(320,240), 8, 3 );
	cvResize(frame, FrameCopy);
	//cvCopy( frame, FrameCopy, 0 );
	//cvFlip( frame, FrameCopy, 1);
	cvFlip( FrameCopy, FrameCopy, 1);
	FrameCopy->origin = frame->origin;
	cvCopy(FrameCopy, DisplayFrame, 0);
	if( 1 == FrameCopy->origin ) // 1 means the image is inverted
	{
		//cvFlip( FrameCopy, 0, 0 );
		FrameCopy->origin = 0;
	}

	//equalize(FrameCopy);
}

void equalize(IplImage* frame)
{
IplImage* blue=cvCreateImage( cvGetSize(frame), 8, 1 );
IplImage* blue_e=cvCreateImage( cvGetSize(frame), 8, 1 );

IplImage* green=cvCreateImage( cvGetSize(frame), 8, 1 );
IplImage* green_e=cvCreateImage( cvGetSize(frame), 8, 1 );

IplImage* red=cvCreateImage( cvGetSize(frame), 8, 1 );
IplImage* red_e=cvCreateImage( cvGetSize(frame), 8, 1 );

//Setting the Image Channel of interest(COI)
cvSplit(frame, blue, green, red, NULL);

cvEqualizeHist(blue,blue_e);

cvEqualizeHist(green,green_e);

cvEqualizeHist(red,red_e);

cvMerge(blue_e, green_e, red_e, NULL, frame);

	
cvReleaseImage( &blue );
cvReleaseImage( &green );
cvReleaseImage( &red );
cvReleaseImage( &blue_e );
cvReleaseImage( &green_e );
cvReleaseImage( &red_e );

}
