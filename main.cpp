/*
Copyright (C) 2011 Michael Sapienza
   
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/


#include <iostream>
#include <stdio.h>
#include<X11/extensions/Xrandr.h> //to get screen resolution

//compatible with opencv2.2
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc_c.h>
//#include "cv.h"

#include "capture.h"
#include "facefeaturedetect.h"
#include "pose-estimation.h"
#include "facefeaturetrack.h"

bool flag=0;

Face F;
Face *FPtr = &F;

FaceGeom G;
FaceGeom *GPtr = &G;

Pose P;
Pose *PPtr = &P;

//camera resolution
extern int W; //width
extern int H; //height

int screen_w, screen_h;

extern int is_tracking;

extern int frame_number;

const char *DISPLAY_WINDOW = "DisplayWindow";

IplImage *FrameCopy = 0;
IplImage *DisplayFrame = 0;

int initAll(int argc, char** argv);
void exitProgram(int code);
void captureVideoFrame();
//void equalize(IplImage *frame);
int GetScreenSize(int&,int&);

int key, delay;
bool isFace = 0;

bool CheckForFace(Face* F){
  const double tol1=15,tol2=25;
	//NOTE Need to write better function to make sure features are initialized when person is facing camera.
	if (F->LeftEye.x > 0. && F->RightEye.x > 0. && F->Nose.x > 0. && F->Mouth.x > 0. ){ //check all features were initialized
	  return(  ((F->RightEye.x - F->LeftEye.x) > ((double)F->FaceBox->width)/4.)
		&& ((F->RightEye.x + F->LeftEye.x + F->Nose.x)/3.) > (F->Nose.x - tol1)
		&& ((F->RightEye.y + F->LeftEye.y + F->Nose.y)/3.) > (F->Nose.y - tol2)
		&& ((F->RightEye.x + F->LeftEye.x + F->Nose.x)/3.) < (F->Nose.x + tol1)
		&& ((F->RightEye.y + F->LeftEye.y + F->Nose.y)/3.) < (F->Nose.y + tol2)
	);
	}
	else return 0;
  
}

// Main Program
//void main( int argc, char** argv )
int main(int argc, char** argv)
{
	int i;
	
	if( !initAll(argc, argv) ) exitProgram(-1);
	while(1){

	while(1)
	{
		for(i=0;i<2;i++){		//repeat 2 times to give person enough time to position their face in a frontal pose approx (250ms*2)

			captureVideoFrame();//get frame from camera

			double t = (double)cvGetTickCount();//start timer
			
			detect_features(DisplayFrame, FPtr);//detect face and facial features

			t = (double)cvGetTickCount() - t;//end timer
			
			printf( "detection time = %gms\n", t/((double)cvGetTickFrequency()*1000.) );//display timem in ms

			//printf( "Left Eye x=%d y=%d\n", F.LeftEye.x, F.LeftEye.y) ;
			//printf( "Right Eye x=%d y=%d\n", F.RightEye.x, F.RightEye.y) ;
			//printf( "Nose x=%d y=%d\n", F.Nose.x, F.Nose.y) ;
			//printf( "Mouth x=%d y=%d\n\n\n", F.Nose.x, F.Nose.y) ;

			cvShowImage(DISPLAY_WINDOW, DisplayFrame );//show result
			
			cvReleaseImage( &DisplayFrame);
			key = cvWaitKey( delay );
			if(key == 1048689 || key == 1048603 || key == 'q' )  exitProgram(0);//if user presses Esc or q , exit program
		}
		
		
		isFace = CheckForFace(FPtr); // exit loop when a face is detected
		
		if(isFace)	break;
	}

	
	//captureVideoFrame();//get frame from camera

	// initialize tracking (NOTE:it may work with grayscale instead of colour)
	is_tracking	= initTracker(FrameCopy, FPtr);
	//is_tracking	= dynamicTracker(FrameCopy, F->LeftEye, F->RightEye, F->Nose, F->Nose);
	i=0;
	//init_geometric_model(FPtr);
	init_geometric_model(FPtr,GPtr,PPtr);
	init_kalman_filter();

	while(1)
	{
		captureVideoFrame();//get frame from camera

		if(is_tracking)
		{
			//dymanic updating suffers from drift over time in its current state
			/*i++;
			if (i == 8){
				//is_tracking	= dynamicTracker(FrameCopy, F->LeftEye, F->RightEye, F->Nose, F->Nose);
				is_tracking	= initTracker(FrameCopy, F->LeftEye, F->RightEye, F->Nose, F->Nose);
				i=0;
			}*/

		double t = (double)cvGetTickCount();//start timer

		FrameCopy = trackObject(FrameCopy, FPtr, GPtr);

		draw_and_calculate(FrameCopy, FPtr, GPtr, PPtr);
		//printf( "Left Eye	x=%d y=%d\n",		F.LeftEye.x, F.LeftEye.y) ;
		//printf( "Right Eye	x=%d y=%d\n",		F.RightEye.x, F.RightEye.y) ;
		//printf( "Nose		x=%d y=%d\n",		F.Nose.x, F.Nose.y) ;
		//printf( "Mouth	x=%d y=%d\n\n",		F.Nose.x, F.Nose.y) ;

		t = (double)cvGetTickCount() - t;//end timer
		printf( "tracking+pose+kalman time = %gms\n\n", t/((double)cvGetTickFrequency()*1000.) );//display timem in ms
		}
		 	if (flag == 0) is_tracking=0;
		 	flag=1;
		cvShowImage( DISPLAY_WINDOW, FrameCopy );//show result

	
		//writeVideo( FrameCopy );
		
		cvReleaseImage( &FrameCopy );
		cvReleaseImage( &DisplayFrame);
		key = cvWaitKey( delay );
		
		if(key == 1048689 || key == 1048603 || key == 'q')  exitProgram(0);//if user presses Esc or q , exit program
		if(key == 1048690 || key == 'r') is_tracking = 0;//manual reinitialization press 'r'
		if(is_tracking == 0)
			break;
	}

	}
	
exitProgram(0);
//return 0;
}


int initAll(int argc, char** argv)
{
  
  if (argc < 2){
    if( !initCapture() ) return 0;
	delay = 10;
	// Startup message tells user how to begin and how to exit
	printf( "\n*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*\n"
		"*-*-*-*-Fasthpe (c) Michael Sapienza - UOM - 2009-*-*-*-*-*\n\n" 
		"Begin -> look straight at the camera and press 'Enter'\n\n"
		"Reinitialize -> press 'r'\n\n"
		"Exit -> press the ESC key,\n\n\n"
		"How to play: Hit the blue circles that\n" 
		"appear on the screen with the green circle\n"
		"that you can control with your head orientation\n\n\n"
		"***WARNING***\n"
		"--Prolonged use of this game may cause neck injuries--"
		"\n*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*\n");
	fgetc(stdin);
    
  }else{
    delay=30;
/*     int i;
     printf("argc = %d\n", argc);
 
     for (i = 0; i<argc; i++){
          printf("argv[%d] = %s\n", i, argv[i]);
     } */   
	if( !initVideoCapture(argv[1]) ) return 0;
  }


	if( !initFaceDet(	
						
	"haarcascades/haarcascade_frontalface_alt2.xml",
	"haarcascades/Nariz_nuevo_20stages.xml",
	//"haarcascades/haarcascade_eye_tree_eyeglasses.xml",
	"haarcascades/haarcascade_eye.xml",
	"haarcascades/mouth.xml"))
	  
	return 0;

	//int width, height;
	GetScreenSize( screen_w, screen_h);
// 	printf("Screen resolution: %d x %d\n",width, height);
// 	fgetc(stdin);
	// Create the display window
	cvNamedWindow( DISPLAY_WINDOW, 1 );
	cvMoveWindow ( DISPLAY_WINDOW, cvRound(screen_w/2)-cvRound(W/2),	20);

	return 1;
}


void exitProgram(int code)
{

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

	if( !FrameCopy )
	FrameCopy = cvCreateImage( cvGetSize(frame), 8, 3 );
	DisplayFrame = cvCreateImage( cvGetSize(frame), 8, 3 );

	cvCopy(frame, FrameCopy);

	cvFlip( FrameCopy, FrameCopy, 1);
	cvCopy(FrameCopy, DisplayFrame, 0);

	//equalize(FrameCopy);
}

/*
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
*/

//http://www.blitzbasic.com/Community/posts.php?topic=86911
int GetScreenSize(int& width, int& height)
{

 		int num_sizes;
 		Rotation original_rotation;

 		Display *dpy = XOpenDisplay(NULL);
 		Window root = RootWindow(dpy, 0);
 		XRRScreenSize *xrrs = XRRSizes(dpy, 0, &num_sizes);
 		//
 		//     GET CURRENT RESOLUTION AND FREQUENCY
 		//
 		XRRScreenConfiguration *conf = XRRGetScreenInfo(dpy, root);
 		short original_rate = XRRConfigCurrentRate(conf);
 		SizeID original_size_id = XRRConfigCurrentConfiguration(conf, &original_rotation);

  		width = xrrs[original_size_id].width;
  		height = xrrs[original_size_id].height;
 
 		XCloseDisplay(dpy);
		return 0;    //Return a value that can be used for error checking. 
  
}
