#include <iostream>
#include <stdio.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/objdetect/objdetect.hpp>

#include "facefeaturedetect.h"

static CvHaarClassifierCascade* faceCascade = 0;			// Create a new Face Haar classifier
static CvHaarClassifierCascade* noseCascade = 0;			// Create a new Nose Haar classifier
static CvHaarClassifierCascade* eyesCascade = 0;			// Create a new Eyes Haar classifier
static CvHaarClassifierCascade* mouthCascade = 0;			// Create a new Mouth Haar classifier

static CvMemStorage* storage = cvCreateMemStorage(0);		// memory for detector to use

double scale_factor = 1;

//create CvPoint structures to hold the located feature coordinates
CvPoint Face_center;
CvPoint LeftEye_center;
CvPoint RightEye_center;
CvPoint Nose_center;
CvPoint Mouth_center;

CvRect* r;



// Function to initialise the face detection process
int initFaceDet(const char * faceCascadePath,
				const char * noseCascadePath,
				const char * eyesCascadePath,
				const char * mouthCascadePath)
{

	if( !(storage = cvCreateMemStorage(0)) )
	{	fprintf(stderr, "Can\'t allocate memory for face detection\n");
		return 0;
	}

	//Load the cascades
	faceCascade = (CvHaarClassifierCascade *)cvLoad( faceCascadePath, 0, 0, 0 );
	noseCascade = (CvHaarClassifierCascade *)cvLoad( noseCascadePath, 0, 0, 0 );
	eyesCascade = (CvHaarClassifierCascade *)cvLoad( eyesCascadePath, 0, 0, 0 );
	mouthCascade = (CvHaarClassifierCascade *)cvLoad( mouthCascadePath, 0, 0, 0 );

	if( !faceCascade )
	{
		fprintf(stderr, "Can\'t load Face Haar classifier cascade from\n"
		                "   %s\n"
		                "Please check that this is the correct path\n",
						faceCascadePath);
		return 0;
	}
	if( !noseCascade )
	{
		fprintf(stderr, "Can\'t load Nose Haar classifier cascade from\n"
		                "   %s\n"
		                "Please check that this is the correct path\n",
						noseCascadePath);
		return 0;
	}
	if( !eyesCascade )
	{
		fprintf(stderr, "Can\'t load Eyes Haar classifier cascade from\n"
		                "   %s\n"
		                "Please check that this is the correct path\n",
						eyesCascadePath);
		return 0;
	}
		if( !mouthCascade )
	{
		fprintf(stderr, "Can\'t load Mouth Haar classifier cascade from\n"
		                "   %s\n"
		                "Please check that this is the correct path\n",
						mouthCascadePath);
		return 0;
	}
	return 1;
}



// Function to release memory after program termination
void closeFaceDet()
{
	if(faceCascade) cvReleaseHaarClassifierCascade(&faceCascade);
	if(noseCascade) cvReleaseHaarClassifierCascade(&noseCascade);
	if(eyesCascade) cvReleaseHaarClassifierCascade(&eyesCascade);
	if(mouthCascade) cvReleaseHaarClassifierCascade(&mouthCascade);
	if(storage) cvReleaseMemStorage(&storage);
}



// Function that returns detected face and detected features
IplImage* detect_features( IplImage* img )
{
    
    IplImage *gray, *small_img;//temporary images

    int i, j, k, l; //loop variables

	CvPoint pt1, pt2;           
		
    int radius;

    // Create a new image based on the input image
    gray = cvCreateImage( cvSize(img->width,img->height), 8, 1 );
    small_img = cvCreateImage( cvSize( cvRound (img->width/scale_factor),
                         cvRound (img->height/scale_factor)), 8, 1 );

    cvCvtColor( img, gray, CV_BGR2GRAY );			//convert to grayscale_factor

    cvResize( gray, small_img, CV_INTER_LINEAR );	//resize

    cvEqualizeHist( small_img, small_img );			//Equalize Image Histogram

	// Clear the memory storage which was used before
    cvClearMemStorage( storage );

	 // Find whether the face cascade is loaded. If yes then ..
    if( faceCascade )
	{
        
        CvSeq* faces = cvHaarDetectObjects( 
											small_img,	//grayscale_factor image. If region of interest (ROI) is set, then the function will
														//respect that region.Th us, one way of speeding up face detection is to trim 
														//down the image boundaries using ROI
											faceCascade,	//The classifier cascade is just the Haar feature cascade 
														//loaded with cvLoad() in the face detect code.
											storage,	//Th e storage argument is an OpenCV �work buffer�
														//for the algorithm; it is allocated with cvCreateMemStorage(0) in the face detection
														//code and cleared for reuse with cvClearMemStorage(storage).
                                            1.1,		//The cvHaarDetectObjects()
														//function scans the input image for faces at all scale_factors. Setting the scale_factor_factor parameter
														//determines how big of a jump there is between each scale_factor; setting this to a higher value
														//means faster computation time at the cost of possible missed detections if the scaling
														//misses faces of certain sizes.
											3,			//Th e min_neighbors parameter is a control for preventing
														//false detection. Actual face locations in an image tend to get multiple �hits� in the same
														//area because the surrounding pixels and scale_factors oft en indicate a face. Setting this to the
														//default (3) in the face detection code indicates that we will only decide a face is present
														//in a location if there are at least three overlapping detections.
											0			//Th e flags parameter has four valid settings, which (as usual) may be combined with the Boolean OR operator.

                                            |CV_HAAR_FIND_BIGGEST_OBJECT	//tells OpenCV to return only the largest object
																			//found (hence the number of objects returned will be either one or none).
                                            |CV_HAAR_DO_ROUGH_SEARCH		//which is used only with CV_HAAR_FIND_BIGGEST_OBJECT.
																			//Th is fl ag is used to terminate the search at whatever scale_factor the fi rst candidate is found
																			//(with enough neighbors to be considered a �hit�).
                                            //|CV_HAAR_DO_CANNY_PRUNING		//Setting flags to this value causes fl at regions (nolines) to be skipped by the classifi er.
                                            //|CV_HAAR_SCALE_IMAGE			//tells the algorithm to scale_factor the image rather than the detector (this can yield
																			//some performance advantages in terms of how memory and cache are used).
                                            ,
                                            cvSize(40, 40) );
        

		// Loop the number of faces found.
        for( i = 0; i < (faces ? faces->total : 0); i++ )
		{
			
            r = (CvRect*)cvGetSeqElem( faces, i ); // Create a new rectangle for drawing the face


			// Find the dimensions of the face, and scale_factor it if necessary
            pt1.x = r->x*scale_factor;
            pt2.x = (r->x+r->width)*scale_factor;
            pt1.y = r->y*scale_factor;
            pt2.y = (r->y+r->height)*scale_factor;

            // Draw rectangle around face in the input image
            cvRectangle( img, pt1, pt2, CV_RGB(255,0,0), 3, 8, 0 );

			//Find the center point of the face
            Face_center.x = cvRound((r->x + r->width*0.5)*scale_factor); // Find the dimensions of the face,and scale_factor it if necessary
            Face_center.y = cvRound((r->y + r->height*0.5)*scale_factor);

            //radius = cvRound((r->width + r->height)*0.25*scale_factor);
            //cvCircle( img, center, radius, color, 3, 8, 0 ); // Draw the circle in the input image

	
			// Find whether the nose cascade is loaded. If yes then ..
			if(noseCascade)
			{
			//Set the Region of Interest to the middle part of the face image to locate the nose
			cvSetImageROI( small_img, cvRect(r->x + cvRound(r->width/4),r->y+10+cvRound(r->height/4),cvRound(r->width/2),cvRound (r->height/2)) );
		
			//See what the ROI is:
			/*cvNamedWindow("ROI Window", CV_WINDOW_AUTOSIZE);
			cvShowImage( "ROI Window", small_img);*/

            CvSeq* noses = cvHaarDetectObjects( 
										small_img,
										noseCascade,
										storage,
                                        1.1, 
										2, 
										0
                                        //|CV_HAAR_FIND_BIGGEST_OBJECT
                                        //|CV_HAAR_DO_ROUGH_SEARCH
                                        //|CV_HAAR_DO_CANNY_PRUNING
                                        //|CV_HAAR_SCALE_IMAGE
										,
                                        cvSize(5, 5) );

			
			// Loop the number of noses found.
            for( j = 0; j < (noses ? noses->total : 0); j++ )
            {
                CvRect* nr = (CvRect*)cvGetSeqElem( noses, j );
                Nose_center.x = cvRound((r->x + nr->x + nr->width*0.5)*scale_factor) + cvRound (r->width/4);
                Nose_center.y = cvRound((r->y + nr->y + nr->height*0.5)*scale_factor) +10+ cvRound (r->height/4);
                radius = cvRound((nr->width + nr->height)*0.25*scale_factor);
                cvCircle( img, Nose_center, radius, CV_RGB(0,255,0), 3, 8, 0 );			//draw circle around nose
			}
			//Reset the image ROI
			cvResetImageROI( small_img);
			}

			// Find whether the eyes cascade is loaded. If yes then ..
			if(eyesCascade)
			{
            
			cvSetImageROI( small_img, cvRect(r->x,r->y+15,r->width,cvRound (r->height/2)) );

	

            CvSeq* eyes = cvHaarDetectObjects( 
										small_img, 
										eyesCascade,
										storage,
                                        1.1, 
										3, 
										0
                                        //|CV_HAAR_FIND_BIGGEST_OBJECT
                                        //|CV_HAAR_DO_ROUGH_SEARCH
                                        |CV_HAAR_DO_CANNY_PRUNING
                                        //|CV_HAAR_SCALE_IMAGE
                                        ,
                                        cvSize(3, 3) );

			// Loop the number of eyes found.
            for( k = 0; k < (eyes ? eyes->total : 0); k++ )
            {
				//for the 1st eye found..
				if(k==0){
                CvRect* nr = (CvRect*)cvGetSeqElem( eyes, k );
                LeftEye_center.x = cvRound((r->x + nr->x + nr->width*0.5)*scale_factor);
                LeftEye_center.y = cvRound((r->y + nr->y + nr->height*0.5)*scale_factor) + 15;
                radius = cvRound((nr->width + nr->height)*0.25*scale_factor);
                cvCircle( img, LeftEye_center, radius, CV_RGB(0,0,255), 3, 8, 0 ); //draw circle around eye1
				}

				//for the 2nd eye found..
				if(k==1){
                CvRect* nr = (CvRect*)cvGetSeqElem( eyes, k );
                RightEye_center.x = cvRound((r->x + nr->x + nr->width*0.5)*scale_factor);
                RightEye_center.y = cvRound((r->y + nr->y + nr->height*0.5)*scale_factor) + 15;
                radius = cvRound((nr->width + nr->height)*0.25*scale_factor);
                cvCircle( img, RightEye_center, radius, CV_RGB(0,0,255), 3, 8, 0 ); //draw circle around eye2
				}

				//make sure the left eye is on the left, and the right eye is on the right!
				int hold1, hold2;
				if (LeftEye_center.x > RightEye_center.x && k==1){
					CvRect* nr = (CvRect*)cvGetSeqElem( eyes, k );
					hold1 = LeftEye_center.x;
					hold2 = LeftEye_center.y;
					LeftEye_center.x = cvRound((r->x + nr->x + nr->width*0.5)*scale_factor);
					LeftEye_center.y = cvRound((r->y + nr->y + nr->height*0.5)*scale_factor) + 15;
					RightEye_center.x = hold1;
					RightEye_center.y = hold2;
			}

			}
			cvResetImageROI( small_img);
			}
			
			
			if( mouthCascade )
			{

			//cvSetImageROI( small_img, cvRect(r->x+cvRound (r->width/4),r->y+10+(cvRound (r->height/2)),cvRound (r->width/2),cvRound (r->height/2)) );
			cvSetImageROI( small_img, cvRect(r->x+cvRound (r->width/4),r->y+(cvRound (r->height/2)),cvRound (r->width/2),cvRound (r->height/2)) );
			
            CvSeq* mouth = cvHaarDetectObjects( 
										small_img, 
										mouthCascade, 
										storage,
                                        1.1, 
										4, 
										0
                                        |CV_HAAR_FIND_BIGGEST_OBJECT
                                        //|CV_HAAR_DO_ROUGH_SEARCH
                                        //|CV_HAAR_DO_CANNY_PRUNING
                                        //|CV_HAAR_SCALE_IMAGE
                                        ,
                                        cvSize(10, 10) );
			
			// Loop the number of mouths found.
			for( l = 0; l < (mouth ? mouth->total : 0); l++ )
            {
                CvRect* nr = (CvRect*)cvGetSeqElem( mouth, l );
                Mouth_center.x = cvRound((r->x + nr->x + nr->width*0.5)*scale_factor)+ cvRound (r->width/4);
                //Mouth_center.y = cvRound((r->y + nr->y + nr->height*0.5)*scale_factor) + 10+(cvRound (r->height/2));
				Mouth_center.y = cvRound((r->y + nr->y + nr->height*0.5)*scale_factor) +(cvRound (r->height/2));
                radius = cvRound((nr->width + nr->height)*0.25*scale_factor);
                cvCircle( img, Mouth_center, radius, CV_RGB(255,255,0), 3, 8, 0 ); //draw circle around mouth
			}
			cvResetImageROI( small_img);

			}
		}
    }

	// Release the temp images created.
    cvReleaseImage( &gray );
    cvReleaseImage( &small_img );

	//return the image with detected features back to main program
	return img;

}

