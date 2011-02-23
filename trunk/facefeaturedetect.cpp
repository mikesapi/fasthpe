/*
Copyright (C) 2010 Michael Sapienza
   
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

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/objdetect/objdetect.hpp>

#include "facefeaturedetect.h"

static CvHaarClassifierCascade* faceCascade = 0;		// Create a new Face Haar classifier
static CvHaarClassifierCascade* noseCascade = 0;		// Create a new Nose Haar classifier
static CvHaarClassifierCascade* eyesCascade = 0;		// Create a new Eyes Haar classifier
static CvHaarClassifierCascade* mouthCascade = 0;		// Create a new Mouth Haar classifier

static CvMemStorage* storage = cvCreateMemStorage(0);		// memory for detector to use

double scale_factor = 1;

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
IplImage* detect_features( IplImage* img, Face* F )
{
  
//     CvSeq* faces;
//     CvSeq* noses;
//     CvSeq* eyes;
//     CvSeq* mouth;
    
    IplImage *gray, *small_img; //temporary images

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
			
            F->FaceBox = (CvRect*)cvGetSeqElem( faces, i ); // Create a new rectangle for drawing the face

	    // Find the dimensions of the face, and scale_factor it if necessary
            pt1.x = F->FaceBox->x*scale_factor;
            pt2.x = (F->FaceBox->x+F->FaceBox->width)*scale_factor;
            pt1.y = F->FaceBox->y*scale_factor;
            pt2.y = (F->FaceBox->y+F->FaceBox->height)*scale_factor;

            // Draw rectangle around face in the input image
            cvRectangle( img, pt1, pt2, CV_RGB(255,0,0), 3, 8, 0 );

	    //Find the center point of the face
            F->Face.x = cvRound((F->FaceBox->x + F->FaceBox->width*0.5)*scale_factor); // Find the dimensions of the face,and scale_factor it if necessary
            F->Face.y = cvRound((F->FaceBox->y + F->FaceBox->height*0.5)*scale_factor);

            //radius = cvRound((r->width + r->height)*0.25*scale_factor);
            //cvCircle( img, center, radius, color, 3, 8, 0 ); // Draw the circle in the input image

	
			// Find whether the nose cascade is loaded. If yes then ..
			if(noseCascade)
			{
			//Set the Region of Interest to the middle part of the face image to locate the nose
			cvSetImageROI( small_img, cvRect(F->FaceBox->x + cvRound(F->FaceBox->width/4),F->FaceBox->y+10+cvRound(F->FaceBox->height/4),cvRound(F->FaceBox->width/2),cvRound (F->FaceBox->height/2)) );
		
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
                F->NoseBox = (CvRect*)cvGetSeqElem( noses, j );
                F->Nose.x = cvRound((F->FaceBox->x + F->NoseBox->x + F->NoseBox->width*0.5)*scale_factor) + cvRound (F->FaceBox->width/4);
                F->Nose.y = cvRound((F->FaceBox->y + F->NoseBox->y + F->NoseBox->height*0.5)*scale_factor) +10+ cvRound (F->FaceBox->height/4);
                radius = cvRound((F->NoseBox->width + F->NoseBox->height)*0.25*scale_factor);
                cvCircle( img, cvPoint(F->Nose.x,F->Nose.y), radius, CV_RGB(0,255,0), 3, 8, 0 );			//draw circle around nose
			}
			//Reset the image ROI
			cvResetImageROI( small_img);
			}

			// Find whether the eyes cascade is loaded. If yes then ..
			if(eyesCascade)
			{
            
			cvSetImageROI( small_img, cvRect(F->FaceBox->x,F->FaceBox->y+15,F->FaceBox->width,cvRound (F->FaceBox->height/2)) );

	

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
                F->EyeBox1 = (CvRect*)cvGetSeqElem( eyes, k );
                F->LeftEye.x = cvRound((F->FaceBox->x + F->EyeBox1->x + F->EyeBox1->width*0.5)*scale_factor);
                F->LeftEye.y = cvRound((F->FaceBox->y + F->EyeBox1->y + F->EyeBox1->height*0.5)*scale_factor) + 15;
                radius = cvRound((F->EyeBox1->width + F->EyeBox1->height)*0.25*scale_factor);
                cvCircle( img, cvPoint(F->LeftEye.x,F->LeftEye.y), radius, CV_RGB(0,0,255), 3, 8, 0 ); //draw circle around eye1
				}

				//for the 2nd eye found..
				if(k==1){
                F->EyeBox2 = (CvRect*)cvGetSeqElem( eyes, k );
                F->RightEye.x = cvRound((F->FaceBox->x + F->EyeBox2->x + F->EyeBox2->width*0.5)*scale_factor);
                F->RightEye.y = cvRound((F->FaceBox->y + F->EyeBox2->y + F->EyeBox2->height*0.5)*scale_factor) + 15;
                radius = cvRound((F->EyeBox2->width + F->EyeBox2->height)*0.25*scale_factor);
                cvCircle( img, cvPoint(F->RightEye.x,F->RightEye.y), radius, CV_RGB(0,0,255), 3, 8, 0 ); //draw circle around eye2
				}

				//make sure the left eye is on the left, and the right eye is on the right!
				int hold1, hold2;
				if (F->LeftEye.x > F->RightEye.x && k==1){
					//CvRect* nr = (CvRect*)cvGetSeqElem( eyes, k );
					hold1 = F->LeftEye.x;
					hold2 = F->LeftEye.y;
					F->LeftEye.x = cvRound((F->FaceBox->x + F->EyeBox2->x + F->EyeBox2->width*0.5)*scale_factor);
					F->LeftEye.y = cvRound((F->FaceBox->y + F->EyeBox2->y + F->EyeBox2->height*0.5)*scale_factor) + 15;
					F->RightEye.x = hold1;
					F->RightEye.y = hold2;
			}

			}
			cvResetImageROI( small_img);
			}
			
			
			if( mouthCascade )
			{

			//cvSetImageROI( small_img, cvRect(r->x+cvRound (F->FaceBox->width/4),F->FaceBox->y+10+(cvRound (F->FaceBox->height/2)),cvRound (F->FaceBox->width/2),cvRound (F->FaceBox->height/2)) );
			cvSetImageROI( small_img, cvRect(F->FaceBox->x+cvRound (F->FaceBox->width/4),F->FaceBox->y+(cvRound (F->FaceBox->height/2)),cvRound (F->FaceBox->width/2),cvRound (F->FaceBox->height/2)) );
			
        CvSeq*    mouth = cvHaarDetectObjects( 
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
                F->MouthBox = (CvRect*)cvGetSeqElem( mouth, l );
                F->Mouth.x = cvRound((F->FaceBox->x + F->MouthBox->x + F->MouthBox->width*0.5)*scale_factor)+ cvRound (F->FaceBox->width/4);
                //Mouth_center.y = cvRound((F->FaceBox->y + F->MouthBox->y + F->MouthBox->height*0.5)*scale_factor) + 10+(cvRound (F->FaceBox->height/2));
		F->Mouth.y = cvRound((F->FaceBox->y + F->MouthBox->y + F->MouthBox->height*0.5)*scale_factor) +(cvRound (F->FaceBox->height/2));
                radius = cvRound((F->MouthBox->width + F->MouthBox->height)*0.25*scale_factor);
                cvCircle( img, cvPoint(F->Mouth.x,F->Mouth.y), radius, CV_RGB(255,255,0), 3, 8, 0 ); //draw circle around mouth
			}
			cvResetImageROI( small_img);

			}
		}
    }

	// Release the temp images and sequences created.
// 	cvClearSeq(faces);
// 	cvClearSeq(noses);
// 	cvClearSeq(eyes);
// 	cvClearSeq(mouth);
	cvReleaseImage( &gray );
	cvReleaseImage( &small_img );

	//return the image with detected features back to main program
	return img;

}

