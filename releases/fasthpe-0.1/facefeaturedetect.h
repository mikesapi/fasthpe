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

#ifndef FACE_FEATURE_DETECT_H
#define FACE_FEATURE_DETECT_H

struct facefeatures{
 //create CvPoint structures to hold the located feature coordinates
 //centre points 
 CvPoint2D32f 	Face; 
 CvPoint2D32f	LeftEye;	
 CvPoint2D32f	RightEye;
 CvPoint2D32f	Nose;
 CvPoint2D32f 	Mouth;
 
 CvPoint2D32f	NoseBase;
 CvPoint2D32f 	MidEyes;
 
 CvRect* FaceBox;
 CvRect* NoseBox;
 CvRect* EyeBox1;
 CvRect* EyeBox2;
 CvRect* MouthBox; 
 
};
typedef struct facefeatures Face;


int initFaceDet(const char * faceCascadePath,
		const char * noseCascadePath,
		const char * eyesCascadePath,
		const char * mouthCascadePath);
void closeFaceDet();
void detect_features( IplImage* img , Face* F);

#endif