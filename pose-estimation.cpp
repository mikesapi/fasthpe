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

#include "pose-estimation.h"

#include "facefeaturedetect.h"
#include "facefeaturetrack.h"

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <iostream>

#include <opencv2/video/video.hpp>
#include <opencv2/legacy/compat.hpp>

//#include <qsound.h>
using namespace std;

/*void gotoxy ( short x, short y )
{
  COORD coord = {x, y};
  SetConsoleCursorPosition ( GetStdHandle ( STD_OUTPUT_HANDLE ), coord );
}
void gotoxy( short x, short y ) {

HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
COORD position = { x, y }; 

SetConsoleCursorPosition( hStdout, position );
}*/

//global variables - coordinates of features
// extern CvPoint LeftEye_center_corner;
// extern CvPoint RightEye_center_corner;

extern int W;
extern int H;

extern int ldx;
extern int ldy;
extern int rdx;
extern int rdy;
extern int frame_number;

extern int is_tracking;

CvMemStorage *storage = cvCreateMemStorage(0);
CvSeq *seq = cvCreateSeq(CV_SEQ_FLAG_CLOSED | CV_SEQ_KIND_CURVE | CV_SEQ_ELTYPE_POINT, sizeof(CvSeq), sizeof(CvPoint), storage); 

//variables for 3D model

//float R_m = 0.52;//llm
//float  R_m = 0.58;//ssm from his face
//float  R_n = 0.5;//ssm
//float  R_m = 0.5;//jam from his face
//float  R_n = 0.55;//jam
float  R_m = 0.5;
float  R_n = 0.5;
float  R_e = 0.91;
double pi = 3.141592653589;
//

// float Pitch = 0;
// float Pitch_k_1;
// float Pitch_kalman;
// float Roll;
// float slant;
// float Yaw = 0;
// float Yaw_k_1;
// float Yaw_kalman;
// float pitch[900];
// float yaw[900];
// float roll[900];

float scale;
CvPoint rand_coord;
int t = 0;


CvPoint pointer_2d;
CvPoint pointer_2d_kalman;
CvPoint mouse;



///////////////////////////
//KALMAN FILTER VARIABLES//
///////////////////////////
CvRandState rng; //Random number generator

CvKalman* kalman = cvCreateKalman(     //Kalman filter structure
				   //6,  //n-dim state vector/
				    4,
				   //6,  //m-dim measurement vector
				    4,
				   0   //c-dim control vector
				 ); 

//CvMat* x_k = cvCreateMat( 6, 1, CV_32FC1 );  //state vector, n-dim
CvMat* x_k = cvCreateMat( 4, 1, CV_32FC1 );  //state vector, n-dim

//CvMat* w_k = cvCreateMat( 6, 1, CV_32FC1 );  //Process noise, n-dim
CvMat* w_k = cvCreateMat( 4, 1, CV_32FC1 );  //Process noise, n-dim

//CvMat* z_k = cvCreateMat( 6, 1, CV_32FC1 );  //measurement vector, m-dim 
CvMat* z_k = cvCreateMat( 4, 1, CV_32FC1 );  //measurement vector, m-dim
///////////////////////////
///////////////////////////

void init_kalman_filter(void)
{
cvRandInit( &rng, 0, 1, -1, CV_RAND_UNI );

cvRandSetRange( &rng, 0, 0.1, 0 );
	rng.disttype = CV_RAND_NORMAL;
	cvRand( &rng, x_k );
// Measurements, only one parameter for angle

cvZero( z_k );

        // Transition matrix F describes model parameters at and k and k+1 (the dynamics of our model)
//	cvSetIdentity( kalman->transition_matrix, cvRealScalar(1) ); // Transfer matrix, nxn-dim, set to Identity Matrix, no dynamics.

	const float F[] = {   1,    1,    0,    0,
			      0,    0.98,  0,    0,
			      0,    0,    1,    1,
			      0,    0,    0,    0.99    };


/*
	const float F[] = {   0.98,    0.027,    0,    0,
			      -1.08,    0.656,    0,    0,
			      0,    0,    1,    0,
			      0,    0,    0,    1    };
*/

/*
	const float F[] = {   1,    0,    0,    0,
			      0,    1,    0,    0,
			      0,    0,    1,    0,
			      0,    0,    0,    1    };
*/
	memcpy( kalman->transition_matrix->data.fl, F, sizeof(F));

	cvSetIdentity( kalman->measurement_matrix, cvRealScalar(1) );	// Initialize Measurement matrix H.

	//cvSetIdentity( kalman->process_noise_cov, cvRealScalar(1e-5) );
	cvSetIdentity( kalman->process_noise_cov, cvRealScalar(20*0.0001) );      //Initialize Process Noise covariance nxn-dim.

	//cvSetIdentity( kalman->measurement_noise_cov, cvRealScalar(1e-2) );  //Initialize measurement Noise covariance mxm-dim.
	cvSetIdentity( kalman->measurement_noise_cov, cvRealScalar(80*1) );
	//cvSetIdentity( kalman->measurement_noise_cov, cvRealScalar(5e-2) );

	cvSetIdentity( kalman->error_cov_post, cvRealScalar(1) );		//Initialize posterior Noise covariance.

	cvRand( &rng, kalman->state_post ); // Choose random initial state
}

void init_geometric_model(Face *F, FaceGeom *G, Pose *P)
{
	G->init_LeftEye_Nose_distance 	= FindDistance2D32f(F->Nose, F->LeftEye);	//initial distance between Nose center and left Eye
	G->init_RightEye_Nose_distance 	= FindDistance2D32f(F->Nose, F->RightEye);	//initial distance between Nose center and right Eye
	G->init_LeftEye_RightEye_distance = FindDistance2D32f(F->LeftEye, F->RightEye);	//initial distance between left Eye and right Eye
	G->init_Nose_Mouth_distance 	= FindDistance2D32f(F->Nose, F->Mouth);		//initial distance between Nose center and Mouth
	G->init_Mean_Feature_distance 	= (G->init_LeftEye_Nose_distance 
					  + G->init_RightEye_Nose_distance 
					  + G->init_LeftEye_RightEye_distance 
					  + G->init_Nose_Mouth_distance)/4;
					  
	G->LeftEye_Nose_distance 	= G->init_LeftEye_Nose_distance;
	G->RightEye_Nose_distance 	= G->init_RightEye_Nose_distance;
	G->LeftEye_RightEye_distance 	= G->init_LeftEye_RightEye_distance;
	G->Nose_Mouth_distance 		= G->init_Nose_Mouth_distance;
	G->Mean_Feature_distance 	= G->init_Mean_Feature_distance;	
	
	P->yaw = 0.;
	P->pitch = 0.;
	P->roll = 0.;
	
	rand_coord.x = W/2;
	rand_coord.y = H/2;
	srand( time(NULL));
	
	pointer_2d.x = (F->NoseBase.x + cvRound(500*(tan((double)P->yaw))));
	pointer_2d.y = (F->NoseBase.y + cvRound(500*(tan((double)P->pitch))));
	//pointer_2d.x = F->NoseBase.x;
	//pointer_2d.y = F->NoseBase.y;
	
	//mouse.x = 1280/2 - 320/2 + ((F->NoseBase.x + cvRound(1500*(tan((double)Yaw)))));
	//mouse.y = 800/2 - 240/2 + ((F->NoseBase.y + cvRound(1500*(tan((double)Pitch)))));
	t = 0;
}

//Function to calculate head pose and draw lines connecting features
void draw_and_calculate( IplImage *img, Face *F, FaceGeom *G, Pose *P){

	//Find center point between the eyes
	F->MidEyes.x = (F->LeftEye.x + F->RightEye.x)/2;
	F->MidEyes.y = (F->LeftEye.y + F->RightEye.y)/2;

	//Find center point between the eyes
	//Eye_center.x = cvRound((F->LeftEye_corner.x + F->RightEye_corner.x)/2);
	//Eye_center.y = cvRound((F->LeftEye_corner.y + F->RightEye_corner.y)/2);

	//Find the nose base along the symmetry axis
	F->NoseBase.x = F->Mouth.x + (F->MidEyes.x - F->Mouth.x)*(R_m);
	F->NoseBase.y = F->Mouth.y - (F->Mouth.y -F->MidEyes.y)*(R_m);
	

	G->LeftEye_Nose_distance 	= FindDistance2D32f(F->Nose, F->LeftEye);	//distance between Nose center and left Eye
	G->RightEye_Nose_distance 	= FindDistance2D32f(F->Nose, F->RightEye);	//distance between Nose center and right Eye
	G->LeftEye_RightEye_distance 	= FindDistance2D32f(F->LeftEye, F->RightEye);	//distance between left Eye and right Eye
	G->Nose_Mouth_distance 		= FindDistance2D32f(F->Nose, F->Mouth);		//distance between Nose center and Mouth

	G->Mean_Feature_distance 	= (G->LeftEye_Nose_distance 
					  + G->RightEye_Nose_distance 
					  + G->LeftEye_RightEye_distance 
					  + G->Nose_Mouth_distance)/4;

	scale = (float)G->Mean_Feature_distance/ (float)G->init_Mean_Feature_distance;

	float Image_Facial_Normal_length = FindDistance2D32f(F->NoseBase, F->Nose);			//distance between Nose base and Nose center
	float Eye_Mouth_distance = FindDistance2D32f(F->MidEyes, F->Mouth);				//distance between Eye center and Mouth center

	/*P->roll = FindAngle(F->LeftEye, F->RightEye);						//roll angle - angle between left Eye and right Eye
	if (P->roll > 180){
		P->roll = P->roll-360;
	}
	roll[frame_number] = P->roll;*/

	P->roll = FindAngle(F->LeftEye, F->RightEye);						//roll angle - angle between left Eye and right Eye
	if (P->roll > 180){
		P->roll = P->roll-360;
	}
	//roll[frame_number] = P->roll;

	float symm = FindAngle(F->NoseBase, F->MidEyes);									//symm angle - angle between the symmetry axis and the 'x' axis 
	float tilt = FindAngle(F->NoseBase, F->Nose);									//tilt angle - angle between normal in image and 'x' axis
	float tita = (abs(tilt-symm))*(pi/180);											//tita angle - angle between the symmetry axis and the image normal
	
	P->slant = Find_slant(Image_Facial_Normal_length, Eye_Mouth_distance, R_n, tita); //slant angle - angle between the facial normal and the image normal
	
	//define a 3D vector for the facial normal
	CvPoint3D32f normal;
	normal.x = (sin(P->slant))*(cos((360-tilt)*(pi/180)));
	normal.y = (sin(P->slant))*(sin((360-tilt)*(pi/180)));
	normal.z = -cos(P->slant);

	//find pitch and yaw
	P->kpitch_pre = P->pitch;
	P->pitch = acos(sqrt((normal.x*normal.x + normal.z*normal.z)/(normal.x*normal.x + normal.y*normal.y + normal.z*normal.z)));
	if((F->Nose.y - F->NoseBase.y)< 0 ){
		P->pitch = P->pitch*(-1);
	}
	//pitch[frame_number] = P->pitch*(180/pi);

	P->kyaw_pre = P->yaw;
	P->yaw = acos((abs(normal.z))/(sqrt(normal.x*normal.x + normal.z*normal.z)));
	if((F->Nose.x - F->NoseBase.x)< 0 ){
		P->yaw = P->yaw*(-1);
	}
	//yaw[frame_number] = P->yaw*(180/pi);

//////////////////////////Kalman Filter Code///////////////////////////////

	//Prediction Step
	const CvMat* y_k = cvKalmanPredict( kalman, 0 );  	// Predict point position

	//Take Measurement
	//z_k = H_k * x_k + v_k; In this case z_k = x_k
	//measurements are already noisy
	//z_k and x_k have equal dimentions and represent the same variables. 

	cvRandSetRange( &rng, 0, sqrt( kalman->measurement_noise_cov->data.fl[0] ), 0 ); //change random number generator mean value to that of measurement error v_k (in this case not needed).
//	double error_noise = 	(1 + (int)( 50.0 * rand() / ( RAND_MAX + 1.0 ) ))*0.01;
//	cvSetIdentity( kalman->measurement_noise_cov, cvRealScalar(5e-2 + error_noise) );

	cvmSet(x_k, 0,0, P->pitch);
	cvmSet(x_k, 1,0, (P->pitch - P->kpitch_pre)   ); //velocity = X pixels/frame ?????
	cvmSet(x_k, 2,0, P->yaw);
	cvmSet(x_k, 3,0, (P->yaw - P->kyaw_pre)   );

/*	
	cvmSet(x_k, 0,0, P->roll);
	cvmSet(x_k, 1,0, P->pitch);
	cvmSet(x_k, 2,0, P->yaw);
	cvmSet(x_k, 3,0, Nose_base.x);
	cvmSet(x_k, 4,0, Nose_base.y);
	cvmSet(x_k, 5,0, scale);
*/
/*	
	cvmSet(z_k, 0,0, P->roll);
	cvmSet(z_k, 1,0, P->pitch);
	cvmSet(z_k, 2,0, P->yaw);
	cvmSet(z_k, 3,0, Nose_base.x);
	cvmSet(z_k, 4,0, Nose_base.y);
	cvmSet(z_k, 5,0, scale);
	*/

	//void cvMatMulAdd( const CvArr* A, const CvArr* B, const CvArr* C, CvArr* D ); [D=A*B+C]
	//cvMatMulAdd( kalman->measurement_matrix, x_k, z_k, z_k);
	//cvMatMulAdd( kalman->measurement_matrix, x_k, 0, z_k);

	//Correction Step
	cvKalmanCorrect(kalman, x_k);



	// Apply the transition matrix F and apply "process noise" w_k
	// x_k = F * x_(k-1) + w_k
		cvRandSetRange( &rng, 0, sqrt( kalman->process_noise_cov->data.fl[0] ), 0 ); //change random number generator mean value to that of measurement error w_k (in this case not needed).
		cvRand( &rng, w_k );
		cvMatMulAdd( kalman->transition_matrix, x_k, w_k, x_k );

///////////////////////////////////////////////////////////////////////////


	//CvPoint img_centre = cvPoint(cvRound(W/2),cvRound(H/2));
	
	pointer_2d.x = ((F->NoseBase.x + cvRound(500*(tan((double)P->yaw))))*0.7) + (pointer_2d.x)*0.3;
	pointer_2d.y = ((F->NoseBase.y + cvRound(500*(tan((double)P->pitch))))*0.7) + (pointer_2d.y)*0.3;

	///////////
	P->kyaw = cvmGet(y_k, 2,0 );
	P->kpitch = cvmGet(y_k, 0,0);
	pointer_2d_kalman.x = ((F->NoseBase.x + cvRound(500*(tan((double)P->kyaw))))) ;
	pointer_2d_kalman.y = ((F->NoseBase.y + cvRound(500*(tan((double)P->kpitch))))) ;
	/////////////////



	//draw lines
	//cvPoint(F->LeftEye.x,F->LeftEye.y)
	cvLine(img, cvPoint(F->Nose.x,F->Nose.y),		cvPoint(F->Mouth.x,F->Mouth.y),		CV_RGB(255,0,0), 1, 4, 0);
	cvLine(img, cvPoint(F->Nose.x,F->Nose.y),		cvPoint(F->LeftEye.x,F->LeftEye.y),	CV_RGB(255,0,0), 1, 4, 0);
	cvLine(img, cvPoint(F->Nose.x,F->Nose.y),		cvPoint(F->RightEye.x,F->RightEye.y),	CV_RGB(255,0,0), 1, 4, 0);
	cvLine(img, cvPoint(F->RightEye.x,F->RightEye.y),	cvPoint(F->LeftEye.x,F->LeftEye.y),	CV_RGB(0,0,255), 1, 4, 0);
	cvLine(img, cvPoint(F->MidEyes.x,F->MidEyes.y),		cvPoint(F->Mouth.x,F->Mouth.y),		CV_RGB(0,0,255), 1, 4, 0);
	cvLine(img, cvPoint(F->NoseBase.x,F->NoseBase.y),	cvPoint(F->Nose.x,F->Nose.y),		CV_RGB(0,0,255), 1, 4, 0);	
	cvLine(img, cvPoint(F->Nose.x,F->Nose.y),		cvPoint(F->MidEyes.x,F->MidEyes.y),	CV_RGB(255,0,0), 1, 4, 0);



	//void draw_crosshair(IplImage* img, CvPoint centre, int circle_radius, int line_radius, CvScalar colour)
	draw_crosshair(img, pointer_2d_kalman, 7, 12, CV_RGB(255,0,0));
      
	//draw covariance error ellipse
	//printf("after  cvKalmanCorrect: kalman->error_cov_pre(2x2)  %1.6f %1.6f %1.6f %1.6f\n ", kalman->error_cov_pre->data.fl[0], kalman->error_cov_pre->data.fl[1],kalman->error_cov_pre->data.fl[2], kalman->error_cov_pre->data.fl[3]);
	//printf("after  cvKalmanCorrect: kalman->error_cov_pre(2x2)  %1.6f %1.6f %1.6f %1.6f\n ", kalman->error_cov_pre->data.fl[4], kalman->error_cov_pre->data.fl[5],kalman->error_cov_pre->data.fl[6], kalman->error_cov_pre->data.fl[7]);
	//printf("after  cvKalmanCorrect: kalman->error_cov_pre(2x2)  %1.6f %1.6f %1.6f %1.6f\n ", kalman->error_cov_pre->data.fl[8], kalman->error_cov_pre->data.fl[9],kalman->error_cov_pre->data.fl[10], kalman->error_cov_pre->data.fl[11]);
	//printf("after  cvKalmanCorrect: kalman->error_cov_pre(2x2)  %1.6f %1.6f %1.6f %1.6f\n ", kalman->error_cov_pre->data.fl[12], kalman->error_cov_pre->data.fl[13],kalman->error_cov_pre->data.fl[14], kalman->error_cov_pre->data.fl[15]);

	double muX = 2*sqrt(kalman->error_cov_post->data.fl[0]);
        double muY = 2*sqrt(kalman->error_cov_post->data.fl[10]);
	//printf("                                                     %1.5f %1.5f\n",  muX, muY);

        static int theta_ellipse = 0;
        cvEllipse( img, pointer_2d_kalman, cvSize( muX , muY ), theta_ellipse, 0, 360, CV_RGB(0,255,0),1);

	cvCircle(img, pointer_2d, 7, CV_RGB(80,80,80), 2, 4, 0); //draw unfiltered estimate
	
	//void play_game(IplImage* img, int precision, CvPoint Position, CvScalar ball_colour)
	play_game(img, 8, pointer_2d_kalman, CV_RGB(0,0,200));

	//void draw_trail(IplImage* img, CvPoint pt)
	//draw_trail(img, &pointer_2d_kalman);
	


//1280x800
//	mouse.x = 1280/2 - 320/2 + ((Nose_base.x + cvRound(1500*(tan((double)P->yaw))))*0.7) + (mouse.x)*0.3;
//	mouse.y = 800/2 - 240/2 + ((Nose_base.y + cvRound(1500*(tan((double)P->pitch))))*0.7) + (mouse.y)*0.3;
//SetCursorPos(mouse.x, mouse.y);



	//void draw_pin(IplImage* img, CvPoint3D32f normal, float slant, float tita, CvScalar colour)	
	draw_pin(img, normal, P->slant, tita, CV_RGB(255,0,0));

	print_text(img, t, CV_RGB(255,0,0));
/*
	printf("LeftEye_Nose_distance		= %.3f\n",	LeftEye_Nose_distance);
	printf("RightEye_Nose_distance		= %.3f\n",	RightEye_Nose_distance);
	printf("LeftEye_RightEye_distance	= %.3f\n",	LeftEye_RightEye_distance);
	printf("Mouth_Nose_distance		= %.3f\n\n",	Nose_Mouth_distance);

	printf("Scale		= %.3f\n\n",	scale);
	
	printf("Tilt		= %.3f\n",	tilt);
	printf("symm angle	= %.3f\n",	symm);
	printf("Tita		= %.3f\n",	tita*(180/pi));
	printf("Slant		= %.3f\n\n",	P->slant*(180/pi));

	printf("Roll		= %.3f\n",	P->roll);
	printf("Pitch		= %.3f\n",	P->pitch*(180/pi));
	printf("Yaw		= %.3f\n\n",	P->yaw*(180/pi));

	printf("normal.x	= %.3f\n",	normal.x);
	printf("normal.y	= %.3f\n",	normal.y);
	printf("normal.z	= %.3f\n\n",	normal.z);
*/
	//determine tracking failure
	if((G->LeftEye_RightEye_distance > G->init_LeftEye_RightEye_distance*1.4
	&& G->LeftEye_Nose_distance 	> G->init_LeftEye_Nose_distance*1.4
	&& G->RightEye_Nose_distance 	> G->init_RightEye_Nose_distance*1.4
	&& G->Nose_Mouth_distance 	> G->init_Nose_Mouth_distance*2)

	|| 
	(G->LeftEye_RightEye_distance 	< G->init_LeftEye_RightEye_distance*0.7
	&& G->LeftEye_Nose_distance 	< G->init_LeftEye_Nose_distance*0.7
	&& G->RightEye_Nose_distance 	< G->init_RightEye_Nose_distance*0.7
	&& G->Nose_Mouth_distance 	< G->init_Nose_Mouth_distance*0.7)

	||
	(G->LeftEye_RightEye_distance 	> G->init_LeftEye_RightEye_distance*1.3
	&& F->Nose.y 	<= F->MidEyes.y + 10
	&& F->Mouth.y 	<= F->Nose.y + 10
	)
	){

		is_tracking = 0;
	}

}

//Function to return distance between 2 points in image
float FindDistance(CvPoint pt1, CvPoint pt2){
	int x,y;
	double z;
	x = pt1.x - pt2.x;
	y = pt1.y - pt2.y;
	z = (x*x)+(y*y);

	return (float)sqrt(z);
}

//Function to return distance between 2 points in image
double FindDistance2D32f(CvPoint2D32f pt1, CvPoint2D32f pt2){
	double x,y,z;
	
	x = pt1.x - pt2.x;
	y = pt1.y - pt2.y;
	z = (x*x)+(y*y);

	return sqrt(z);
}

//unctio to return angle between 2 points in image
float FindAngle(CvPoint2D32f pt1, CvPoint2D32f pt2)
	{

	float angle;
	angle = cvFastArctan(pt2.y - pt1.y, pt2.x - pt1.x);

	return 360-angle;
	}


//Function to find slant angle in image 'Gee & Cipolla'
double Find_slant(int ln, int lf, float Rn, float tita)
{
	float dz=0;
	double slant;
	float m1 = ((float)ln*ln)/((float)lf*lf);
	float m2 = (cos(tita))*(cos(tita));

	//printf("ln = %d\n", ln);
	//printf("lf = %d\n", lf);
	//printf("m1 = %.4f\n", m1);
	//printf("m2 = %.4f\n", m2);

	if (m2 == 1)
	{
		 dz = sqrt(	(Rn*Rn)/(m1 + (Rn*Rn))	);
	}
	if (m2>=0 && m2<1)
	{
		 dz = sqrt(	((Rn*Rn) - m1 - 2*m2*(Rn*Rn) + sqrt(	((m1-(Rn*Rn))*(m1-(Rn*Rn))) + 4*m1*m2*(Rn*Rn)	))/ (2*(1-m2)*(Rn*Rn))	);
	}
	slant = acos(dz);
	return slant;


}

void draw_crosshair(IplImage* img, CvPoint centre, int circle_radius, int line_radius, CvScalar colour)
{
	CvPoint pt1,pt2,pt3,pt4;

	pt1.x = centre.x;
	pt2.x = centre.x;
	pt1.y = centre.y - line_radius;
	pt2.y = centre.y + line_radius;
	pt3.x = centre.x - line_radius;
	pt4.x = centre.x + line_radius;
	pt3.y = centre.y;
	pt4.y = centre.y;


	cvCircle(img, centre, circle_radius, colour, 2, 4, 0);

	cvLine(img, pt1, pt2, colour, 1, 4, 0);
	cvLine(img, pt3, pt4, colour, 1, 4, 0);

}


void play_game(IplImage* img, int precision, CvPoint Position, CvScalar ball_colour)
{
	static int counter;

		if(	pointer_2d_kalman.x < rand_coord.x + precision && pointer_2d_kalman.x > rand_coord.x - precision
		&& pointer_2d_kalman.y < rand_coord.y + precision && pointer_2d_kalman.y > rand_coord.y - precision
	){
	//if(	pointer_2d.x < rand_coord.x + 12 && pointer_2d.x > rand_coord.x - 12 		
//		&& pointer_2d.y < rand_coord.y +12 && pointer_2d.y > rand_coord.y - 12
//	){
	
	counter++;

	if (counter == 10){
	rand_coord.x = 30 + ( rand() % (W - 60));
	rand_coord.y = 30 + ( rand() % (H - 60));
	t++;
	counter = 0;

   
	//PlaySound(L"c://magnum.wav", NULL, SND_FILENAME | SND_ASYNC);
	//QSound::play("mysounds/bells.wav");
	int status = system("play sounds/44magnum.wav &");
  if(status < 0) std::cout << "Warning: something wrong with playing of sound";
	}
	}
	cvCircle(img, rand_coord, 5, ball_colour, 6, 4, 0);
}


void draw_pin(IplImage* img, CvPoint3D32f normal, float slant, float tita, CvScalar colour)
{
	//define a 2d origin point for the view pointer
	const CvPoint origin = cvPoint(50,50);

	//define a 2d point for the projected 3D vector in 2d
	CvPoint projection_2d;
	projection_2d.x = origin.x + cvRound(60*(normal.x));
	projection_2d.y = origin.y + cvRound(60*(normal.y));

	//draw pin circle
	if (normal.x > 0 && normal.y < 0)
	{
	cvEllipse(img, origin, cvSize(25,abs(cvRound(25-slant*(180/(2*pi))))) , abs(180-(tita*(180/pi))), 0, 360, colour, 2,4,0);
	}
	else
	{
	cvEllipse(img, origin, cvSize(25,abs(cvRound(25-slant*(180/(2*pi))))) , abs(tita*(180/pi)), 0, 360, colour, 2,4,0);
	}

	//draw pin head
	cvLine(img, origin,			projection_2d,		CV_RGB(255,0,0), 2, 4, 0);
}

void print_text(IplImage* img, int counter, CvScalar colour)
{
	//routines for printing text on image 'looking left or right'
	CvFont font;
	
	char msg1[4];
	sprintf(msg1, "%d", counter);
	//char msg2[] = "Left";

	const CvPoint text1 = cvPoint(5,200);

	cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 1,1,0,1,8);

	cvPutText(img, msg1, text1, &font, colour);




}

void draw_trail(IplImage* img, CvPoint* pt)
{
	static int counter;
	const int length = 30;

  static CvPoint p_1 = cvPoint(0,0);
  static CvPoint p = cvPoint(0,0);
  static float dist;
  CvPoint* p1,*p2;


  p 	= cvPoint(pt->x,pt->y);
  dist = FindDistance(p_1,p);

  printf("dist = %f\n",dist);

	  if(dist > 20){
	p_1	= cvPoint(p.x,p.y);
	cvSeqPush(seq, pt); 

	if (counter < length )
	counter ++;

	if (counter >= length){
	cvSeqPopFront(seq, NULL);
	}

	}

	for ( int i=0; i<(seq->total)-1;++i){
	  //CvPoint* p = (CvPoint*)cvGetSeqElem (seq, i);
	   p1 = (CvPoint*)cvGetSeqElem (seq, i);
	   p2 = (CvPoint*)cvGetSeqElem (seq, i+1);
	  //cvCircle(img, *p1, 1, CV_RGB(255,0,0), 1, 4, 0);
	  cvLine(img, *p1, *p2, CV_RGB(0,0,0), 1, 4, 0);	
	}
    

}


void closeDraw()
{
	cvReleaseMemStorage( &storage );

	return;
}





	//draw eye box
	/*cvEllipse( img, cvPoint(35,90), cvSize(15,10), 0, 0, 360, CV_RGB(255,0,0),2, 4, 0);

	cvCircle ( img, cvPoint( cvRound(20+ldx*1.5), cvRound(90+ldy*1.5)), 3, cvScalar( 0, 0, 255, 0), 2, 4, 0);

	cvEllipse( img, cvPoint(65,90), cvSize(15,10), 0, 0, 360, CV_RGB(255,0,0),2, 4, 0);

	cvCircle ( img, cvPoint( cvRound(50+ldx*1.5), cvRound(90+ldy*1.5)), 3, cvScalar( 0, 0, 255, 0), 2, 4, 0);*/
	//cvRectangle( img, cvPoint( 40, 80 ), cvPoint( 70, 95), cvScalar( 0, 0, 255, 0 ), 1, 0, 0 );
	//draw eye point w.r.t eye corner

	/*if(40+ldx*1.5>65){
	cvCircle (img, cvPoint(cvRound(40+ldx*1.5), cvRound(87.5+ldy*1.5)), 3, cvScalar( 0, 0, 255, 0), 2, 8, 0);
	}
	else if(80-rdx*1.5<65){
	cvCircle (img, cvPoint(cvRound(70-rdx*1.5), cvRound(87.5+ldy*1.5)), 3, cvScalar( 0, 0, 255, 0), 2, 8, 0);
	}
	else{
	cvCircle (img, cvPoint(cvRound(40+(ldx)*1.5), cvRound(87.5+(ldy)*1.5)), 3, cvScalar( 0, 0, 255, 0), 2, 8, 0);
	}*/
