#include <stdio.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc_c.h>

#include "facefeaturetrack.h"
#include "pose-estimation.h"

//camera resolution
extern int W;
extern int H;

#define  EYE_TPL_WIDTH		 12			// eye template width   
#define  EYE_TPL_HEIGHT      	 10			// eye template height      
#define  EYE_WINDOW_WIDTH	 40			// eye search window width  
#define  EYE_WINDOW_HEIGHT	 40			// eye search window height 

#define  NOSE_TPL_WIDTH       20		// nose template width       
#define  NOSE_TPL_HEIGHT      20		// nose template height      
#define  NOSE_WINDOW_WIDTH    40		// nose search window width  
#define  NOSE_WINDOW_HEIGHT   40		// nose search window height

#define  MOUTH_TPL_WIDTH		20		// mouth template width       
#define  MOUTH_TPL_HEIGHT		14		// mouth template height      
#define  MOUTH_WINDOW_WIDTH		40		// mouth search window width  
#define  MOUTH_WINDOW_HEIGHT	40		// mouth search window height 

#define  THRESHOLD       0.3         //threshold defining the maximum amount of error to consider a good match
#define  TH_EYE			 0.3


struct feature {
      
      IplImage * tpl;	// create template image
      IplImage * tpl_2;	// create template image

      IplImage * tm;	// create images to store template matching result
      IplImage * tm_2;	// create images to store template matching result

      IplImage * roi;	// region of interest image
      int	x0;
      int	y0;
      int	win_x0;
      int 	win_y0;
      double	minval;
      double	maxval;
      CvPoint 	minloc;
      CvPoint	maxloc;

      int MATCH_METHOD;

};

typedef struct feature Feature;

extern CvPoint LeftEye_center;
extern CvPoint RightEye_center;
extern CvPoint Eye_center;
extern CvPoint Nose_center;
extern CvPoint Nose_base;
extern CvPoint Mouth_center;

extern float init_LeftEye_Nose_distance;
extern float init_RightEye_Nose_distance;
extern float init_LeftEye_RightEye_distance;
extern float init_Nose_Mouth_distance;

extern float LeftEye_Nose_distance;
extern float RightEye_Nose_distance;
extern float LeftEye_RightEye_distance;
extern float Nose_Mouth_distance;

extern float scale;

extern float Roll;
extern float slant;
extern float Yaw;

CvPoint LeftEye_center_corner;
CvPoint RightEye_center_corner;

IplImage  *Iat,*gray, *mask;
IplImage *frame;

int is_tracking=0;


//variables for 3D model
extern float  R_m;
extern float  R_n;
extern float  R_e;

//initialisations of confidence levels
float C_leye = 0.5;
float C_reye = 0.5;
float C_nose = 0.01;
float C_mouth = 0.01;

Feature Mouth;
Feature * MouthPtr = &Mouth;

Feature Nose;
Feature * NosePtr = &Nose;

Feature LeftEye;
Feature * LeftEyePtr = &LeftEye;

Feature RightEye;
Feature * RightEyePtr = &RightEye;

static char window_name[12][255] = {{NULL},{NULL}};


// Function to free memory when program terminates
void closeTemplateMatch()
{

	cvReleaseImage( &Mouth.tpl );
	cvReleaseImage( &Mouth.tpl_2 );
	cvReleaseImage( &Mouth.tm );
	cvReleaseImage( &Mouth.tm_2 );
	cvReleaseImage( &Mouth.roi );

	cvReleaseImage( &Nose.tpl );
	cvReleaseImage( &Nose.tpl_2 );
	cvReleaseImage( &Nose.tm );
	cvReleaseImage( &Nose.tm_2 );
	cvReleaseImage( &Nose.roi );

	cvReleaseImage( &LeftEye.tpl );
	cvReleaseImage( &LeftEye.tpl_2 );
	cvReleaseImage( &LeftEye.tm );
	cvReleaseImage( &LeftEye.tm_2 );
	cvReleaseImage( &LeftEye.roi );

	cvReleaseImage( &RightEye.tpl );
	cvReleaseImage( &RightEye.tpl_2 );
	cvReleaseImage( &RightEye.tm );
	cvReleaseImage( &RightEye.tm_2 );
	cvReleaseImage( &RightEye.roi );

	cvReleaseImage( &Iat );
	cvReleaseImage( &gray );
	cvReleaseImage( &mask );

}


void match_feature(IplImage * frame, Feature * f, const int W, const int H, int METHOD)
{

double minval_2 = 0;
double maxval_2 = 0;

CvPoint	minloc_2;
CvPoint	maxloc_2;

    cvSetImageROI( frame, 
                   cvRect( f->win_x0, 
                           f->win_y0, 
                           W, 
                           H ) );

    cvMatchTemplate( frame, f->tpl, f->tm, METHOD );
    
    cvMinMaxLoc( f->tm, &f->minval, &f->maxval, &f->minloc, &f->maxloc, 0 );


//scale invariance test code
/*
    cvMatchTemplate( frame, f->tpl_2, f->tm_2, CV_TM_SQDIFF_NORMED );
    cvMinMaxLoc( f->tm_2, &minval_2, &maxval_2, &minloc_2, &maxloc_2, 0 );

    if(minval_2 < f->minval){
     f->minval = minval_2;
     f->minloc.x = minloc_2.x;
     f->minloc.y = minloc_2.y;
      printf("WORKING!!\n");
    }
*/
    cvResetImageROI( frame );
	//printf("Min value Mouth_3		= %.3f\n", f->minval);

}

void display_roi(IplImage * frame, Feature * f, const int W, const int H, char window_name[255])
{

	cvSetImageROI( frame, 
                       cvRect( f->win_x0, 
                               f->win_y0, 
                               W, 
                               H ) );
        cvCopy( frame, f->roi, NULL );
	cvResetImageROI( frame );
		
	cvShowImage(window_name, f->roi);//print left eye roi
	

}

void display_tmr(IplImage * frame, Feature * f, char window_name[255])
{
		cvNormalize(f->tm,f->tm,1,0,CV_MINMAX);
		cvShowImage( window_name, f->tm);
}

void update_and_draw(IplImage * frame, Feature * f, CvPoint * feature_center,const int W, const int H, int METHOD)
{

if(METHOD == CV_TM_SQDIFF_NORMED){
        // save current location
        f->x0 = f->win_x0 + f->minloc.x;
        f->y0 = f->win_y0 + f->minloc.y;
}
if(METHOD == CV_TM_CCOEFF_NORMED || CV_TM_CCORR_NORMED){
        // save current location
        f->x0 = f->win_x0 + f->maxloc.x;
        f->y0 = f->win_y0 + f->maxloc.y;
}

		//Find the center position of the feature
		feature_center->x = f->x0 + ( W  / 2 );
		feature_center->y = f->y0 + ( H / 2 );

        //Draw rectangle around feature
        cvRectangle( frame,
                     cvPoint( f->x0, f->y0 ),
                     cvPoint( f->x0 + W, f->y0 + H ),
                     cvScalar( 0, 0, 10, 0 ), 1, 0, 0 );
}

void init_tpl(IplImage * frame, Feature * f, CvPoint f_init_pos, const int W, const int H, char window_name[255])
{
	f->x0 = (f_init_pos.x - ( W  / 2 ));
        f->y0 = (f_init_pos.y - ( H / 2 ));

	cvSetImageROI( frame, 
                       cvRect( f->x0, 
                               f->y0, 
                               W, 
                               H ) );
        cvCopy( frame, f->tpl, NULL );
	cvResize(frame, f->tpl_2, CV_INTER_LINEAR);
	cvResetImageROI( frame );

//		cvNamedWindow(window_name, CV_WINDOW_AUTOSIZE);
		cvShowImage(window_name, f->tpl);

	f->minval = 0;
	f->maxval = 0;

	f->MATCH_METHOD = CV_TM_CCOEFF_NORMED;
	//f->MATCH_METHOD = CV_TM_SQDIFF_NORMED;
	  //f->MATCH_METHOD = CV_TM_CCORR_NORMED;
}

/*
int dynamicTracker(IplImage* frame, CvPoint L, CvPoint R, CvPoint N, CvPoint M )
{
    L_tpl_d = cvCreateImage( cvSize( EYE_TPL_WIDTH, EYE_TPL_HEIGHT ), 
                         frame->depth, frame->nChannels );

    // create images to store template matching result
    L_tm_d = cvCreateImage( cvSize( EYE_WINDOW_WIDTH  - EYE_TPL_WIDTH  + 1,
                                EYE_WINDOW_HEIGHT - EYE_TPL_HEIGHT + 1 ),
                        IPL_DEPTH_32F,
						//8,
						1 );
		
		//Find the coordinates of the box enclosing the located features
		LeftEye_x0_d = (L.x - ( EYE_TPL_WIDTH  / 2 ));
        LeftEye_y0_d = (L.y - ( EYE_TPL_HEIGHT / 2 )); 

		//Copy region of interest defined by the box to template image and display ROI
		cvSetImageROI( frame, 
                       cvRect( LeftEye_x0_d, 
                               LeftEye_y0_d, 
                               EYE_TPL_WIDTH, 
                               EYE_TPL_HEIGHT ) );
        cvCopy( frame, L_tpl_d, NULL );
		cvResetImageROI( frame );
		cvNamedWindow("Left Eye dynamic", CV_WINDOW_AUTOSIZE);
		cvShowImage( "Left Eye dynamic", L_tpl_d);

	
        printf("All templates sucessfuly selected, start tracking.. \n" );

        is_tracking = 1; //signifies successful initialisation

		return is_tracking;

}
*/
int initTracker(IplImage* frame, CvPoint L, CvPoint R, CvPoint N, CvPoint M )
{
const int plus = -5;


	//Initialise template images, roi images
	LeftEye.tpl = cvCreateImage( cvSize( EYE_TPL_WIDTH, EYE_TPL_HEIGHT), 
                         frame->depth, frame->nChannels );
	LeftEye.tpl_2 = cvCreateImage( cvSize( EYE_TPL_WIDTH + plus, EYE_TPL_HEIGHT + plus ), 
                         frame->depth, frame->nChannels );
	LeftEye.roi = cvCreateImage( cvSize( EYE_WINDOW_WIDTH, EYE_WINDOW_HEIGHT), 
                         frame->depth, frame->nChannels );


	RightEye.tpl = cvCreateImage( cvSize( EYE_TPL_WIDTH, EYE_TPL_HEIGHT ), 
                         frame->depth, frame->nChannels );  
	RightEye.tpl_2 = cvCreateImage( cvSize( EYE_TPL_WIDTH + plus, EYE_TPL_HEIGHT + plus), 
                         frame->depth, frame->nChannels ); 
	RightEye.roi = cvCreateImage( cvSize( EYE_WINDOW_WIDTH, EYE_WINDOW_HEIGHT ), 
                         frame->depth, frame->nChannels );  

	Nose.tpl = cvCreateImage( cvSize( NOSE_TPL_WIDTH, NOSE_TPL_HEIGHT ), 
                         frame->depth, frame->nChannels ); 
	Nose.tpl_2 = cvCreateImage( cvSize( NOSE_TPL_WIDTH + plus, NOSE_TPL_HEIGHT + plus), 
                         frame->depth, frame->nChannels ); 
	Nose.roi = cvCreateImage( cvSize( NOSE_WINDOW_WIDTH, NOSE_WINDOW_HEIGHT ), 
                         frame->depth, frame->nChannels ); 

	Mouth.tpl = cvCreateImage( cvSize( MOUTH_TPL_WIDTH, MOUTH_TPL_HEIGHT ), 
                         frame->depth, frame->nChannels ); 
	Mouth.tpl_2 = cvCreateImage( cvSize( MOUTH_TPL_WIDTH + plus, MOUTH_TPL_HEIGHT + plus), 
                         frame->depth, frame->nChannels ); 
	Mouth.roi = cvCreateImage( cvSize( MOUTH_WINDOW_WIDTH, MOUTH_WINDOW_HEIGHT ), 
                         frame->depth, frame->nChannels ); 


	Iat =	cvCreateImage(cvSize(EYE_WINDOW_WIDTH,EYE_WINDOW_HEIGHT),
                          IPL_DEPTH_8U, 1);

	gray =	cvCreateImage(cvSize(EYE_WINDOW_WIDTH,EYE_WINDOW_HEIGHT),
                          IPL_DEPTH_8U, 1);

	mask = cvCreateImage( cvSize( EYE_WINDOW_WIDTH  - EYE_TPL_WIDTH  + 1,
                                EYE_WINDOW_HEIGHT - EYE_TPL_HEIGHT + 1 ),
                        IPL_DEPTH_8U,
						1 );


	// Initialise images to store template matching result
	LeftEye.tm = cvCreateImage( cvSize( EYE_WINDOW_WIDTH  - EYE_TPL_WIDTH  + 1,
                                EYE_WINDOW_HEIGHT - EYE_TPL_HEIGHT + 1 ),
                        IPL_DEPTH_32F,
						//8,
						1 );

	LeftEye.tm_2 = cvCreateImage( cvSize( EYE_WINDOW_WIDTH  - EYE_TPL_WIDTH  + 1 - plus,
                                EYE_WINDOW_HEIGHT - EYE_TPL_HEIGHT + 1 - plus),
                        IPL_DEPTH_32F,
						//8,
						1 );

	RightEye.tm = cvCreateImage( cvSize( EYE_WINDOW_WIDTH  - EYE_TPL_WIDTH  + 1,
                                EYE_WINDOW_HEIGHT - EYE_TPL_HEIGHT + 1 ),
                        IPL_DEPTH_32F,
						//8,
						1 );
	RightEye.tm_2 = cvCreateImage( cvSize( EYE_WINDOW_WIDTH  - EYE_TPL_WIDTH  + 1 - plus,
                                EYE_WINDOW_HEIGHT - EYE_TPL_HEIGHT + 1 - plus),
                        IPL_DEPTH_32F,
						//8,
						1 );

	Nose.tm = cvCreateImage( cvSize( NOSE_WINDOW_WIDTH  - NOSE_TPL_WIDTH  + 1,
                                NOSE_WINDOW_HEIGHT - NOSE_TPL_HEIGHT + 1 ),
                        IPL_DEPTH_32F,
						//8,
						1 );
	Nose.tm_2 = cvCreateImage( cvSize( NOSE_WINDOW_WIDTH  - NOSE_TPL_WIDTH  + 1 - plus,
                                NOSE_WINDOW_HEIGHT - NOSE_TPL_HEIGHT + 1 - plus),
                        IPL_DEPTH_32F,
						//8,
						1 );

	Mouth.tm = cvCreateImage( cvSize( MOUTH_WINDOW_WIDTH  - MOUTH_TPL_WIDTH  + 1,
                                MOUTH_WINDOW_HEIGHT - MOUTH_TPL_HEIGHT + 1 ),
                        IPL_DEPTH_32F,
						//8,
						1 );
	Mouth.tm_2 = cvCreateImage( cvSize( MOUTH_WINDOW_WIDTH  - MOUTH_TPL_WIDTH  + 1 - plus,
                                MOUTH_WINDOW_HEIGHT - MOUTH_TPL_HEIGHT + 1 - plus),
                        IPL_DEPTH_32F,
						//8,
						1 );



	//init_window;
	int screen_w = 1280;
	sprintf ( & ( window_name[0][0] ), "Left_Eye" );
	sprintf ( & ( window_name[1][0] ), "Right_Eye" );
	sprintf ( & ( window_name[2][0] ), "Nose" );
	sprintf ( & ( window_name[3][0] ), "Mouth" );

 	cvNamedWindow ( window_name[0], CV_WINDOW_AUTOSIZE);
	cvNamedWindow ( window_name[1], CV_WINDOW_AUTOSIZE);
	cvNamedWindow ( window_name[2], CV_WINDOW_AUTOSIZE);
	cvNamedWindow ( window_name[3], CV_WINDOW_AUTOSIZE);

	cvMoveWindow ( window_name[0], cvRound(screen_w /2) - cvRound(W/2) - 50,	20);
	cvMoveWindow ( window_name[1], cvRound(screen_w /2) - cvRound(W/2) - 50 , 	120);
	cvMoveWindow ( window_name[2], cvRound(screen_w /2) - cvRound(W/2) - 50,	220);
	cvMoveWindow ( window_name[3], cvRound(screen_w /2) - cvRound(W/2) - 50, 	320);


	sprintf ( & ( window_name[4][0] ), "Left_Eye ROI" );
	sprintf ( & ( window_name[5][0] ), "Right_Eye ROI" );
	sprintf ( & ( window_name[6][0] ), "Nose ROI" );
	sprintf ( & ( window_name[7][0] ), "Mouth ROI" );

 	cvNamedWindow ( window_name[4], CV_WINDOW_AUTOSIZE);
	cvNamedWindow ( window_name[5], CV_WINDOW_AUTOSIZE);
	cvNamedWindow ( window_name[6], CV_WINDOW_AUTOSIZE);
	cvNamedWindow ( window_name[7], CV_WINDOW_AUTOSIZE);

	cvMoveWindow ( window_name[4], cvRound(screen_w /2) - cvRound(W/2) - 100,	20);
	cvMoveWindow ( window_name[5], cvRound(screen_w /2) - cvRound(W/2) - 100, 	120);
	cvMoveWindow ( window_name[6], cvRound(screen_w /2) - cvRound(W/2) - 100,	220);
	cvMoveWindow ( window_name[7], cvRound(screen_w /2) - cvRound(W/2) - 100, 	320);

	sprintf ( & ( window_name[8][0] ), "Left_Eye TMR" );
	sprintf ( & ( window_name[9][0] ), "Right_Eye TMR" );
	sprintf ( & ( window_name[10][0] ), "Nose TMR" );
	sprintf ( & ( window_name[11][0] ), "Mouth TMR" );

 	cvNamedWindow ( window_name[8], CV_WINDOW_AUTOSIZE);
	cvNamedWindow ( window_name[9], CV_WINDOW_AUTOSIZE);
	cvNamedWindow ( window_name[10], CV_WINDOW_AUTOSIZE);
	cvNamedWindow ( window_name[11], CV_WINDOW_AUTOSIZE);

	cvMoveWindow ( window_name[8], cvRound(screen_w /2) - cvRound(W/2) - 200,	20);
	cvMoveWindow ( window_name[9], cvRound(screen_w /2) - cvRound(W/2) - 200, 	120);
	cvMoveWindow ( window_name[10], cvRound(screen_w /2) - cvRound(W/2) - 200,	220);
	cvMoveWindow ( window_name[11], cvRound(screen_w /2) - cvRound(W/2) - 200, 	320);

	//Initialise template images
	init_tpl(frame, LeftEyePtr, L, EYE_TPL_WIDTH, EYE_TPL_HEIGHT, window_name[0]);
//printf("Arrived here\n");
	init_tpl(frame, RightEyePtr, R, EYE_TPL_WIDTH, EYE_TPL_HEIGHT, window_name[1]);

	init_tpl(frame, NosePtr, N, NOSE_TPL_WIDTH, NOSE_TPL_HEIGHT, window_name[2]);

	init_tpl(frame, MouthPtr, M, MOUTH_TPL_WIDTH, MOUTH_TPL_HEIGHT, window_name[3]);		
		
        printf("All templates sucessfuly selected, start tracking.. \n" );

//use if features should be matched with different methods
/*	LeftEye.MATCH_METHOD = CV_TM_CCOEFF_NORMED;
	RightEye.MATCH_METHOD = CV_TM_CCOEFF_NORMED;
	Nose.MATCH_METHOD = CV_TM_CCOEFF_NORMED;
	Mouth.MATCH_METHOD = CV_TM_CCOEFF_NORMED;
*/

        is_tracking = 1; //signifies successful initialisation

		return is_tracking;

}

int i = 0;
// Function to track the face features
IplImage* trackObject( IplImage* frame)
{


 int flag = 0;

/*if (i<200){

	C_leye = (C_leye + L_minval)/2;
	C_reye = (C_reye + R_minval)/2;
	C_nose = N_minval;
	C_mouth = M_minval;
	i++;
}*/


	//if(L_minval>C_leye*1.5 && R_minval<C_reye  ){
 if((LeftEye_RightEye_distance > init_LeftEye_RightEye_distance*1.1 && LeftEye_Nose_distance > init_LeftEye_Nose_distance*1.1)
	 || (LeftEye_RightEye_distance < init_LeftEye_RightEye_distance*scale*0.8 && LeftEye_Nose_distance < init_LeftEye_Nose_distance*scale*0.8)
	 || (LeftEye_RightEye_distance < init_LeftEye_RightEye_distance*scale*0.6)
	 || (LeftEye_Nose_distance < init_LeftEye_Nose_distance*scale*0.6)
	 ){
		
	
		LeftEye.win_x0 = Nose_center.x - (RightEye_center.x - Nose_center.x) - EYE_WINDOW_WIDTH/2 ;
		LeftEye.win_y0 = RightEye.win_y0;


 }
	else{

	LeftEye.win_x0 = LeftEye.x0 - ( ( EYE_WINDOW_WIDTH  - EYE_TPL_WIDTH  ) / 2 );
	LeftEye.win_y0 = LeftEye.y0 - ( ( EYE_WINDOW_HEIGHT - EYE_TPL_HEIGHT ) / 2 );
	}
	



    //if(R_minval>C_reye*1.5 && L_minval<C_leye   ){
	if((LeftEye_RightEye_distance > init_LeftEye_RightEye_distance*1.1 && RightEye_Nose_distance > init_RightEye_Nose_distance*1.1)
	 || (LeftEye_RightEye_distance < init_LeftEye_RightEye_distance*scale*0.8 && RightEye_Nose_distance < init_RightEye_Nose_distance*scale*0.8)
	 || (LeftEye_RightEye_distance < init_LeftEye_RightEye_distance*scale*0.6)
	 || (RightEye_Nose_distance < init_RightEye_Nose_distance*scale*0.6)
	 ){
		
		RightEye.win_x0 = Nose_center.x + (Nose_center.x - LeftEye_center.x ) - EYE_WINDOW_WIDTH/2;
		RightEye.win_y0 = LeftEye.win_y0;
		
	}
	else{

	RightEye.win_x0 = RightEye.x0 - ( ( EYE_WINDOW_WIDTH  - EYE_TPL_WIDTH  ) / 2 );
	RightEye.win_y0 = RightEye.y0 - ( ( EYE_WINDOW_HEIGHT - EYE_TPL_HEIGHT ) / 2 );
	}

	if(LeftEye_center.x > RightEye_center.x){
	int hold1 = RightEye.win_x0;
	int hold2 = RightEye.win_y0;

		RightEye.win_x0 = LeftEye.win_x0;
		RightEye.win_y0 = LeftEye.win_y0;

		LeftEye.win_x0 = hold1;
		LeftEye.win_y0 = hold2;
	}
/*
	int L_C_win_x0 = L_win_x0 -10;
	int L_C_win_y0 = L_win_y0;

	int R_C_win_x0 = R_win_x0 + 10;
	int R_C_win_y0 = R_win_y0;
*/


		if(((LeftEye_Nose_distance > init_LeftEye_Nose_distance*scale*1.1 && RightEye_Nose_distance > init_RightEye_Nose_distance*scale*1.1)
	 && (Nose_Mouth_distance < init_Nose_Mouth_distance*scale*0.7))
	 || ((LeftEye_Nose_distance < init_LeftEye_Nose_distance*scale*0.8 && RightEye_Nose_distance < init_RightEye_Nose_distance*scale*0.8)
	 && (Nose_Mouth_distance > init_Nose_Mouth_distance*scale*1.2))
	 || (Nose_center.y < Eye_center.y )
	 || (Nose_center.y > Mouth_center.y )	
	 ){
	//flag = 1;
		Nose.win_x0 =	Nose_base.x - NOSE_WINDOW_WIDTH/2;
		Nose.win_y0 =	Nose_base.y - NOSE_WINDOW_HEIGHT/2;
	}
	else{
		//flag = 0;
	Nose.win_x0 = Nose.x0 - ( ( NOSE_WINDOW_WIDTH  - NOSE_TPL_WIDTH  ) / 2 );
	Nose.win_y0 = Nose.y0 - ( ( NOSE_WINDOW_HEIGHT - NOSE_TPL_HEIGHT ) / 2 );
	
	}



	if(	(Nose_Mouth_distance > init_Nose_Mouth_distance*scale*2)
	    
	 || (Nose_Mouth_distance < init_Nose_Mouth_distance*scale*0.5)
	 || (Mouth_center.y < Nose_center.y + 5 )
	 ){

		flag = 1;
	Mouth.win_x0 = Eye_center.x - MOUTH_WINDOW_WIDTH/2;
	Mouth.win_y0 = Eye_center.y + 2*(Nose_center.y - Eye_center.y) - MOUTH_WINDOW_HEIGHT/2;
	}
	else{
		flag = 0;
	Mouth.win_x0 = Mouth.x0 - ( ( MOUTH_WINDOW_WIDTH  - MOUTH_TPL_WIDTH  ) / 2 );
	Mouth.win_y0 = Mouth.y0 - ( ( MOUTH_WINDOW_HEIGHT - MOUTH_TPL_HEIGHT ) / 2 );

	}
	//printf("flag = %d\n", flag);



	
	//Do not let window coordinates exceed the image frame

	LeftEye.win_x0 = LeftEye.win_x0 > 0 ? LeftEye.win_x0 : 0;
	LeftEye.win_y0 = LeftEye.win_y0 > 0 ? LeftEye.win_y0 : 0;

	RightEye.win_x0 = RightEye.win_x0 > 0 ? RightEye.win_x0 : 0;
	RightEye.win_y0 = RightEye.win_y0 > 0 ? RightEye.win_y0 : 0; 

	Nose.win_x0 = Nose.win_x0 > 0 ? Nose.win_x0 : 0;
	Nose.win_y0 = Nose.win_y0 > 0 ? Nose.win_y0 : 0;

	Mouth.win_x0 = Mouth.win_x0 > 0 ? Mouth.win_x0 : 0;
	Mouth.win_y0 = Mouth.win_y0 > 0 ? Mouth.win_y0 : 0;

	LeftEye.win_x0 = LeftEye.win_x0  >= W - EYE_WINDOW_WIDTH ? W - EYE_WINDOW_WIDTH : LeftEye.win_x0;
	LeftEye.win_y0 = LeftEye.win_y0  >= H - EYE_WINDOW_HEIGHT ? H - EYE_WINDOW_HEIGHT : LeftEye.win_y0;
	
	RightEye.win_x0 = RightEye.win_x0  >= W - EYE_WINDOW_WIDTH ? W - EYE_WINDOW_WIDTH : RightEye.win_x0;
	RightEye.win_y0 = RightEye.win_y0  >= H - EYE_WINDOW_HEIGHT ? H - EYE_WINDOW_HEIGHT : RightEye.win_y0;

	Nose.win_x0 = Nose.win_x0  >= W - NOSE_WINDOW_WIDTH ? W - NOSE_WINDOW_WIDTH : Nose.win_x0;
	Nose.win_y0 = Nose.win_y0  >= H - NOSE_WINDOW_HEIGHT ? H - NOSE_WINDOW_HEIGHT : Nose.win_y0;

	Mouth.win_x0 = Mouth.win_x0  >= W - MOUTH_WINDOW_WIDTH ? W - MOUTH_WINDOW_WIDTH : Mouth.win_x0;
	Mouth.win_y0 = Mouth.win_y0  >= H - MOUTH_WINDOW_HEIGHT ? H - MOUTH_WINDOW_HEIGHT : Mouth.win_y0;


//look for the feature in the search window
/*
    cvSetImageROI( frame, 
                   cvRect( L_win_x0, 
                           L_win_y0, 
                           EYE_WINDOW_WIDTH, 
                           EYE_WINDOW_HEIGHT ) );

	//cvCvtColor( frame, gray, CV_BGR2GRAY );
	//cvAdaptiveThreshold(gray, Iat, 255, CV_ADAPTIVE_THRESH_MEAN_C,
    //                      CV_THRESH_BINARY, 9, 15);
	//cvNamedWindow("Adaptive Threshold",CV_WINDOW_AUTOSIZE);
	//cvShowImage("Adaptive Threshold",Iat);

	//cvResize(Iat, mask, 1 );
	//cvNamedWindow("mask",CV_WINDOW_AUTOSIZE);
	//cvShowImage("mask",Iat);


    cvMatchTemplate( frame, L_tpl, L_tm, CV_TM_SQDIFF_NORMED );
	//cvMatchTemplate( frame, L_tpl_d, L_tm_d, CV_TM_SQDIFF_NORMED );

	//cvAddWeighted(L_tm, 0.5, L_tm_d, 0.5, 0, L_tm);
	cvMinMaxLoc( L_tm, &L_minval, &L_maxval, &L_minloc, &L_maxloc, NULL);
   
    cvResetImageROI( frame );
		cvNormalize(L_tm,L_tm,1,0,CV_MINMAX);
		cvNamedWindow("Left Eye TMR", CV_WINDOW_AUTOSIZE);
		cvShowImage( "Left Eye TMR", L_tm);
		printf("Min value L_EYE		= %.3f\n",L_minval);
		printf("Min value C_L_EYE		= %.3f\n",C_leye);
*/

//display image roi's
//void display_roi(IplImage * frame, Feature * f, const int W, const int H, char window_name[255])
display_roi(frame, LeftEyePtr, EYE_WINDOW_WIDTH, EYE_WINDOW_HEIGHT, window_name[4]);

display_roi(frame, RightEyePtr, EYE_WINDOW_WIDTH, EYE_WINDOW_HEIGHT, window_name[5]);

display_roi(frame, NosePtr, NOSE_WINDOW_WIDTH, NOSE_WINDOW_HEIGHT, window_name[6]);

display_roi(frame, MouthPtr, MOUTH_WINDOW_WIDTH, MOUTH_WINDOW_HEIGHT, window_name[7]);



//display image tmr's
//void display_tmr(IplImage * frame, Feature * f, char window_name[255])
display_tmr(frame, LeftEyePtr, window_name[8]);

display_tmr(frame, RightEyePtr, window_name[9]);

display_tmr(frame, NosePtr, window_name[10]);

display_tmr(frame, MouthPtr, window_name[11]);


//void match_feature(IplImage * frame, Feature * f, const int W, const int H, int METHOD)
match_feature(frame, LeftEyePtr, EYE_WINDOW_WIDTH, EYE_WINDOW_HEIGHT, LeftEye.MATCH_METHOD);
/*
cvSetImageROI( frame, 
                   cvRect( LeftEye.win_x0, 
                           LeftEye.win_y0, 
                           EYE_WINDOW_WIDTH, 
                           EYE_WINDOW_HEIGHT ) );

	cvCvtColor( frame, gray, CV_BGR2GRAY );
	cvAdaptiveThreshold(gray, Iat, 255, CV_ADAPTIVE_THRESH_MEAN_C,
                          CV_THRESH_BINARY, 9, 15);
	cvNamedWindow("Adaptive Threshold",CV_WINDOW_AUTOSIZE);
	cvShowImage("Adaptive Threshold",Iat);

	cvResize(Iat, mask, 1 );
	cvNamedWindow("mask",CV_WINDOW_AUTOSIZE);
	cvShowImage("mask",Iat);
*/
match_feature(frame, RightEyePtr, EYE_WINDOW_WIDTH, EYE_WINDOW_HEIGHT, RightEye.MATCH_METHOD);

match_feature(frame, NosePtr, NOSE_WINDOW_WIDTH, NOSE_WINDOW_HEIGHT, Nose.MATCH_METHOD);

match_feature(frame, MouthPtr, MOUTH_WINDOW_WIDTH, MOUTH_WINDOW_HEIGHT, Mouth.MATCH_METHOD);


    if( (LeftEye.minval <= TH_EYE && LeftEye.MATCH_METHOD == CV_TM_SQDIFF_NORMED) ||  (LeftEye.maxval >= TH_EYE && LeftEye.MATCH_METHOD == CV_TM_CCOEFF_NORMED || LeftEye.MATCH_METHOD == CV_TM_CCORR_NORMED) ) {

    update_and_draw(frame, LeftEyePtr, &LeftEye_center, EYE_TPL_WIDTH, EYE_TPL_HEIGHT, LeftEye.MATCH_METHOD);

    } else {
        fprintf( stdout, "Lost Left Eye.\n" );
        is_tracking = 0;
    }

    if( (RightEye.minval <= TH_EYE && RightEye.MATCH_METHOD == CV_TM_SQDIFF_NORMED) ||  (RightEye.maxval >= TH_EYE && RightEye.MATCH_METHOD == CV_TM_CCOEFF_NORMED || RightEye.MATCH_METHOD == CV_TM_CCORR_NORMED) ) {

    //void update_and_draw(IplImage * frame, Feature * f, CvPoint feature_center,const int W, const int H)
    update_and_draw(frame, RightEyePtr, &RightEye_center, EYE_TPL_WIDTH, EYE_TPL_HEIGHT, RightEye.MATCH_METHOD);
    } else {
        fprintf( stdout, "Lost Right Eye.\n" );
        is_tracking = 0;
    }


    if( (Nose.minval <= TH_EYE && Nose.MATCH_METHOD == CV_TM_SQDIFF_NORMED) ||  (Nose.maxval >= TH_EYE && Nose.MATCH_METHOD == CV_TM_CCOEFF_NORMED || Nose.MATCH_METHOD == CV_TM_CCORR_NORMED) ) {
    //void update_and_draw(IplImage * frame, Feature * f, CvPoint feature_center,const int W, const int H)
    update_and_draw(frame, NosePtr, &Nose_center, NOSE_TPL_WIDTH, NOSE_TPL_HEIGHT, Nose.MATCH_METHOD);

    } else {
        fprintf( stdout, "Lost Nose.\n" );
        is_tracking = 0;
    }


    if( (Mouth.minval <= TH_EYE && Mouth.MATCH_METHOD == CV_TM_SQDIFF_NORMED) ||  (Mouth.maxval >= TH_EYE && Mouth.MATCH_METHOD == CV_TM_CCOEFF_NORMED || Mouth.MATCH_METHOD == CV_TM_CCORR_NORMED) ) {
    //void update_and_draw(IplImage * frame, Feature * f, CvPoint feature_center,const int W, const int H)
    update_and_draw(frame, MouthPtr, &Mouth_center, MOUTH_TPL_WIDTH, MOUTH_TPL_HEIGHT, Mouth.MATCH_METHOD);

    } else {
        fprintf( stdout, "Lost Mouth.\n" );
        is_tracking = 0;
    }

	//determine tracking failure
	
	//return the image back to main program
	return frame;
}


/*
    cvSetImageROI( frame, 
                   cvRect( L_C_win_x0, 
                           L_C_win_y0, 
                           EYE_WINDOW_WIDTH, 
                           EYE_WINDOW_HEIGHT ) );

    cvMatchTemplate( frame, L_C_tpl, L_C_tm, CV_TM_SQDIFF_NORMED );
    cvMinMaxLoc( L_C_tm, &L_C_minval, &L_C_maxval, &L_C_minloc, &L_C_maxloc, NULL);
    cvResetImageROI( frame );
	printf("Min value L_EYE corner		= %.3f\n",L_C_minval);


    //look for the feature in the search window
    cvSetImageROI( frame, 
                   cvRect( R_C_win_x0, 
                           R_C_win_y0, 
                           EYE_WINDOW_WIDTH, 
                           EYE_WINDOW_HEIGHT ) );
    
    cvMatchTemplate( frame, R_C_tpl, R_C_tm, CV_TM_SQDIFF_NORMED );
    cvMinMaxLoc( R_C_tm, &R_C_minval, &R_C_maxval, &R_C_minloc, &R_C_maxloc, NULL);
    cvResetImageROI( frame );
	printf("Min value R_EYE corner		= %.3f\n",R_C_minval);
*/

