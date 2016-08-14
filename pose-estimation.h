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

#ifndef POSE_ESTIMATION_H
#define POSE_ESTIMATION_H

#include <opencv2/core.hpp>

typedef struct facefeatures Face;

struct facegeometry{
  
double LeftEye_Nose_distance;
double RightEye_Nose_distance;
double LeftEye_RightEye_distance;
double Nose_Mouth_distance;

double init_LeftEye_Nose_distance;
double init_RightEye_Nose_distance;
double init_LeftEye_RightEye_distance;
double init_Nose_Mouth_distance;

double init_Mean_Feature_distance;
double Mean_Feature_distance;
  
};
typedef struct facegeometry FaceGeom;

struct pose{
  
double pitch, yaw, roll;
double slant;
double zaxis, xaxis, yaxis;


//Kalman filer
double kpitch, kyaw;
double kpitch_pre, kyaw_pre;

//float pitch[900];
//float yaw[900];
//float roll[900];
  
};
typedef struct pose Pose;



void draw_and_calculate( IplImage *img, Face *F, FaceGeom *G, Pose *P );
float	FindDistance	(CvPoint pt1, CvPoint pt2);
double	FindDistance2D32f	(CvPoint2D32f pt1, CvPoint2D32f pt2);
float	FindAngle		(CvPoint2D32f pt1, CvPoint2D32f pt2);
double	Find_slant		(int ln, int lf, float Rn, float tita);
void 	draw_crosshair(IplImage* img, CvPoint centre, int circle_radius, int line_radius, CvScalar colour);
void 	play_game(IplImage* img, int precision, CvPoint Position, CvScalar ball_colour);
void 	draw_pin(IplImage* img, CvPoint3D32f normal, float slant, float tita, CvScalar colour);
void 	init_geometric_model(Face* F,FaceGeom* G, Pose *P);
void    init_kalman_filter(void);
void 	print_text(IplImage* img, int counter, CvScalar colour);
void 	draw_trail(IplImage* img, CvPoint* pt);
void 	closeDraw();

#endif
