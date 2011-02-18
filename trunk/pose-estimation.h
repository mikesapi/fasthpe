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

#ifndef POSE_ESTIMATION_H
#define POSE_ESTIMATION_H

IplImage* draw_and_calculate( IplImage* img, Face* F );
float	FindDistance	(CvPoint pt1, CvPoint pt2);
float	FindDistance2D32f	(CvPoint2D32f pt1, CvPoint2D32f pt2);
float	FindAngle		(CvPoint2D32f pt1, CvPoint2D32f pt2);
float	Find_slant		(int ln, int lf, float Rn, float tita);
void 	draw_crosshair(IplImage* img, CvPoint centre, int circle_radius, int line_radius, CvScalar colour);
void 	play_game(IplImage* img, int precision, CvPoint Position, CvScalar ball_colour);
void 	draw_pin(IplImage* img, CvPoint3D32f normal, float slant, float tita, CvScalar colour);
void    init_geometric_model2(Face* F);
void    init_kalman_filter(void);
void 	print_text(IplImage* img, int counter, CvScalar colour);
void 	draw_trail(IplImage* img, CvPoint* pt);
void 	closeDraw();

#endif