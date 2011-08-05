Implementation of the Head Pose Estimation algorithm as described in:

M. Sapienza. Real-time head pose estimation in the 6 degrees of freedom. 
Undergraduate thesis, University of Malta, 2009.

This software was developed by Michael Sapienza within the department of 'Systems & Control Engineering (SCE)' at the University of Malta under the supervision of Prof. Kenneth Camilleri.



What to do to run this software (under Linux):
1) Install the OPENCV Library.
2) Connect your webcam.
3) Open a terminal and navigate to the directory containing the code.
3) Type "make" to compile "Head_Pose_Estimation".
4) To run type "./Head_Pose_Estimation".



Contributers:
Michael Sapienza
Kenneth Camilleri
Kenneth Scerri

To Do:
System will fail if not all the face features are visible from the webcam.
Unsatisfactory performance if the lighting conditions are not favorable. 
It currently uses an image resolution of 320x240; needs changing to work with variable camera resolution.
