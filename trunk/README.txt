#######################
General Information
#######################

Implementation of the Head Pose Estimation algorithm as described in:
M. Sapienza and K.P. Camilleri, “Fasthpe: A recipe for quick head pose estimation,” 
Department of Systems & Control Engineering, University of Malta,” Technical Report, 2011.

and as seen in the YouTube video:
http://www.youtube.com/watch?v=6MfKMT-tfMs

The code is available for download at: https://sites.google.com/site/mikesapi/research

This software was developed as part of an undergraduate thesis 
within the department of 'Systems & Control Engineering (SCE)' 
at the University of Malta in 2008-2009.

Any questions about the code/dissertation can be sent to mikesapi AT gmail DOT com.



#########################
How to use (Ubuntu Linux)
#########################
Note: this software is able to run on Windows and Mac with minor tweaks.

1) Install the OPENCV Library (version 2.2 +)
2) Install SoX sound processing software to play sounds in ubuntu: "sudo apt-get install sox")
3) Open a terminal and navigate to the directory containing the code.
4) Type "make" to compile "fasthpe".

1) To run fasthpe on sample video file type "./fasthpe videos/ssm9.mpg".
2) To play game yourself from webcam stream just type "./fasthpe"



######################
How to cite
######################

@techreport{sapienza-2011,
  author       = "Michael Sapienza and Kenneth P. Camilleri",
  title        = "Fasthpe: A recipe for quick head pose estimation",
  type	       = "Technical Report",
  institution  = "Department of Systems \& Control Engineering, University of Malta",
  year 	       = "2011"
}



#########################
List of Contributors
#########################
Michael Sapienza
Kenneth Camilleri
Kenneth Scerri



#########################
ToDo
#########################
System will fail if not all the face features are visible from the webcam.

Unsatisfactory performance if the lighting conditions are not favourable. 

It currently uses an image resolution of 320x240; needs changing to work with variable camera resolution.
