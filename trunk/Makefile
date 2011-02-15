CC = g++

CFLAGS = -c -g -Wall -I/usr/local/include/opencv -I/usr/local/include
#CFLAGS = -c -g -Wall $(shell pkg-config opencv --cflags)



#OPENCV<2.2#LDFLAGS = -L/usr/local/lib -lcxcore -lcv -lhighgui -lcvaux -lml

LDFLAGS = -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_calib3d -lopencv_video -lopencv_features2d -lopencv_ml -lopencv_highgui -lopencv_objdetect -lopencv_contrib -lopencv_legacy
#LDFLAGS = -L/usr/local/lib $(shell pkg-config opencv --libs)

#-L../../lib -lflycapture${D}
#-ldc1394 -lraw1394

SOURCES = main.cpp draw_and_calculate.cpp Template_match.cpp capture.cpp Feature_detect.cpp
OBJECTS = $(SOURCES:.c=.o)
TARGET = Head_Pose_Estimation

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) $(LDFLAGS) -o $@

.c.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -rf *.o $(TARGET)

#Notes
#g++ `pkg-config opencv --cflags --libs` source_name.c -o source_name
# -I../../include -I/usr/include/flycapture
