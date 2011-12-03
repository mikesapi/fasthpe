CC = g++

CFLAGS = -g -Wall -I/usr/local/include/opencv -I/usr/local/include
#CFLAGS = -c -g -Wall $(shell pkg-config opencv --cflags)

#OPENCV<2.1#LDFLAGS = -L/usr/local/lib -lcxcore -lcv -lhighgui -lcvaux -lml

LDFLAGS = -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_calib3d -lopencv_video -lopencv_features2d -lopencv_ml -lopencv_highgui -lopencv_objdetect -lopencv_contrib -lopencv_legacy
#LDFLAGS = -L/usr/local/lib $(shell pkg-config opencv --libs)

SOURCES = main.cpp facefeaturedetect.cpp facefeaturetrack.cpp pose-estimation.cpp capture.cpp 
OBJECTS = $(SOURCES:.c=.o)
TARGET = fasthpe

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) $(LDFLAGS) -o $@

.c.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -rf *.o $(TARGET)
