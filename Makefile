DEBUG=1
CC=g++
CFLAGS=-Wall -Wfatal-errors
OPTS=-Ofast
ifeq ($(DEBUG), 1)
OPTS=-O0 -g
endif
CFLAGS+=$(OPTS)
INCLUDES=-I/usr/local/include/opencv2
LDFLAGS = -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_calib3d -lopencv_video -lopencv_videoio -lopencv_features2d -lopencv_ml -lopencv_highgui -lopencv_objdetect -lXrandr -lX11

SOURCES = main.cpp facefeaturedetect.cpp facefeaturetrack.cpp pose-estimation.cpp capture.cpp 
OBJECTS = $(SOURCES:.c=.o)
TARGET = fasthpe

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) $(LDFLAGS) -o $@

.c.o:
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -rf *.o $(TARGET)
