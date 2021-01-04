SHELL = /bin/sh

CC = gcc
CHMOD = chmod
ECHO = echo
RM = rm -f
TAR = tar
MKDIR = mkdir
CP = rsync -R

DEBUG ?= 0
TEST = $(shell n=0; while [[ $n -lt 1000 ]]; do ./som iris.data; n=$((n+1)); done)

CFLAGS = -Wall -O3

PROGNAME = som
FILENAME = iris.data
CONFIGF = som.cfg
README = README.md
distdir = $(PROGNAME)
HEADERS = som.h parser.h ll.h config.h mnist.h matrix.h
SOURCES = main.c som.c parser.c ll.c matrix.c
OBJ = $(SOURCES:.c=.o)

DOXYFILE = documentation/Doxyfile
DISTFILES = $(SOURCES) Makefile $(HEADERS) $(DOXYFILE) $(FILENAME) $(CONFIGF) $(README)

all: $(PROGNAME)

ifeq ($(DEBUG), 1) 
    CFLAGS += -DDEBUG
endif

$(PROGNAME): $(OBJ)
	$(CC) $(OBJ) -o $(PROGNAME)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

dist: distdir
	$(CHMOD) -R a+r $(distdir)
	$(TAR) zcvf $(distdir).tgz $(distdir)
	$(RM) -r $(distdir)

distdir: $(DISTFILES)
	$(RM) -r $(distdir)
	$(MKDIR) $(distdir)
	$(CHMOD) 777 $(distdir)
	$(CP) $(DISTFILES) $(distdir)

doc: $(DOXYFILE)
	cd documentation && doxygen && cd ..

test:
	for number in {1..1000} ; do \
	    ./som $(FILENAME) ; \
	done

clean:
	@$(RM) -r $(PROGNAME) $(OBJ) *~ $(distdir).tgz documentation/*~ documentation/html