SHELL = /bin/sh

CC = gcc
CHMOD = chmod
ECHO = echo
RM = rm -f
TAR = tar
MKDIR = mkdir
CP = rsync -R

CFLAGS = -Wall -O3

PROGNAME = gan
FILENAME = iris.data
CONFIGF = gan.cfg
README = README.md
STATIC = libgan.a
distdir = $(PROGNAME)
HEADERS = matrix.h config.h mnist.h matrix.h mnist.h gan.h
SOURCES = main.c matrix.c mnist.c config.c gan.c
OBJ = $(SOURCES:.c=.o)

DOXYFILE = documentation/Doxyfile
DISTFILES = $(SOURCES) Makefile $(HEADERS) $(DOXYFILE) $(FILENAME) $(CONFIGF) $(README)

all: $(PROGNAME)


$(PROGNAME): $(OBJ)
	$(CC) $(OBJ) -o $(PROGNAME)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

libs: $(STATIC)

$(STATIC): $(OBJ)
	ar rcs $@ $^

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

clean:
	@$(RM) -r $(PROGNAME) $(OBJ) *~ $(distdir).tgz documentation/*~ documentation/html
