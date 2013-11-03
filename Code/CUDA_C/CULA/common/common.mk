# Common Makefile for cula projects

CC=gcc
ifdef debug
CFLAGS=-g
else
CFLAGS=-DNDEBUG -O3
endif

