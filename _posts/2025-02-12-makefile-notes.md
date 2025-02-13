---
title: "Makefile Basics"
categories:
  - Blog
tags:
  - makefile
---

Some notes for basic makefile usage.

## Basic Grammar
```
target1 target2: prereq1 prereq2
  command1
  command2
'#': To make comments
'\': Use as escape character to separate long command
```
- Commands have tab indent
- Prerequisite are updated by its own rules first
- If prerequisite is newer than target, target will remake
- commands are put in shell and each command has its own subshell

## Rules
### Explicit Rules
Most of the rules in makefile are explicit rules, following is an example:
```
target1 target2: prereq1
# This is equivalent to the following:
target1: prereq1
target2: prereq1
```

### Wildcards
**Frequently used symbols:**

| Symbol  | Usage                   |
|---------|-------------------------|
|   ~     | Home directory          |
|   *     | Expansion               |
|   ?     | Single character        |
| [...]   | Character class         |
| [^...]  | Negated character class |

### Phony Targets
Sometimes we may want the target to be operation instead of files. In order to do this, we introduce phony target, a classic example would be ***clean***:
```
.PHONY: clean
clean:
    rm -f *.o
``` 
Declaring phony target will have the following effects:  
- It avoids the naming issue that source file might have the same name as the operation, the command in makefile will always be executed.
- The status of the target will always *out of date*, this tells *make* this target is not made with source files.

Below are some standard phony targets:

| Symbol   | Usage                                                                           |
|----------|---------------------------------------------------------------------------------|
|all       | Perform all tasks to build the application                                      |
|install   | Create an installation of the application from compiled binaries                |
|clean     | Delete the binary files generated from sources                                  |
|distclean | Delete all the generated files that were not in the original source distribution|
|TAGS      | Create a tags table for use by editors                                          |
|info      | Create GNU info files from their Texinfo sources                                |
|check     | Run any tests associated with this application                                  |


*Phony targets* can be viewed as embedded shell scripts inside makefile. It can be used to print debug info, makefile is written in a **top-down** manner but executed in **bottom-up** manner. Since phony target is always *out of date*, it will be updated and executed first as the prerequisite. We can utilize this feature to print logs we need. 

### Variables
Below is the typical grammar for using variable:
```
$(variable-name) # You can use either $() or ${} for variable
$@               # You don't need to add () for single character
```

*Automatic variables* are often used to provide access to elements from the target and prerequisite without having to explicitly specify the filenames.

| Symbol | Usage                                                                    |
|--------|--------------------------------------------------------------------------|
|$@      | The filename representing the target                                     |
|$%      | The filename element of an archive member specification                  |
|$<      | The filename of the first prerequisite                                   |
|$?      | The names of all prerequisites that are newer than the target            |
|$^      | The filenames of all the prerequisites, has duplicated filenames removed |
|$+      | Similar to $^ but duplicate filenames are included                       |
|$*      | The stem of the target filename, typically without its suffix            |


### VPATH and vpath
*VPATH* can be used to add search paths for source files. The default *makefile* search path only includes the current directory. Any file need to be searched will search in *VPATH*. *vpath* on the other hand is more precise. See below for the example:
```
# VPATH
VPATH = SRC # Add SRC to search path

# vpath
# vpath pattern directory-list
vpath %.c src      # search .c file in src directory
vpath %.h include  # search header file in include directory
```

### Variable types
Variables can be classified as either simply expanded or recursively expanded:
```
MAKE_DEPEND := $(CC) -M # simple expanded, expanded at once
MAKE_DEPEND = $(CC) -M  # recursively expanded, expanded at use

...
# some time later
CC = gcc
```
In the example above, simply expanded *$(CC)* would become " -M" as *CC* was not defined until later. However, the recursively expanded variable will become "gcc -M" because CC is defined to gcc when used later.

We also have the following two assignment operators:
- *?=*: Assign values if the variable has not been assigned
- *+=*: Append to the variable, mostly used for appending paths, targets or prerequisites

## Reference
> <cite>*Managing Projects with GNU Make*<cite>

