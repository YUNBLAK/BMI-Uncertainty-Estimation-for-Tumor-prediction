#!/bin/bash

# Define the folder path
FOLDER_PATH="output"

# Check if the folder exists
if [ -d "$FOLDER_PATH" ]; then
    # Delete the folder and its contents
    rm -rf "$FOLDER_PATH"
    echo "Deleted the folder: $FOLDER_PATH"
else
    echo "Folder does not exist: $FOLDER_PATH"
fi

# Recreate the folder
mkdir -p "$FOLDER_PATH"
echo "Recreated the folder: $FOLDER_PATH"


FOLDER_PATH="prediction_values"

# Check if the folder exists
if [ -d "$FOLDER_PATH" ]; then
    # Delete the folder and its contents
    rm -rf "$FOLDER_PATH"
    echo "Deleted the folder: $FOLDER_PATH"
else
    echo "Folder does not exist: $FOLDER_PATH"
fi

# Recreate the folder
mkdir -p "$FOLDER_PATH"
echo "Recreated the folder: $FOLDER_PATH"


FOLDER_PATH="prediction_masks"

# Check if the folder exists
if [ -d "$FOLDER_PATH" ]; then
    # Delete the folder and its contents
    rm -rf "$FOLDER_PATH"
    echo "Deleted the folder: $FOLDER_PATH"
else
    echo "Folder does not exist: $FOLDER_PATH"
fi

# Recreate the folder
mkdir -p "$FOLDER_PATH"
echo "Recreated the folder: $FOLDER_PATH"



FOLDER_PATH="logs_pred"

# Check if the folder exists
if [ -d "$FOLDER_PATH" ]; then
    # Delete the folder and its contents
    rm -rf "$FOLDER_PATH"
    echo "Deleted the folder: $FOLDER_PATH"
else
    echo "Folder does not exist: $FOLDER_PATH"
fi

# Recreate the folder
mkdir -p "$FOLDER_PATH"
echo "Recreated the folder: $FOLDER_PATH"



FOLDER_PATH="logs"

# Check if the folder exists
if [ -d "$FOLDER_PATH" ]; then
    # Delete the folder and its contents
    rm -rf "$FOLDER_PATH"
    echo "Deleted the folder: $FOLDER_PATH"
else
    echo "Folder does not exist: $FOLDER_PATH"
fi

# Recreate the folder
mkdir -p "$FOLDER_PATH"
echo "Recreated the folder: $FOLDER_PATH"


