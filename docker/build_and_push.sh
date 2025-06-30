#!/bin/bash

# Usage: ./script.sh [list of images]
# Example: ./script.sh latex python

# Available images
IMAGES=("latex" "python" "ros")

# If no arguments provided, build all images
if [ $# -eq 0 ]; then
    TARGETS=("${IMAGES[@]}")
else
    TARGETS=("$@")
fi

for target in "${TARGETS[@]}"; do
    case "$target" in
        latex)
            docker build -t sebsti1/mt_latex -f docker/latex.Dockerfile .
            ;;
        python)
            docker build -t sebsti1/mt_python -f docker/python.Dockerfile .
            ;;
        ros)
            docker build -t sebsti1/mt_ros -f docker/ros.Dockerfile .
            ;;
        *)
            echo "Unknown image: $target"
            echo "Available images: ${IMAGES[*]}"
            exit 1
            ;;
    esac
done

if [[ -n "$PUSH" ]]; then
    for target in "${TARGETS[@]}"; do
        case "$target" in
            latex)
                docker push sebsti1/mt_latex
                ;;
            python)
                docker push sebsti1/mt_python
                ;;
            ros)
                docker push sebsti1/mt_ros
                ;;
        esac
    done
fi