#!/bin/bash

set -x

# obligatory docker/singularity container namings
export PROJECT="mesh_diffusion"
export SIMAGES_DIR=/gpfs/gpfs0/3ddl/singularity-images
export IMAGE_NAME="elyaishere/${PROJECT}"
export IMAGE_VERSION="17022024"
export IMAGE_NAME_TAG="${IMAGE_NAME}:${IMAGE_VERSION}"
export SIMAGE_FILENAME="${SIMAGES_DIR}/$(echo ${IMAGE_NAME_TAG} | tr /: _).sif"
