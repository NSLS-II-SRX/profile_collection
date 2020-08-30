#!/bin/bash

# The beamline-specific metapackages are not used since 2020-2.0 deployment.
# conda install -y -c ${CONDA_CHANNEL_NAME} <package>

sudo mkdir -v -p /nsls2/xf05id1/shared/config/ \
                 /nsls2/xf05id1/experiments/ \
                 /nsls2/xf05id1/shared/src/bluesky_scripts/

sudo chmod -Rv go+rw /nsls2/xf05id1/

touch /nsls2/xf05id1/shared/src/bluesky_scripts/simple_batch.py
touch /nsls2/xf05id1/shared/src/bluesky_scripts/fly_batch.py
