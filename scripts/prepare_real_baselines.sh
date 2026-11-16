#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
mkdir -p outputs/baselines
Rscript --vanilla scripts/export_reference.R
for model in binary ordinal dynamic hierarchical network poisson; do
    Rscript --vanilla "scripts/export_${model}_baseline.R"
done
