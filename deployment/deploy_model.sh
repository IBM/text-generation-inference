#!/bin/bash

if [ -z "$1" ] || [ ! -d "models/$1" ]; then
  echo "Must provide valid model name"
  exit 1
fi

kustomize build --load-restrictor LoadRestrictionsNone "models/$1" | kubectl apply -f -
