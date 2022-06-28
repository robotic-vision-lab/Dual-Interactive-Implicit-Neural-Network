#!/bin/bash
  
TURQ_REGEXP="^(ba|cp|gr|ko|sn)-fe[1-9]?(\.lanl\.gov)?$"
  
if [[ $1 =~ $TURQ_REGEXP ]]; then
  
  exec ssh wtrw2.lanl.gov ssh "$@"
  
else
  exec ssh "$@"
fi