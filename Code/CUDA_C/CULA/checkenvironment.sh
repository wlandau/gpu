#!/bin/sh
# Check that expected CULA environment variables are defined

check_environment_variable()
{
   var="$1"
   val='$'$1
   if [ -z  `eval "echo $val"` ]; then
      echo "Warning: $var is not defined"
      eval $2=true
   fi
}

warn=false
check_environment_variable "CULA_ROOT" warn
check_environment_variable "CULA_INC_PATH" warn
check_environment_variable "CULA_BIN_PATH_32" warn
check_environment_variable "CULA_BIN_PATH_64" warn
check_environment_variable "CULA_LIB_PATH_32" warn
check_environment_variable "CULA_LIB_PATH_64" warn

if $warn ; then
    echo ""
    echo "------------------------------------------------------------------------"
    echo "Warning: Some CULA environment variables could not be found."
    echo "         This may prevent successful building of the example projects"
    echo "------------------------------------------------------------------------"
    echo ""
fi

