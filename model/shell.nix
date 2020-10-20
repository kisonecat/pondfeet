let
  pkgs = import <nixpkgs> {};
in
  pkgs.mkShell {
    name = "simpleEnv";
    buildInputs = with pkgs; [
    # basic python dependencies
    python37
    python37Packages.pip
      python37Packages.numpy
      python37Packages.scikitlearn
      python37Packages.scipy
      python37Packages.matplotlib
    # a couple of deep learning libraries
      #python37Packages.tensorflowWithCuda # note if you get rid of WithCuda then you will not be using Cuda
      #python37Packages.Keras
      python37Packages.pytorch
    ];
   shellHook = ''
            alias pip="PIP_PREFIX='$(pwd)/_build/pip_packages' \pip"
            export PYTHONPATH="$(pwd)/_build/pip_packages/lib/python3.7/site-packages:$PYTHONPATH"
            unset SOURCE_DATE_EPOCH
      '';
  }

 
