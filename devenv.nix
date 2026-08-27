{ config, pkgs, inputs,... }:

{
  packages = with pkgs; [
    curl
    ffmpeg
    git
    tmux
  ];

  languages.python = {
    enable = true;
    version = "3.11";
    venv.enable = true;
    manylinux.enable = true;
    libraries = with pkgs; [
      stdenv.cc.cc.lib
      zlib
      glib
      libGL
      libxcrypt
      portaudio
    ];
    uv = {
      enable = true;
      sync = {
        enable = true;
        arguments = [ "--frozen" ];
      };
    };
  };

  env = {
    TF_USE_LEGACY_KERAS = "1";
    TF_FORCE_GPU_ALLOW_GROWTH = "true";
    KERAS_HOME = "${config.env.DEVENV_STATE}/keras";
    FACEDANCER_ROOT = config.devenv.root;
  };

  files."arcface_model/ArcFace-Res50.h5".source = inputs.arcface;
  files."expressionembedder_model/ExpressionEmbedder-B0.h5".source = inputs.expressionembedder;
  files."retinaface/RetinaFace-Res50.h5".source = inputs.retinaface;
  files."assets/dataset/hands".source = inputs.hands;
  files."assets/dataset/celeb".source = inputs.celeb;

  enterShell = ''
    # VIRTUAL_ENV is exported after enterShell by devenv, but the managed
    # environment always lives at this state path.
    wheel_site_packages="$DEVENV_STATE/venv/lib/python3.11/site-packages"
    wheel_library_path=""

    for library_dir in "$wheel_site_packages"/nvidia/*/lib; do
      if [ -d "$library_dir" ]; then
        wheel_library_path="''${wheel_library_path:+$wheel_library_path:}$library_dir"
      fi
    done

    export LD_LIBRARY_PATH="/run/opengl-driver/lib''${wheel_library_path:+:$wheel_library_path}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

    cuda_nvcc_root="$wheel_site_packages/nvidia/cuda_nvcc"
    if [ -x "$cuda_nvcc_root/bin/ptxas" ]; then
      export PATH="$cuda_nvcc_root/bin:$PATH"
      export XLA_FLAGS="--xla_gpu_cuda_data_dir=$cuda_nvcc_root''${XLA_FLAGS:+ $XLA_FLAGS}"
    fi

    mkdir -p "$KERAS_HOME"
  '';

  scripts.fd-check-env.exec = ''
    exec python "$FACEDANCER_ROOT/scripts/check_environment.py"
  '';
}
