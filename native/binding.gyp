{
  "targets": [{
    "target_name": "speex_aec",
    "cflags!": [ "-fno-exceptions" ],
    "cflags_cc!": [ "-fno-exceptions" ],
    "sources": [ "speex_aec.cc" ],
    "include_dirs": [
      "<!@(node -p \"require('node-addon-api').include\")",
      "/opt/homebrew/include"
    ],
    "libraries": [
      "-L/opt/homebrew/lib",
      "-lspeexdsp"
    ],
    "defines": [ "NAPI_DISABLE_CPP_EXCEPTIONS" ],
    "conditions": [
      ["OS=='mac'", {
        "xcode_settings": {
          "GCC_ENABLE_CPP_EXCEPTIONS": "YES",
          "CLANG_CXX_LIBRARY": "libc++",
          "MACOSX_DEPLOYMENT_TARGET": "10.15"
        }
      }]
    ]
  }]
}


